from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from vsrx.app.options import PipelineResult, ProcessOptions
from vsrx.cleanplate import TemporalCleanPlateReconstructor
from vsrx.detection import CascadedTextDetector, DiscoverySampler, ROIDiscoverer
from vsrx.domain.contracts import (
    CleanPlateResult,
    InpaintRequest,
    MaskFrame,
    QualityReport,
    RouteDecision,
    Segment,
    Shot,
    SubtitleTrack,
    TextDetection,
    VideoFrame,
    jsonable,
)
from vsrx.domain.enums import JobState, Route, TrackClassification
from vsrx.domain.errors import CancelledError, VSRXError
from vsrx.inpaint import InpaintRegistry
from vsrx.mask import ExternalMaskProvider, ProbabilityMaskGenerator
from vsrx.media.decode import PyAVFrameReader
from vsrx.media.encode import FinalEncoder, write_json_atomic
from vsrx.media.probe import FFProbeAdapter
from vsrx.media.stream_map import SoftSubtitleHandler
from vsrx.quality import AutomaticQualityGate, RetryPlanner
from vsrx.reporting import JobAudit, SegmentAudit
from vsrx.routing import AdaptiveRouter, VramCalibrator, build_segment_features
from vsrx.scene import AdaptiveSceneDetector
from vsrx.scheduler import GpuLease, JobRepository, SegmentCheckpointStore
from vsrx.tracking import TrackPipeline
from vsrx.utils.config import Config
from vsrx.utils.geometry import bbox_iou, expand_bbox, mask_bbox, union_bboxes
from vsrx.utils.hash import sha256_file, stable_json_hash

logger = logging.getLogger(__name__)


class VSRXPipeline:
    def __init__(self, config: Config, *, model_manifest_path: Path | None = None) -> None:
        self.config = config
        self.prober = FFProbeAdapter(config)
        self.scene_detector = AdaptiveSceneDetector(config)
        self.detector = CascadedTextDetector(config)
        self.roi_discoverer = ROIDiscoverer(config)
        self.sampler = DiscoverySampler(config)
        self.track_pipeline = TrackPipeline(config)
        self.mask_generator = ProbabilityMaskGenerator(config)
        self.clean_plate = TemporalCleanPlateReconstructor(config)
        self.router = AdaptiveRouter(config)
        self._device_index = 0
        self.inpainters = InpaintRegistry(config, device_index=self._device_index)
        self.quality_gate = AutomaticQualityGate(config, detector=self.detector)
        self.retry_planner = RetryPlanner(config)
        self.soft_subtitles = SoftSubtitleHandler(config)
        self.final_encoder = FinalEncoder(config)
        self.model_manifest_path = model_manifest_path
        self._model_file_hash_cache: dict[tuple[str, int, int], str] = {}

    @staticmethod
    def _resolve_path(path: Path | str, base: Path | None = None) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = (base or Path.cwd()) / candidate
        return candidate.resolve()

    def _work_dir(self, options: ProcessOptions) -> Path:

        value = options.work_dir or Path(str(self.config.get("runtime.work_dir", "./work")))
        return self._resolve_path(value)

    def _output_path(self, input_path: Path, options: ProcessOptions) -> Path:
        if options.output_path:
            return self._resolve_path(options.output_path)
        configured = self._resolve_path(
            Path(str(self.config.get("runtime.output_dir", "./output")))
        )
        configured.mkdir(parents=True, exist_ok=True)
        extension = (
            input_path.suffix
            if input_path.suffix.lower() in {".mkv", ".mp4", ".mov", ".webm"}
            else ".mkv"
        )
        return configured / f"{input_path.stem}.subtitle_removed{extension}"

    def _cached_file_identity(self, path: Path) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return {"path": str(resolved), "present": False}
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        digest = self._model_file_hash_cache.get(key)
        if digest is None:
            digest = sha256_file(resolved)
            self._model_file_hash_cache[key] = digest
        return {
            "path": str(resolved),
            "present": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }

    def _repository_identity(self, path: Path) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            return {"path": str(resolved), "present": False}
        commit: str | None = None
        try:
            commit = subprocess.run(
                ["git", "-C", str(resolved), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except Exception:
            commit = None
        weight_files: list[dict[str, Any]] = []
        for folder_name in ("weights", "checkpoints", "models"):
            folder = resolved / folder_name
            if not folder.is_dir():
                continue
            for item in sorted(folder.rglob("*")):
                if item.is_file() and item.suffix.lower() in {
                    ".pth",
                    ".pt",
                    ".ckpt",
                    ".onnx",
                    ".safetensors",
                }:
                    identity = self._cached_file_identity(item)
                    identity["relative"] = str(item.relative_to(resolved))
                    weight_files.append(identity)
        return {
            "path": str(resolved),
            "present": True,
            "git_commit": commit,
            "weights": weight_files,
        }

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def _manifest_hash(self) -> str:
        lama = Path(
            os.environ.get("VSRX_LAMA_MODEL", "")
            or self.config.get("spatial_inpainting.lama.model_path")
            or "models/lama.onnx"
        )
        migan = Path(
            os.environ.get("VSRX_MIGAN_MODEL", "")
            or self.config.get("spatial_inpainting.migan.model_path")
            or "models/migan.onnx"
        )
        propainter = Path(
            os.environ.get("VSRX_PROPAINTER_REPO", "")
            or self.config.get("video_inpainting.propainter.repo_path")
            or "third_party/ProPainter"
        )
        sttn_repo = Path(
            os.environ.get("VSRX_STTN_REPO", "")
            or self.config.get("video_inpainting.sttn.repo_path")
            or "third_party/STTN"
        )
        sttn_checkpoint = Path(
            os.environ.get("VSRX_STTN_CHECKPOINT", "")
            or self.config.get("video_inpainting.sttn.checkpoint")
            or sttn_repo / "checkpoints/sttn.pth"
        )
        payload = {
            "schema": 2,
            "static_manifest": (
                self._cached_file_identity(self.model_manifest_path)
                if self.model_manifest_path is not None
                else None
            ),
            "runtime_packages": {
                name: self._package_version(name)
                for name in (
                    "rapidocr",
                    "onnxruntime",
                    "openvino",
                    "opencv-python-headless",
                    "torch",
                )
            },
            "models": {
                "lama": self._cached_file_identity(lama),
                "migan": self._cached_file_identity(migan),
                "propainter": self._repository_identity(propainter),
                "sttn_repository": self._repository_identity(sttn_repo),
                "sttn_checkpoint": self._cached_file_identity(sttn_checkpoint),
            },
            "availability": self.inpainters.model_status(),
        }
        return stable_json_hash(payload)

    def _ensure_device(self, device_index: int) -> None:
        """Bind device-aware model adapters to the requested GPU exactly once."""

        normalized = max(0, int(device_index))
        if normalized == self._device_index:
            return
        self._device_index = normalized
        self.inpainters = InpaintRegistry(self.config, device_index=normalized)

    def _execution_hash(self, options: ProcessOptions, external_mask_digest: str | None) -> str:
        """Hash every option that can alter processed pixels or subtitle policy."""

        return stable_json_hash(
            {
                "schema": 2,
                "base_config_hash": self.config.hash,
                "fixed_rois": [list(item) for item in sorted(options.fixed_rois)],
                "external_mask_digest": external_mask_digest,
                "force_hard_subtitle_scan": options.force_hard_subtitle_scan,
                "aggressive_uncertain_removal": options.aggressive_uncertain_removal,
            }
        )

    @contextmanager
    def _timed(self, audit: JobAudit, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            audit.stage_timings[stage] = audit.stage_timings.get(stage, 0.0) + (
                time.perf_counter() - started
            )

    @staticmethod
    def _deduplicate_detections(detections: Sequence[TextDetection]) -> list[TextDetection]:
        result: list[TextDetection] = []
        for item in sorted(
            detections, key=lambda value: (value.frame.frame_index, -value.confidence)
        ):
            duplicate = False
            for existing in result:
                if (
                    existing.frame.frame_index == item.frame.frame_index
                    and bbox_iou(existing.bbox_xyxy, item.bbox_xyxy) >= 0.72
                ):
                    duplicate = True
                    break
            if not duplicate:
                result.append(item)
        return result

    @staticmethod
    def _shot_for_pts(shots: Sequence[Shot], pts_us: int) -> Shot:
        for shot in shots:
            if shot.start_pts_us <= pts_us < shot.end_pts_us:
                return shot
        return shots[-1]

    def _discover_rois(
        self,
        probe,
        shots: Sequence[Shot],
        reader: PyAVFrameReader,
        fixed_rois: Sequence[tuple[int, int, int, int]],
    ) -> tuple[list[tuple[int, int, int, int]], list[TextDetection]]:
        if fixed_rois:
            return list(fixed_rois), []
        sample_pts = self.sampler.sample_pts(probe, list(shots))
        # Cap global discovery to keep very long files bounded. Periodic local
        # rescans during processing cover positions not represented here.
        if len(sample_pts) > 180:
            indices = np.linspace(0, len(sample_pts) - 1, 180).round().astype(int)
            sample_pts = [sample_pts[index] for index in indices]
        samples = reader.sample_at_pts(sample_pts, shot_id=0)
        detections = self.detector.detect(
            samples, None, tier=str(self.config.get("ocr.default_tier", "small"))
        )
        rois = self.roi_discoverer.discover(detections, probe.width, probe.height)
        # Always retain a bottom-band safety ROI when capacity allows and no
        # discovered region already covers it.
        bottom_start = int(
            probe.height
            * (1.0 - float(self.config.get("subtitle_discovery.default_bottom_band_hint", 0.38)))
        )
        bottom = (0, bottom_start, probe.width, probe.height)
        if not any(
            box[3] >= probe.height * 0.90 and box[1] <= bottom_start + probe.height * 0.08
            for box in rois
        ) and len(rois) < int(self.config.get("subtitle_discovery.max_rois", 4)):
            rois.append(bottom)
        return rois, detections

    def _chunk_ranges(self, shot: Shot, fps: float) -> list[tuple[int, int, int]]:
        configured_max = int(self.config.get("routing.segment_max_frames", 96))
        practical_max = int(self.config.get("video_inpainting.propainter.default_chunk_frames", 32))
        frame_count = max(8, min(configured_max, practical_max))
        duration = max(1, int(round(frame_count / max(fps, 1e-6) * 1_000_000)))
        ranges: list[tuple[int, int, int]] = []
        start = shot.start_pts_us
        ordinal = 0
        while start < shot.end_pts_us:
            end = min(shot.end_pts_us, start + duration)
            if end <= start:
                break
            ranges.append((ordinal, start, end))
            start = end
            ordinal += 1
        return ranges or [(0, shot.start_pts_us, shot.end_pts_us)]

    def _local_rois_with_rescan(
        self,
        frames: Sequence[VideoFrame],
        base_rois: Sequence[tuple[int, int, int, int]],
        chunk_ordinal: int,
    ) -> tuple[list[tuple[int, int, int, int]], list[TextDetection]]:
        rois = list(base_rois)
        rescan_seconds = float(
            self.config.get("subtitle_discovery.periodic_full_frame_rescan_seconds", 8.0)
        )
        chunk_seconds = (
            max(0.25, (frames[-1].ref.pts_us - frames[0].ref.pts_us) / 1_000_000.0)
            if len(frames) > 1
            else 1.0
        )
        interval_chunks = max(1, int(round(rescan_seconds / chunk_seconds)))
        extra: list[TextDetection] = []
        if chunk_ordinal % interval_chunks == 0 and frames:
            sample = [frames[len(frames) // 2]]
            extra = self.detector.detect(
                sample, None, tier=str(self.config.get("ocr.default_tier", "small"))
            )
            height, width = frames[0].image_bgr.shape[:2]
            padding = max(
                12,
                int(
                    min(width, height)
                    * float(self.config.get("subtitle_discovery.roi_padding_ratio", 0.025))
                ),
            )
            for detection in extra:
                if any(bbox_iou(detection.bbox_xyxy, roi) > 0.08 for roi in rois):
                    continue
                rois.append(expand_bbox(detection.bbox_xyxy, padding, width, height))
                if len(rois) >= int(self.config.get("subtitle_discovery.max_rois", 4)):
                    break
        return rois, extra

    def _build_tracks(
        self,
        frames: list[VideoFrame],
        shot: Shot,
        rois: Sequence[tuple[int, int, int, int]],
        chunk_ordinal: int,
    ) -> tuple[list[SubtitleTrack], list[TextDetection]]:
        if not frames:
            return [], []
        local_rois, full_scan = self._local_rois_with_rescan(frames, rois, chunk_ordinal)
        stride = max(1, int(self.config.get("tracking.detection_stride_stable", 4)))
        sample_indices = sorted({0, len(frames) - 1, *range(0, len(frames), stride)})
        sample_frames = [frames[index] for index in sample_indices]
        tier = str(self.config.get("ocr.default_tier", "small"))
        detections = [*full_scan, *self.detector.detect(sample_frames, local_rois, tier=tier)]
        detections = self._deduplicate_detections(detections)
        tracks = self.track_pipeline.build(frames, detections, shot)

        removable = [
            item
            for item in tracks
            if item.classification in {TrackClassification.SUBTITLE, TrackClassification.OVERLAY}
        ]
        unstable = [
            item
            for item in tracks
            if item.is_karaoke
            or item.is_moving
            or item.classification == TrackClassification.UNCERTAIN
        ]
        if unstable:
            unstable_boxes = [item.roi_xyxy for item in unstable]
            union = union_bboxes(unstable_boxes)
            if union is not None:
                detailed = self.detector.detect(frames, [union], tier=tier)
                detections = self._deduplicate_detections([*detections, *detailed])
                tracks = self.track_pipeline.build(frames, detections, shot)
                removable = [
                    item
                    for item in tracks
                    if item.classification
                    in {TrackClassification.SUBTITLE, TrackClassification.OVERLAY}
                ]

        # Localized medium retry prevents short/stylized captions from silently
        # disappearing merely because the fast detector found too few samples.
        if not removable:
            retry_tier = str(self.config.get("ocr.retry_tier", "medium"))
            retry_stride = 1 if len(frames) <= 48 else 2
            retry_frames = frames[::retry_stride]
            retry = self.detector.detect(retry_frames, local_rois, tier=retry_tier)
            if retry:
                detections = self._deduplicate_detections([*detections, *retry])
                tracks = self.track_pipeline.build(frames, detections, shot)
        return tracks, detections

    @staticmethod
    def _force_decision(decision: RouteDecision, route_name: str | None) -> RouteDecision:
        if not route_name:
            return decision
        try:
            route = Route(route_name)
        except ValueError:
            return decision
        return RouteDecision(
            route,
            (*decision.reason_codes, f"retry_forced_{route.value}"),
            decision.features,
            decision.parameters,
        )

    def _inpaint_roi(
        self, masks: Sequence[np.ndarray], height: int, width: int, route: Route
    ) -> tuple[int, int, int, int]:
        boxes = [mask_bbox(mask) for mask in masks]
        bbox = union_bboxes(item for item in boxes if item is not None)
        if bbox is None:
            return (0, 0, width, height)
        if route in {Route.OFFICIAL_PROPAINTER, Route.STTN_FALLBACK}:
            configured = int(
                self.config.get("video_inpainting.propainter.roi_padding_px_1080p", 72)
            )
        else:
            configured = int(self.config.get("spatial_inpainting.lama.roi_padding_px_1080p", 64))
        padding = max(16, int(round(configured * height / 1080.0)))
        return expand_bbox(bbox, padding, width, height)

    @staticmethod
    def _quality_penalty(report: QualityReport) -> float:
        penalty = 0.0
        for metric in report.metrics:
            if metric.passed:
                continue
            scale = max(abs(metric.threshold), 1e-3)
            penalty += 1.0 + min(10.0, abs(metric.value - metric.threshold) / scale)
        return penalty

    def _process_segment(
        self,
        *,
        job_id: str,
        repository: JobRepository,
        checkpoint_store: SegmentCheckpointStore,
        work_dir: Path,
        probe,
        shot: Shot,
        chunk_ordinal: int,
        core_start_us: int,
        core_end_us: int,
        global_rois: Sequence[tuple[int, int, int, int]],
        options: ProcessOptions,
        audit: JobAudit,
        vram_calibrator: VramCalibrator,
        external_masks: ExternalMaskProvider | None = None,
    ) -> tuple[str, bool, bool]:
        context_seconds = float(
            self.config.get("clean_plate.reference_window_seconds.default_each_side", 0.9)
        )
        context_us = int(context_seconds * 1_000_000)
        context_start = max(shot.start_pts_us, core_start_us - context_us)
        context_end = min(shot.end_pts_us, core_end_us + context_us)
        start_frame_index = max(
            shot.start_frame_index, int(round(context_start / 1_000_000 * probe.fps))
        )
        reader = PyAVFrameReader(probe)
        frames = reader.read_range(
            context_start, context_end, shot_id=shot.shot_id, start_frame_index=start_frame_index
        )
        core_indices = [
            index
            for index, frame in enumerate(frames)
            if core_start_us <= frame.ref.pts_us < core_end_us
        ]
        if not core_indices:
            raise RuntimeError(f"decoded no core frames for {core_start_us}-{core_end_us}")
        core_frames = [frames[index] for index in core_indices]
        segment_id = (
            f"shot-{shot.shot_id:06d}-chunk-{chunk_ordinal:06d}-{core_start_us}-{core_end_us}"
        )
        segment = Segment(
            segment_id=segment_id,
            shot_id=shot.shot_id,
            start_pts_us=core_frames[0].ref.pts_us,
            end_pts_us=core_frames[-1].ref.pts_us + (core_frames[-1].duration_us or 1),
            start_frame_index=core_frames[0].ref.frame_index,
            end_frame_index=core_frames[-1].ref.frame_index + 1,
            core_start_index=core_indices[0],
            core_end_index=core_indices[-1] + 1,
        )
        repository.upsert_segment(job_id, segment)
        if options.resume and checkpoint_store.valid(segment_id):
            metadata = checkpoint_store.metadata(segment_id)
            # Old checkpoints did not record this flag.  Conservatively treat
            # them as modified so resume can never replace processed frames
            # with the untouched input during final assembly.
            modified = bool(metadata.get("modified", True))
            resumed_review = bool(metadata.get("review_required", False))
            resumed_route = str(metadata.get("route", "resumed"))
            resumed_engine = str(metadata.get("engine", "checkpoint"))
            resumed_qc = bool(metadata.get("qc_passed", True))
            resumed_reason = metadata.get("review_reason")
            repository.update_segment(
                job_id,
                segment_id,
                state="done",
                output_checkpoint_path=checkpoint_store.paths(segment_id)[0],
            )
            audit.segments.append(
                SegmentAudit(
                    segment_id,
                    shot.shot_id,
                    segment.start_pts_us,
                    segment.end_pts_us,
                    resumed_route,
                    ["validated_checkpoint", "resumed"],
                    0,
                    resumed_qc,
                    resumed_review,
                    resumed_reason,
                    resumed_engine,
                    0.0,
                    metadata.get("peak_vram_mb"),
                )
            )
            return segment_id, modified, resumed_review

        self._check_cancel(repository, job_id)
        repository.transition_job(job_id, JobState.ANALYZED, force=True)
        if external_masks is None:
            tracks, _detections = self._build_tracks(frames, shot, global_rois, chunk_ordinal)
            repository.save_tracks(job_id, tracks)
            uncertain = any(item.classification == TrackClassification.UNCERTAIN for item in tracks)
            suspicious_protected = any(
                item.classification in {TrackClassification.SCENE_TEXT, TrackClassification.LOGO}
                and item.features.layout_prior > 0.55
                for item in tracks
            )
        else:
            tracks = []
            uncertain = False
            suspicious_protected = False

        original_policy = self.config.get("subtitle_discovery.uncertain_track_policy", "review")
        if options.aggressive_uncertain_removal and original_policy != "remove":
            segment_config = self.config.with_overrides(
                {"subtitle_discovery": {"uncertain_track_policy": "remove"}}
            )
            mask_generator = ProbabilityMaskGenerator(segment_config)
        else:
            mask_generator = self.mask_generator

        best_output: list[np.ndarray] | None = None
        best_report: QualityReport | None = None
        best_decision: RouteDecision | None = None
        best_engine = "none"
        best_peak_vram: int | None = None
        best_penalty = float("inf")
        best_elapsed = 0.0
        review_reason: str | None = None
        attempt = 0
        retry_expand = 0
        flow_preset: str | None = None
        force_route: str | None = None
        last_features = None
        cached_reconstruction_key: tuple[int, str] | None = None
        cached_reconstruction: (
            tuple[list[MaskFrame], list[CleanPlateResult], list[MaskFrame], Any] | None
        ) = None

        while True:
            self._check_cancel(repository, job_id)
            effective_flow = flow_preset or str(
                self.config.get("motion_analysis.local_flow.default", "opencv_dis_fast")
            )
            reconstruction_key = (retry_expand, effective_flow)
            if (
                cached_reconstruction_key == reconstruction_key
                and cached_reconstruction is not None
            ):
                masks, clean_core, core_masks, features = cached_reconstruction
            else:
                repository.transition_job(job_id, JobState.MASKED, force=True)
                if external_masks is not None:
                    masks = external_masks.generate(frames, retry_expand_px=retry_expand)
                else:
                    masks = mask_generator.generate(frames, tracks, retry_expand_px=retry_expand)
                repository.transition_job(job_id, JobState.RECONSTRUCTING, force=True)
                clean_core = self.clean_plate.reconstruct_sequence(
                    frames,
                    masks,
                    flow_preset=flow_preset,
                    target_indices=core_indices,
                )
                core_masks = [masks[index] for index in core_indices]
                features = build_segment_features(
                    core_frames,
                    core_masks,
                    clean_core,
                    vram_calibrator=vram_calibrator,
                    fp16=bool(self.config.get("video_inpainting.propainter.fp16", True)),
                )
                # Keep only the latest reconstruction.  Route-only retries can
                # reuse it without retaining multiple full-resolution maps.
                cached_reconstruction_key = reconstruction_key
                cached_reconstruction = (masks, clean_core, core_masks, features)
            last_features = features
            decision = self._force_decision(self.router.decide(features), force_route)
            repository.update_segment(
                job_id,
                segment_id,
                state="processing",
                route=decision.route.value,
                reasons=decision.reason_codes,
                features=features,
                attempt=attempt,
            )

            clean_images = [item.image_bgr for item in clean_core]
            residual_core = [item.residual_mask for item in clean_core]
            inpaint_result = None
            started = time.perf_counter()
            actual_decision = decision
            if decision.route == Route.REVIEW:
                review_reason = "router_requested_review"
                fallback = (
                    Route.OFFICIAL_PROPAINTER
                    if features.residual_mask_ratio_of_roi > 0.08
                    else Route.TBE_LAMA
                )
                actual_decision = RouteDecision(
                    fallback,
                    (*decision.reason_codes, "best_effort_for_review"),
                    features,
                    decision.parameters,
                )

            if actual_decision.route in {Route.COPY, Route.TBE_ONLY}:
                output_core = [item.copy() for item in clean_images]
                engine_name = "copy" if actual_decision.route == Route.COPY else "tbe_clean_plate"
                peak_vram = 0
            else:
                height, width = frames[0].image_bgr.shape[:2]
                if actual_decision.route in {Route.OFFICIAL_PROPAINTER, Route.STTN_FALLBACK}:
                    # Give video models temporal context. Outside the core, the
                    # full subtitle mask is used so unreconstructed subtitle
                    # pixels cannot leak into the model as known content.
                    clean_by_index = {
                        context_index: result
                        for context_index, result in zip(core_indices, clean_core, strict=True)
                    }
                    request_frames: list[np.ndarray] = []
                    request_masks: list[np.ndarray] = []
                    for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
                        if index in clean_by_index:
                            request_frames.append(clean_by_index[index].image_bgr)
                            request_masks.append(clean_by_index[index].residual_mask)
                        else:
                            request_frames.append(frame.image_bgr)
                            request_masks.append(mask.hard_mask)
                    roi = self._inpaint_roi(request_masks, height, width, actual_decision.route)
                    request = InpaintRequest(
                        segment_id,
                        request_frames,
                        request_masks,
                        [item.ref.pts_us for item in frames],
                        roi,
                        {
                            "fps": probe.fps,
                            "runtime_dir": str(work_dir / "runtime"),
                            "keep_runtime_artifacts": options.keep_intermediates,
                        },
                    )
                    with GpuLease(work_dir.parent.parent / "locks", options.device_index):
                        repository.transition_job(job_id, JobState.INPAINTING, force=True)
                        inpaint_result = self.inpainters.execute(actual_decision, request)
                    output_core = [inpaint_result.frames_bgr[index] for index in core_indices]
                else:
                    roi = self._inpaint_roi(residual_core, height, width, actual_decision.route)
                    request = InpaintRequest(
                        segment_id,
                        clean_images,
                        residual_core,
                        [item.ref.pts_us for item in core_frames],
                        roi,
                        {
                            "fps": probe.fps,
                            "runtime_dir": str(work_dir / "runtime"),
                            "keep_runtime_artifacts": options.keep_intermediates,
                        },
                    )
                    repository.transition_job(job_id, JobState.INPAINTING, force=True)
                    if actual_decision.route in {Route.TBE_LAMA, Route.TBE_MIGAN}:
                        # ONNX providers may use CUDA as well.  Serialize every
                        # GPU-capable backend, not only the video models.
                        with GpuLease(work_dir.parent.parent / "locks", options.device_index):
                            inpaint_result = self.inpainters.execute(actual_decision, request)
                    else:
                        inpaint_result = self.inpainters.execute(actual_decision, request)
                    output_core = inpaint_result.frames_bgr
                engine_name = inpaint_result.engine
                peak_vram = inpaint_result.peak_vram_mb
                repository.save_model_run(
                    job_id,
                    inpaint_result,
                    attempt,
                    peak_rss_mb=int(psutil.Process().memory_info().rss / 2**20),
                )
                if peak_vram and features.predicted_vram_mb:
                    vram_calibrator.observe(features.predicted_vram_mb, peak_vram)

            repository.transition_job(job_id, JobState.QUALITY_CHECK, force=True)
            if bool(self.config.get("quality_control.enabled", True)):
                report = self.quality_gate.evaluate(
                    [item.image_bgr for item in core_frames],
                    output_core,
                    core_masks,
                    segment_id,
                )
            else:
                report = QualityReport(segment_id, True, (), None, None)
            repository.save_quality_report(job_id, report, attempt)
            elapsed = time.perf_counter() - started
            penalty = self._quality_penalty(report)
            if penalty < best_penalty:
                best_penalty = penalty
                best_output = [item.copy() for item in output_core]
                best_report = report
                best_decision = decision
                best_engine = engine_name
                best_peak_vram = peak_vram
                best_elapsed = elapsed
            if report.passed:
                break
            plan = self.retry_planner.next(report, attempt)
            if plan is None:
                review_reason = report.review_reason or "quality_gate_exhausted"
                break
            attempt = plan.attempt
            retry_expand = max(retry_expand, plan.mask_expand_px)
            flow_preset = plan.flow_preset or flow_preset
            force_route = plan.force_route or force_route
            repository.transition_job(job_id, JobState.RETRY_PENDING, force=True)

        assert best_output is not None and best_report is not None and best_decision is not None
        output_frames = [
            VideoFrame(
                ref=source.ref,
                image_bgr=image,
                source_pts=source.source_pts,
                source_time_base=source.source_time_base,
                duration_us=source.duration_us,
            )
            for source, image in zip(core_frames, best_output, strict=True)
        ]
        review_required = (
            not best_report.passed
            or review_reason is not None
            or (uncertain and not options.aggressive_uncertain_removal)
            or suspicious_protected
        )
        if uncertain and not options.aggressive_uncertain_removal and review_reason is None:
            review_reason = "uncertain_text_track_protected"
        if suspicious_protected and review_reason is None:
            review_reason = "subtitle_like_protected_scene_text"
        modified = any(
            np.any(before.image_bgr != after)
            for before, after in zip(core_frames, best_output, strict=True)
        )
        checkpoint_path = checkpoint_store.write(
            segment_id,
            output_frames,
            extra_metadata={
                "modified": modified,
                "review_required": review_required,
                "review_reason": review_reason,
                "route": best_decision.route.value,
                "engine": best_engine,
                "qc_passed": best_report.passed,
                "attempts": attempt + 1,
                "peak_vram_mb": best_peak_vram,
            },
        )
        repository.update_segment(
            job_id,
            segment_id,
            state="done",
            route=best_decision.route.value,
            reasons=best_decision.reason_codes,
            features=last_features,
            attempt=attempt,
            output_checkpoint_path=checkpoint_path,
        )
        audit.segments.append(
            SegmentAudit(
                segment_id=segment_id,
                shot_id=shot.shot_id,
                start_pts_us=segment.start_pts_us,
                end_pts_us=segment.end_pts_us,
                route=best_decision.route.value,
                route_reasons=list(best_decision.reason_codes),
                attempts=attempt + 1,
                qc_passed=best_report.passed,
                review_required=review_required,
                review_reason=review_reason,
                engine=best_engine,
                elapsed_seconds=best_elapsed,
                peak_vram_mb=best_peak_vram,
                feature_summary=jsonable(last_features) if last_features is not None else {},
                quality_metrics=[jsonable(item) for item in best_report.metrics],
            )
        )
        return segment_id, modified, review_required

    @staticmethod
    def _check_cancel(repository: JobRepository, job_id: str) -> None:
        if repository.cancellation_requested(job_id):
            raise CancelledError(f"job cancellation requested: {job_id}")

    def process(self, input_path: Path, options: ProcessOptions | None = None) -> PipelineResult:
        options = options or ProcessOptions()
        self._ensure_device(options.device_index)
        source_path = self._resolve_path(input_path)
        output_path = self._output_path(source_path, options)
        if output_path.exists() and not options.overwrite and not options.resume:
            raise FileExistsError(output_path)
        work_root = self._work_dir(options)
        work_root.mkdir(parents=True, exist_ok=True)
        database_path = self._resolve_path(
            Path(str(self.config.get("runtime.state_db", work_root / "vsrx.sqlite3"))), work_root
        )
        # A relative state_db in the reference config is rooted under the active
        # work directory rather than the caller's current directory.
        if not Path(str(self.config.get("runtime.state_db", ""))).is_absolute():
            database_path = (
                work_root / Path(str(self.config.get("runtime.state_db", "vsrx.sqlite3"))).name
            )
        repository = JobRepository(database_path)

        with self._timed_dummy() as _probe_timer:
            probe = self.prober.probe(source_path)
        external_masks = (
            ExternalMaskProvider(options.external_mask_path) if options.external_mask_path else None
        )
        external_mask_digest = external_masks.digest if external_masks is not None else None
        execution_hash = self._execution_hash(options, external_mask_digest)
        manifest_hash = self._manifest_hash()
        job_id, created = repository.create_or_get_job(
            input_path=source_path,
            input_hash=probe.input_hash,
            config_hash=execution_hash,
            model_manifest_hash=manifest_hash,
            output_path=output_path,
            priority=options.priority,
        )
        job_dir = work_root / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        audit = JobAudit.start(
            job_id,
            source_path,
            output_path,
            probe.input_hash,
            execution_hash,
            str(self.config.get("profile", "balanced")),
        )
        audit.model_status = self.inpainters.model_status()
        checkpoint_store = SegmentCheckpointStore(job_dir / "segments")
        audit_paths = (job_dir / "report" / "audit.json", job_dir / "report" / "audit.md")

        existing = repository.get_job(job_id)
        final_manifest_path = job_dir / "final_output.json"
        completed_output_valid = False
        if (
            existing
            and existing["state"] in {JobState.DONE.value, JobState.REVIEW_REQUIRED.value}
            and output_path.is_file()
            and final_manifest_path.is_file()
            and options.resume
            and not options.overwrite
        ):
            try:
                final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
                completed_output_valid = (
                    Path(str(final_manifest.get("output_path", ""))).resolve() == output_path
                    and int(final_manifest.get("byte_size", -1)) == output_path.stat().st_size
                    and str(final_manifest.get("sha256", "")) == sha256_file(output_path)
                    and str(final_manifest.get("input_hash", "")) == probe.input_hash
                    and str(final_manifest.get("config_hash", "")) == execution_hash
                    and str(final_manifest.get("requested_codec", "auto")) == options.codec
                )
            except Exception:
                completed_output_valid = False
        if completed_output_valid:
            final_state = str(existing["state"])
            audit.finish(final_state)
            audit.warnings.append("复用了已校验的完整输出；历史分段审计保存在 SQLite 中。")
            audit_paths = audit.write(job_dir / "report")
            return PipelineResult(
                job_id,
                output_path,
                *audit_paths,
                final_state,
                bool(existing.get("review_required")),
            )

        if output_path.exists() and not options.overwrite:
            resumable_output = False
            if final_manifest_path.is_file() and options.resume:
                try:
                    previous = json.loads(final_manifest_path.read_text(encoding="utf-8"))
                    resumable_output = (
                        Path(str(previous.get("output_path", ""))).resolve() == output_path
                        and str(previous.get("input_hash", "")) == probe.input_hash
                        and str(previous.get("config_hash", "")) == execution_hash
                    )
                except Exception:
                    resumable_output = False
            if not resumable_output:
                repository.close()
                raise FileExistsError(
                    f"output already exists and is not a resumable artifact: {output_path}; use --overwrite"
                )

        try:
            repository.save_probe(job_id, probe)
            repository.transition_job(job_id, JobState.PROBED, force=not created)

            soft_policy = str(self.config.get("input.handle_soft_subtitles", "remove"))
            scan_hard_with_soft = bool(
                self.config.get("input.also_scan_hard_when_soft_present", False)
            )
            if (
                external_masks is None
                and probe.has_soft_subtitles
                and soft_policy in {"remove", "extract"}
                and not options.force_hard_subtitle_scan
                and not scan_hard_with_soft
            ):
                repository.transition_job(job_id, JobState.ENCODING, force=True)
                if soft_policy == "extract":
                    self.soft_subtitles.extract(
                        probe, output_path.parent / f"{output_path.stem}_subtitles"
                    )
                self.soft_subtitles.remove(probe, output_path)
                audit.soft_subtitle_fast_path = True
                write_json_atomic(
                    final_manifest_path,
                    {
                        "schema_version": 1,
                        "job_id": job_id,
                        "output_path": str(output_path),
                        "byte_size": output_path.stat().st_size,
                        "sha256": sha256_file(output_path),
                        "input_hash": probe.input_hash,
                        "config_hash": execution_hash,
                        "base_config_hash": self.config.hash,
                        "requested_codec": options.codec,
                        "soft_subtitle_fast_path": True,
                    },
                )
                audit.finish(JobState.DONE.value)
                repository.transition_job(job_id, JobState.DONE, force=True)
                audit_paths = audit.write(job_dir / "report")
                return PipelineResult(job_id, output_path, *audit_paths, JobState.DONE.value, False)

            with self._timed(audit, "scene_detection"):
                shots = self.scene_detector.detect(probe)
            repository.save_shots(job_id, shots)
            reader = PyAVFrameReader(probe)
            with self._timed(audit, "subtitle_discovery"):
                if options.fixed_rois:
                    rois, discovery_detections = list(options.fixed_rois), []
                elif external_masks is not None:
                    rois = external_masks.discover_rois(probe.width, probe.height)
                    if not rois:
                        bottom = int(
                            probe.height
                            * (
                                1.0
                                - float(
                                    self.config.get(
                                        "subtitle_discovery.default_bottom_band_hint", 0.38
                                    )
                                )
                            )
                        )
                        rois = [(0, bottom, probe.width, probe.height)]
                    discovery_detections = []
                else:
                    rois, discovery_detections = self._discover_rois(probe, shots, reader, ())
            audit.rois = rois
            repository.transition_job(job_id, JobState.ANALYZED, force=True)

            analysis_payload = {
                "job_id": job_id,
                "input": str(source_path),
                "probe": jsonable(probe),
                "shots": [jsonable(item) for item in shots],
                "rois": rois,
                "discovery_detection_count": len(discovery_detections),
                "model_status": audit.model_status,
                "external_mask_path": str(options.external_mask_path)
                if options.external_mask_path
                else None,
                "external_mask_digest": external_mask_digest,
                "execution_hash": execution_hash,
            }
            (job_dir / "analysis.json").write_text(
                json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if options.dry_run:
                audit.finish(JobState.ANALYZED.value)
                audit_paths = audit.write(job_dir / "report")
                return PipelineResult(job_id, None, *audit_paths, JobState.ANALYZED.value, False)

            segment_order: list[str] = []
            any_modified = False
            review_required = False
            vram_calibrator = VramCalibrator(job_dir / "vram_calibration.json")
            with self._timed(audit, "segment_processing"):
                for shot in shots:
                    for ordinal, core_start, core_end in self._chunk_ranges(shot, probe.fps):
                        segment_id, modified, review = self._process_segment(
                            job_id=job_id,
                            repository=repository,
                            checkpoint_store=checkpoint_store,
                            work_dir=job_dir,
                            probe=probe,
                            shot=shot,
                            chunk_ordinal=ordinal,
                            core_start_us=core_start,
                            core_end_us=core_end,
                            global_rois=rois,
                            options=options,
                            audit=audit,
                            vram_calibrator=vram_calibrator,
                            external_masks=external_masks,
                        )
                        segment_order.append(segment_id)
                        any_modified = any_modified or modified
                        review_required = review_required or review

            repository.transition_job(job_id, JobState.ENCODING, force=True)
            with self._timed(audit, "encoding"):
                if not any_modified:
                    # Pixel-identical path avoids a needless lossy encode.
                    self.soft_subtitles.remove(probe, output_path)
                else:
                    assembled = checkpoint_store.assemble(
                        segment_order, job_dir / "assembled.ffv1.mkv", probe
                    )
                    self.final_encoder.encode(assembled, probe, output_path, codec=options.codec)
            write_json_atomic(
                final_manifest_path,
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "output_path": str(output_path),
                    "byte_size": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "input_hash": probe.input_hash,
                    "config_hash": execution_hash,
                    "base_config_hash": self.config.hash,
                    "requested_codec": options.codec,
                    "model_manifest_hash": manifest_hash,
                    "segment_ids": segment_order,
                    "modified": any_modified,
                },
            )
            final_state = JobState.REVIEW_REQUIRED if review_required else JobState.DONE
            repository.transition_job(
                job_id, final_state, review_required=review_required, force=True
            )
            audit.finish(final_state.value)
            audit_paths = audit.write(job_dir / "report")
            if (
                not options.keep_intermediates
                and not review_required
                and not bool(self.config.get("runtime.keep_success_intermediates", False))
            ):
                runtime_dir = job_dir / "runtime"
                shutil.rmtree(runtime_dir, ignore_errors=True)
            return PipelineResult(
                job_id, output_path, *audit_paths, final_state.value, review_required
            )
        except CancelledError as exc:
            repository.transition_job(
                job_id, JobState.CANCELLED, error_code=exc.code, error_message=str(exc), force=True
            )
            audit.finish(JobState.CANCELLED.value)
            audit.warnings.append(str(exc))
            audit_paths = audit.write(job_dir / "report")
            return PipelineResult(job_id, None, *audit_paths, JobState.CANCELLED.value, False)
        except Exception as exc:
            code = exc.code if isinstance(exc, VSRXError) else type(exc).__name__
            repository.transition_job(
                job_id, JobState.FAILED, error_code=code, error_message=str(exc), force=True
            )
            audit.finish(JobState.FAILED.value)
            audit.warnings.append(f"{code}: {exc}")
            audit_paths = audit.write(job_dir / "report")
            raise
        finally:
            repository.close()

    @contextmanager
    def _timed_dummy(self):
        started = time.perf_counter()
        holder: dict[str, float] = {}
        try:
            yield holder
        finally:
            holder["elapsed"] = time.perf_counter() - started
