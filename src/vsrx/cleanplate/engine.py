from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from vsrx.cleanplate.exposure import apply_exposure, fit_per_channel_affine, unmasked_ring
from vsrx.cleanplate.fusion import (
    best_reference_fusion,
    trimmed_mean_fusion,
    weighted_median_fusion,
)
from vsrx.cleanplate.reference_selector import ReferenceSelector
from vsrx.domain.contracts import CleanPlateResult, MaskFrame, VideoFrame
from vsrx.motion.dis_flow import align_reference_to_target
from vsrx.motion.global_registration import GlobalRegistrar, RegistrationResult
from vsrx.utils.config import Config
from vsrx.utils.geometry import expand_bbox, mask_bbox


@dataclass(frozen=True, slots=True)
class ReconstructionStats:
    attempted_references: int
    accepted_references: int
    registration_methods: tuple[str, ...]


@dataclass(slots=True)
class _SequenceMotion:
    """Transforms from every frame into its shot-local anchor coordinates."""

    to_anchor: list[np.ndarray]
    edge_score: list[float]
    edge_valid: list[float]
    shot_root: list[int]


class TemporalCleanPlateReconstructor:
    """Motion-compensated Temporal Background Exposure / clean-plate engine.

    The expensive operation count is linear in the number of frames, not the
    number of target/reference pairs.  Adjacent transforms are estimated once
    and composed for arbitrary references.  Dense local flow is triggered only
    when the globally aligned ROI still disagrees.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.registrar = GlobalRegistrar(config)
        self.selector = ReferenceSelector(config)
        self.last_stats: dict[int, ReconstructionStats] = {}

    def _padding(self, height: int) -> int:
        configured = int(self.config.get("motion_analysis.local_flow.context_padding_px_1080p", 96))
        return max(24, int(round(configured * height / 1080.0)))

    @staticmethod
    def _aggregate_confidence(weights: list[np.ndarray]) -> np.ndarray:
        if not weights:
            raise ValueError("weights cannot be empty")
        stack = np.stack(weights, axis=0).astype(np.float32)
        # Probability-like aggregation: several independent moderate references
        # should produce stronger confidence than one isolated candidate.
        return 1.0 - np.prod(1.0 - np.clip(stack, 0.0, 0.98), axis=0)

    def _build_sequence_motion(
        self,
        frames: Sequence[VideoFrame],
        masks: Sequence[MaskFrame],
    ) -> _SequenceMotion:
        count = len(frames)
        to_anchor = [np.eye(3, dtype=np.float64) for _ in range(count)]
        edge_score = [1.0] * count
        edge_valid = [1.0] * count
        shot_root = list(range(count))
        if count <= 1:
            return _SequenceMotion(to_anchor, edge_score, edge_valid, shot_root)

        anchor_stride = max(
            0,
            int(self.config.get("motion_analysis.global_registration.anchor_stride_frames", 12)),
        )
        current_root = 0
        for index in range(1, count):
            current = frames[index]
            previous = frames[index - 1]
            if current.ref.shot_id != previous.ref.shot_id:
                current_root = index
                shot_root[index] = index
                continue
            shot_root[index] = current_root
            exclude = cv2.bitwise_or(masks[index].hard_mask, masks[index - 1].hard_mask)
            adjacent = self.registrar.estimate(
                current.image_bgr,
                previous.image_bgr,
                source_key=current.ref.pts_us,
                target_key=previous.ref.pts_us,
                exclude_mask=exclude,
            )
            to_anchor[index] = to_anchor[index - 1] @ adjacent.transform.astype(np.float64)
            edge_score[index] = adjacent.score
            edge_valid[index] = adjacent.valid_fraction

            # Periodic direct anchors constrain accumulated phase/affine drift
            # while adding only O(n / stride) registrations.
            if (
                anchor_stride > 0
                and index - current_root >= anchor_stride
                and (index - current_root) % anchor_stride == 0
            ):
                anchor_index = max(current_root, index - anchor_stride)
                direct_exclude = cv2.bitwise_or(
                    masks[index].hard_mask, masks[anchor_index].hard_mask
                )
                direct = self.registrar.estimate(
                    current.image_bgr,
                    frames[anchor_index].image_bgr,
                    source_key=current.ref.pts_us,
                    target_key=frames[anchor_index].ref.pts_us,
                    exclude_mask=direct_exclude,
                )
                path_score = min(edge_score[anchor_index + 1 : index + 1], default=0.0)
                if direct.score >= max(0.48, path_score - 0.03) and direct.valid_fraction >= 0.55:
                    to_anchor[index] = to_anchor[anchor_index] @ direct.transform.astype(np.float64)
                    edge_score[index] = max(edge_score[index], direct.score)
                    edge_valid[index] = max(edge_valid[index], direct.valid_fraction)

        return _SequenceMotion(to_anchor, edge_score, edge_valid, shot_root)

    @staticmethod
    def _valid_transform(matrix: np.ndarray) -> bool:
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            return False
        determinant = float(np.linalg.det(matrix[:2, :2]))
        return 0.15 <= abs(determinant) <= 6.0

    def _registration_between(
        self,
        source_index: int,
        target_index: int,
        frames: Sequence[VideoFrame],
        masks: Sequence[MaskFrame],
        motion: _SequenceMotion | None,
    ) -> RegistrationResult:
        source = frames[source_index]
        target = frames[target_index]
        exclude = cv2.bitwise_or(masks[source_index].hard_mask, masks[target_index].hard_mask)
        if (
            motion is not None
            and source.ref.shot_id == target.ref.shot_id
            and motion.shot_root[source_index] == motion.shot_root[target_index]
        ):
            try:
                composed = (
                    np.linalg.inv(motion.to_anchor[target_index]) @ motion.to_anchor[source_index]
                )
            except np.linalg.LinAlgError:
                composed = np.empty((0, 0), dtype=np.float64)
            low, high = sorted((source_index, target_index))
            hops = high - low
            path_scores = motion.edge_score[low + 1 : high + 1]
            path_valid = motion.edge_valid[low + 1 : high + 1]
            score = min(path_scores, default=1.0) * float(np.exp(-0.0035 * hops))
            valid = min(path_valid, default=1.0)
            if self._valid_transform(composed):
                result = RegistrationResult(
                    composed.astype(np.float32),
                    f"composed_adjacent_{hops}",
                    float(np.clip(score, 0.0, 1.0)),
                    float(np.clip(valid, 0.0, 1.0)),
                )
                fallback_below = float(
                    self.config.get(
                        "motion_analysis.global_registration.direct_fallback_score_below",
                        0.52,
                    )
                )
                if result.score >= fallback_below:
                    return result

        # Only a poor/invalid composed path pays for direct registration.
        return self.registrar.estimate(
            source.image_bgr,
            target.image_bgr,
            source_key=source.ref.pts_us,
            target_key=target.ref.pts_us,
            exclude_mask=exclude,
        )

    @staticmethod
    def _early_stop(
        weights: list[np.ndarray],
        target_pixels: np.ndarray,
        minimum_refs: int,
        coverage_threshold: float,
        confidence_threshold: float,
    ) -> bool:
        if len(weights) < minimum_refs or not np.any(target_pixels):
            return False
        stack = np.stack(weights, axis=0)
        valid_count = np.sum(stack > 0.16, axis=0)
        enough = float(np.mean((valid_count[target_pixels] >= minimum_refs).astype(np.float32)))
        aggregate = 1.0 - np.prod(1.0 - np.clip(stack, 0.0, 0.98), axis=0)
        mean_confidence = float(np.mean(aggregate[target_pixels]))
        return enough >= coverage_threshold and mean_confidence >= confidence_threshold

    def reconstruct_sequence(
        self,
        frames: Sequence[VideoFrame],
        masks: Sequence[MaskFrame],
        *,
        flow_preset: str | None = None,
        target_indices: Sequence[int] | None = None,
    ) -> list[CleanPlateResult]:
        if len(frames) != len(masks):
            raise ValueError("frames and masks must have identical lengths")
        if not frames:
            return []
        preset = flow_preset or str(
            self.config.get("motion_analysis.local_flow.default", "opencv_dis_fast")
        )
        minimum_refs = int(self.config.get("clean_plate.minimum_valid_references_per_pixel", 3))
        preferred_refs = int(self.config.get("clean_plate.preferred_valid_references_per_pixel", 5))
        fb_check = bool(self.config.get("motion_analysis.local_flow.forward_backward_check", True))
        fb_threshold = float(
            self.config.get("motion_analysis.local_flow.fb_error_threshold_px", 1.8)
        )
        local_trigger = float(
            self.config.get("motion_analysis.local_flow.auto_trigger_error", 0.105)
        )
        local_min_score = float(
            self.config.get("motion_analysis.local_flow.auto_min_registration_score", 0.82)
        )
        fusion_method = str(self.config.get("clean_plate.fusion", "weighted_median"))
        use_exposure = bool(self.config.get("clean_plate.exposure_compensation.enabled", True))
        compose_motion = bool(
            self.config.get("motion_analysis.global_registration.compose_adjacent_transforms", True)
        )
        early_coverage = float(self.config.get("clean_plate.early_stop_coverage", 0.965))
        early_confidence = float(self.config.get("clean_plate.early_stop_mean_confidence", 0.84))
        motion = self._build_sequence_motion(frames, masks) if compose_motion else None

        results: list[CleanPlateResult] = []
        selected_indices = (
            list(range(len(frames)))
            if target_indices is None
            else [int(index) for index in target_indices]
        )
        for target_index in selected_indices:
            target = frames[target_index]
            target_mask_frame = masks[target_index]
            target_mask = target_mask_frame.hard_mask
            height, width = target.image_bgr.shape[:2]
            if not np.any(target_mask):
                results.append(
                    CleanPlateResult(
                        frame=target.ref,
                        image_bgr=target.image_bgr.copy(),
                        coverage=np.ones((height, width), dtype=np.float16),
                        confidence=np.ones((height, width), dtype=np.float16),
                        residual_mask=np.zeros((height, width), dtype=np.uint8),
                        mean_coverage_in_mask=1.0,
                        mean_confidence_in_mask=1.0,
                        reference_frames=(),
                    )
                )
                continue

            bbox = mask_bbox(target_mask)
            if bbox is None:
                raise RuntimeError("non-empty mask has no bounding box")
            roi = expand_bbox(bbox, self._padding(height), width, height)
            x1, y1, x2, y2 = roi
            target_crop = target.image_bgr[y1:y2, x1:x2]
            target_mask_crop = target_mask[y1:y2, x1:x2]
            target_pixels = target_mask_crop > 0
            ring = unmasked_ring(target_mask_crop, max(8, self._padding(height) // 4))

            candidates = self.selector.select(target_index, frames, masks)
            aligned_images: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            references = []
            methods: list[str] = []
            for candidate in candidates:
                source = frames[candidate.index]
                source_mask = masks[candidate.index].hard_mask
                registration = self._registration_between(
                    candidate.index,
                    target_index,
                    frames,
                    masks,
                    motion,
                )
                aligned = align_reference_to_target(
                    source.image_bgr,
                    target.image_bgr,
                    source_mask,
                    target_mask,
                    registration,
                    roi,
                    preset=preset,
                    forward_backward_check=fb_check,
                    fb_threshold_px=fb_threshold,
                    use_local_flow=None,
                    local_flow_trigger_error=local_trigger,
                    local_flow_min_registration_score=local_min_score,
                )
                image = aligned.image_bgr
                if use_exposure:
                    fit_valid = (
                        ring
                        & (aligned.source_mask == 0)
                        & (~aligned.occlusion)
                        & (aligned.confidence > 0.2)
                    )
                    model = fit_per_channel_affine(image, target_crop, fit_valid)
                    image = apply_exposure(image, model)
                    exposure_weight = 0.78 + model.confidence * 0.22
                else:
                    exposure_weight = 1.0
                temporal = candidate.temporal_weight
                scalar = (
                    candidate.score * 0.30
                    + temporal * 0.14
                    + registration.score * 0.24
                    + candidate.ring_similarity * 0.12
                    + exposure_weight * 0.20
                )
                weight = aligned.confidence * float(np.clip(scalar, 0.05, 1.0))
                weight[aligned.occlusion] = 0.0
                if float(np.max(weight[target_pixels])) < 0.08:
                    continue
                aligned_images.append(image)
                weights.append(weight.astype(np.float32))
                references.append(source.ref)
                methods.append(registration.method)
                if self._early_stop(
                    weights,
                    target_pixels,
                    minimum_refs,
                    early_coverage,
                    early_confidence,
                ):
                    break

            self.last_stats[target.ref.frame_index] = ReconstructionStats(
                attempted_references=len(candidates),
                accepted_references=len(aligned_images),
                registration_methods=tuple(methods),
            )

            if not aligned_images:
                coverage_crop = np.zeros(target_crop.shape[:2], dtype=np.float32)
                confidence_crop = np.zeros_like(coverage_crop)
                fused_crop = target_crop.copy()
            else:
                if fusion_method == "trimmed_mean":
                    fused_crop = trimmed_mean_fusion(
                        aligned_images,
                        weights,
                        target_crop,
                        float(self.config.get("clean_plate.trimmed_mean_fraction", 0.2)),
                    )
                elif fusion_method == "best_reference":
                    fused_crop = best_reference_fusion(aligned_images, weights, target_crop)
                else:
                    fused_crop = weighted_median_fusion(aligned_images, weights, target_crop)
                valid_count = np.sum(np.stack(weights, axis=0) > 0.16, axis=0).astype(np.float32)
                coverage_crop = np.clip(valid_count / max(preferred_refs, 1), 0.0, 1.0)
                confidence_crop = self._aggregate_confidence(weights)

            if aligned_images:
                valid_count = np.sum(np.stack(weights, axis=0) > 0.16, axis=0)
                best = np.max(np.stack(weights, axis=0), axis=0)
            else:
                valid_count = np.zeros(target_crop.shape[:2], dtype=np.int16)
                best = np.zeros(target_crop.shape[:2], dtype=np.float32)
            trusted = ((valid_count >= minimum_refs) & (confidence_crop >= 0.42)) | (
                (valid_count >= 1) & (best >= 0.88) & (confidence_crop >= 0.82)
            )
            recovered_pixels = target_pixels & trusted
            output = target.image_bgr.copy()
            output_crop = output[y1:y2, x1:x2]
            output_crop[recovered_pixels] = fused_crop[recovered_pixels]
            residual_crop = np.zeros_like(target_mask_crop, dtype=np.uint8)
            residual_crop[target_pixels & ~trusted] = 255

            coverage_full = np.ones((height, width), dtype=np.float16)
            confidence_full = np.ones((height, width), dtype=np.float16)
            coverage_full[y1:y2, x1:x2] = coverage_crop.astype(np.float16)
            confidence_full[y1:y2, x1:x2] = confidence_crop.astype(np.float16)
            residual_full = np.zeros((height, width), dtype=np.uint8)
            residual_full[y1:y2, x1:x2] = residual_crop
            mask_select = target_mask > 0
            mean_coverage = (
                float(np.mean(coverage_full[mask_select])) if np.any(mask_select) else 1.0
            )
            mean_confidence = (
                float(np.mean(confidence_full[mask_select])) if np.any(mask_select) else 1.0
            )
            results.append(
                CleanPlateResult(
                    frame=target.ref,
                    image_bgr=output,
                    coverage=coverage_full,
                    confidence=confidence_full,
                    residual_mask=residual_full,
                    mean_coverage_in_mask=mean_coverage,
                    mean_confidence_in_mask=mean_confidence,
                    reference_frames=tuple(references),
                )
            )
        return results
