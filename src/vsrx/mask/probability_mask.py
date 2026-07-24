from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np

from vsrx.domain.contracts import MaskFrame, SubtitleTrack, VideoFrame
from vsrx.domain.enums import TrackClassification
from vsrx.mask.panel_detector import BackgroundPanelDetector
from vsrx.mask.temporal_union import motion_compensated_union
from vsrx.mask.text_effects import (
    expand_text_effects,
    remove_small_components,
    text_probability_seed,
)
from vsrx.utils.config import Config
from vsrx.utils.geometry import mask_bbox
from vsrx.utils.image import fill_holes, soft_alpha_from_mask


class ProbabilityMaskGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.panel_detector = BackgroundPanelDetector()

    def _should_remove(self, track: SubtitleTrack) -> bool:
        if track.classification in {TrackClassification.SUBTITLE, TrackClassification.OVERLAY}:
            return True
        if track.classification == TrackClassification.UNCERTAIN:
            return (
                str(self.config.get("subtitle_discovery.uncertain_track_policy", "review"))
                == "remove"
            )
        return False

    @staticmethod
    def _polygon_mask(
        shape: tuple[int, int], points: tuple[tuple[float, float], ...]
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        polygon = np.round(np.asarray(points, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _dilation_limits(self, height: int) -> tuple[int, int]:
        if height >= 900:
            return (
                int(self.config.get("mask_generation.dilation.min_px_1080p", 5)),
                int(self.config.get("mask_generation.dilation.max_px_1080p", 11)),
            )
        return (
            int(self.config.get("mask_generation.dilation.min_px_720p", 3)),
            int(self.config.get("mask_generation.dilation.max_px_720p", 7)),
        )

    def generate(
        self,
        frames: Sequence[VideoFrame],
        tracks: Sequence[SubtitleTrack],
        retry_expand_px: int = 0,
    ) -> list[MaskFrame]:
        if not frames:
            return []
        height, width = frames[0].image_bgr.shape[:2]
        frame_map = {frame.ref.frame_index: frame for frame in frames}
        detections: dict[int, list[tuple[SubtitleTrack, object]]] = defaultdict(list)
        for track in tracks:
            if not self._should_remove(track):
                continue
            for detection in track.detections:
                if detection.frame.frame_index in frame_map:
                    detections[detection.frame.frame_index].append((track, detection))

        hard_masks: list[np.ndarray] = []
        track_ids_by_frame: list[list[str]] = []
        confidence_by_frame: list[list[float]] = []
        min_dilate, max_dilate = self._dilation_limits(height)
        fraction = float(self.config.get("mask_generation.dilation.text_height_fraction", 0.12))
        panel_threshold = float(
            self.config.get("mask_generation.panel_mode_requires_track_score", 0.72)
        )
        minimum_component = int(
            self.config.get("mask_generation.morphology.remove_components_below_px", 6)
        )

        for frame in frames:
            combined = np.zeros((height, width), dtype=np.uint8)
            source_ids: list[str] = []
            confidences: list[float] = []
            for track, detection in detections.get(frame.ref.frame_index, []):
                polygon = self._polygon_mask((height, width), detection.polygon.points)
                bbox = detection.bbox_xyxy
                text_height = max(1, bbox[3] - bbox[1])
                seed = text_probability_seed(frame.image_bgr, polygon, text_height)
                expanded = expand_text_effects(
                    seed,
                    text_height,
                    fraction,
                    min_dilate + retry_expand_px,
                    max_dilate + retry_expand_px,
                )
                if (
                    str(self.config.get("mask_generation.include_background_panel", "auto"))
                    != "false"
                    and track.score >= panel_threshold
                ):
                    panel = self.panel_detector.detect(
                        frame.image_bgr,
                        bbox,
                        minimum_score=max(panel_threshold, track.score * 0.86),
                    )
                    expanded = cv2.bitwise_or(expanded, panel)
                combined = cv2.bitwise_or(combined, expanded)
                source_ids.append(track.track_id)
                confidences.append(float(detection.confidence * track.score))
            close_size = int(self.config.get("mask_generation.morphology.close_kernel_px", 3)) | 1
            if close_size > 1:
                combined = cv2.morphologyEx(
                    combined, cv2.MORPH_CLOSE, np.ones((close_size, close_size), dtype=np.uint8)
                )
            if self.config.get("mask_generation.morphology.fill_holes", True):
                combined = fill_holes(combined)
            combined = remove_small_components(combined, minimum_component)
            hard_masks.append(combined)
            track_ids_by_frame.append(source_ids)
            confidence_by_frame.append(confidences)

        temporal_cfg = self.config.get("mask_generation.temporal_union", {})
        if temporal_cfg.get("enabled", True):
            hard_masks = motion_compensated_union(
                [frame.image_bgr for frame in frames],
                hard_masks,
                radius=int(temporal_cfg.get("radius_frames", 1)),
            )

        feather = int(
            self.config.get(
                "mask_generation.feather.radius_px_1080p"
                if height >= 900
                else "mask_generation.feather.radius_px_720p",
                5 if height >= 900 else 3,
            )
        )
        result: list[MaskFrame] = []
        for index, (frame, hard) in enumerate(zip(frames, hard_masks, strict=True)):
            bbox = mask_bbox(hard)
            result.append(
                MaskFrame(
                    frame=frame.ref,
                    hard_mask=hard,
                    soft_alpha=soft_alpha_from_mask(hard, feather),
                    source_track_ids=tuple(track_ids_by_frame[index]),
                    confidence=float(np.mean(confidence_by_frame[index]))
                    if confidence_by_frame[index]
                    else 1.0,
                    mask_ratio_of_frame=float((hard > 0).mean()),
                    expanded_bbox_xyxy=bbox,
                )
            )
        return result
