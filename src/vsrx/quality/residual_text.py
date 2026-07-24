from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vsrx.detection.rapidocr_adapter import CascadedTextDetector
from vsrx.domain.contracts import MaskFrame, VideoFrame
from vsrx.utils.config import Config
from vsrx.utils.geometry import bbox_iou, mask_bbox


class ResidualTextChecker:
    def __init__(self, config: Config, detector: CascadedTextDetector | None = None) -> None:
        self.config = config
        self.detector = detector or CascadedTextDetector(config)

    @staticmethod
    def _score(detections, masks: Sequence[MaskFrame]) -> float:
        mask_by_frame = {item.frame.frame_index: item for item in masks}
        scores: list[float] = []
        for detection in detections:
            mask = mask_by_frame.get(detection.frame.frame_index)
            if mask is None:
                continue
            bbox = mask_bbox(mask.hard_mask)
            if bbox is None:
                continue
            overlap = bbox_iou(detection.bbox_xyxy, bbox)
            if overlap > 0.01:
                scores.append(float(detection.confidence * min(1.0, overlap * 4.0)))
        return float(max(scores, default=0.0))

    def evaluate(
        self,
        source_frames: Sequence[np.ndarray],
        output_frames: Sequence[np.ndarray],
        masks: Sequence[MaskFrame],
    ) -> tuple[float, float, float]:
        active = [index for index, mask in enumerate(masks) if np.any(mask.hard_mask)]
        if not active:
            return 0.0, 0.0, 1.0
        # Bound OCR QC cost for long segments while preserving start/middle/end.
        max_samples = 12
        if len(active) > max_samples:
            positions = np.linspace(0, len(active) - 1, max_samples).round().astype(int)
            active = [active[position] for position in positions]
        source_video_frames: list[VideoFrame] = []
        output_video_frames: list[VideoFrame] = []
        rois: list[tuple[int, int, int, int]] = []
        selected_masks: list[MaskFrame] = []
        for index in active:
            ref = masks[index].frame
            source_video_frames.append(VideoFrame(ref, source_frames[index], None, None))
            output_video_frames.append(VideoFrame(ref, output_frames[index], None, None))
            bbox = mask_bbox(masks[index].hard_mask)
            if bbox is not None:
                rois.append(bbox)
            selected_masks.append(masks[index])
        # Per-frame ROIs are not supported by the detector protocol; a union is
        # safer and still much cheaper than full-frame OCR.
        if rois:
            union = (
                min(x[0] for x in rois),
                min(x[1] for x in rois),
                max(x[2] for x in rois),
                max(x[3] for x in rois),
            )
            scan_rois = [union]
        else:
            scan_rois = None
        tier = str(self.config.get("quality_control.residual_text.detector_tier", "medium"))
        before = self.detector.detect(source_video_frames, scan_rois, tier=tier)
        after = self.detector.detect(output_video_frames, scan_rois, tier=tier)
        before_score = self._score(before, selected_masks)
        after_score = self._score(after, selected_masks)
        drop = (
            float(np.clip((before_score - after_score) / max(before_score, 1e-6), 0.0, 1.0))
            if before_score > 0
            else 1.0
        )
        return before_score, after_score, drop
