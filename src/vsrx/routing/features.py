from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from vsrx.cleanplate.coverage import aggregate_clean_plate_metrics
from vsrx.domain.contracts import CleanPlateResult, MaskFrame, SegmentFeatures, VideoFrame
from vsrx.motion.confidence import foreground_crossing_score
from vsrx.routing.budget import VramCalibrator
from vsrx.utils.geometry import union_bboxes


def _motion_score(frames: Sequence[VideoFrame], masks: Sequence[MaskFrame]) -> float:
    if len(frames) < 2:
        return 0.0
    values: list[float] = []
    for previous, current, mask in zip(frames, frames[1:], masks[1:], strict=False):
        previous_gray = cv2.cvtColor(previous.image_bgr, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current.image_bgr, cv2.COLOR_BGR2GRAY)
        shift, response = cv2.phaseCorrelate(
            previous_gray.astype(np.float32), current_gray.astype(np.float32)
        )
        translation = min(1.0, np.hypot(*shift) / 18.0)
        difference = cv2.absdiff(previous_gray, current_gray).astype(np.float32) / 255.0
        select = mask.hard_mask == 0
        local = (
            float(np.percentile(difference[select], 85))
            if np.any(select)
            else float(np.mean(difference))
        )
        values.append(
            float(
                np.clip(translation * 0.55 + local * 1.4 * 0.45 + (1.0 - response) * 0.08, 0.0, 1.0)
            )
        )
    return float(np.mean(values)) if values else 0.0


def _flicker_risk(clean: Sequence[CleanPlateResult], masks: Sequence[MaskFrame]) -> float:
    if len(clean) < 2:
        return 0.0
    values: list[float] = []
    for previous, current, mask in zip(clean, clean[1:], masks[1:], strict=False):
        select = mask.hard_mask > 0
        if not np.any(select):
            continue
        delta = cv2.absdiff(previous.image_bgr, current.image_bgr)
        gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        values.append(float(np.percentile(gray[select], 90)))
    return float(np.clip(np.mean(values) * 2.2 if values else 0.0, 0.0, 1.0))


def build_segment_features(
    frames: Sequence[VideoFrame],
    masks: Sequence[MaskFrame],
    clean: Sequence[CleanPlateResult],
    *,
    vram_calibrator: VramCalibrator | None = None,
    fp16: bool = True,
) -> SegmentFeatures:
    if not frames:
        raise ValueError("segment must contain frames")
    mean_coverage, mean_confidence, residual_ratio, largest = aggregate_clean_plate_metrics(
        clean, masks
    )
    mask_ratio = float(np.mean([item.mask_ratio_of_frame for item in masks])) if masks else 0.0
    mask_boxes = [item.expanded_bbox_xyxy for item in masks if item.expanded_bbox_xyxy is not None]
    roi = union_bboxes(mask_boxes)
    if roi is None:
        roi_width = roi_height = 1
    else:
        roi_width, roi_height = max(1, roi[2] - roi[0]), max(1, roi[3] - roi[1])
    calibrator = vram_calibrator or VramCalibrator()
    predicted = calibrator.predict(roi_width, roi_height, len(frames), fp16=fp16)
    source_images = [item.image_bgr for item in frames]
    hard_masks = [item.hard_mask for item in masks]

    flow_confidence_values: list[float] = []
    for result, mask in zip(clean, masks, strict=True):
        select = mask.hard_mask > 0
        if np.any(select):
            flow_confidence_values.append(float(np.mean(result.confidence[select])))
    return SegmentFeatures(
        shot_id=frames[0].ref.shot_id,
        start_pts_us=frames[0].ref.pts_us,
        end_pts_us=frames[-1].ref.pts_us + (frames[-1].duration_us or 1),
        frame_count=len(frames),
        mask_ratio_of_frame=mask_ratio,
        mean_clean_plate_coverage=mean_coverage,
        mean_clean_plate_confidence=mean_confidence,
        mean_flow_confidence=float(np.mean(flow_confidence_values))
        if flow_confidence_values
        else 1.0,
        motion_score=_motion_score(frames, masks),
        foreground_crossing_score=foreground_crossing_score(source_images, hard_masks),
        flicker_risk=_flicker_risk(clean, masks),
        largest_residual_component_px=largest,
        residual_mask_ratio_of_roi=residual_ratio,
        predicted_vram_mb=predicted,
    )
