from __future__ import annotations

import cv2
import numpy as np


def motion_magnitude_score(flow_xy: np.ndarray, mask: np.ndarray | None = None) -> float:
    magnitude = np.linalg.norm(flow_xy.astype(np.float32), axis=-1)
    values = magnitude[(mask > 0)] if mask is not None and np.any(mask > 0) else magnitude.ravel()
    if values.size == 0:
        return 0.0
    # 20 px/frame and above is treated as very high motion at the working scale.
    return float(np.clip(np.percentile(values, 85) / 20.0, 0.0, 1.0))


def foreground_crossing_score(frames: list[np.ndarray], masks: list[np.ndarray]) -> float:
    if len(frames) < 2 or not any(np.any(mask) for mask in masks):
        return 0.0
    values: list[float] = []
    for previous, current, mask in zip(frames, frames[1:], masks[1:], strict=False):
        if not np.any(mask):
            continue
        delta = cv2.absdiff(previous, current)
        gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        inside = gray[mask > 0]
        if inside.size:
            values.append(float(np.percentile(inside, 85)))
    return float(np.clip(np.mean(values) * 2.5 if values else 0.0, 0.0, 1.0))


def flow_confidence_mean(
    confidence_maps: list[np.ndarray], masks: list[np.ndarray] | None = None
) -> float:
    values: list[float] = []
    for index, confidence in enumerate(confidence_maps):
        select = None
        if masks is not None and index < len(masks) and np.any(masks[index] > 0):
            select = masks[index] > 0
        sample = confidence[select] if select is not None else confidence.ravel()
        if sample.size:
            values.append(float(np.mean(sample)))
    return float(np.mean(values)) if values else 1.0
