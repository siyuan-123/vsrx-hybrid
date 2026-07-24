from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def _ring(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    outer = cv2.dilate(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    )
    return (outer > 0) & (binary == 0)


def temporal_flicker_metrics(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> tuple[float, float]:
    if len(frames) < 2:
        return 0.0, 0.0
    ratios: list[float] = []
    p95_values: list[float] = []
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    for previous, current, mask in zip(frames, frames[1:], masks[1:], strict=False):
        binary = mask > 0
        if not np.any(binary):
            continue
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        try:
            # Current->previous flow gives remap coordinates in previous.
            flow = engine.calc(curr_gray, prev_gray, None)
            yy, xx = np.mgrid[0 : curr_gray.shape[0], 0 : curr_gray.shape[1]].astype(np.float32)
            warped = cv2.remap(
                previous,
                xx + flow[..., 0],
                yy + flow[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        except cv2.error:
            warped = previous
        delta = (
            cv2.cvtColor(cv2.absdiff(warped, current), cv2.COLOR_BGR2GRAY).astype(np.float32)
            / 255.0
        )
        inside = delta[binary]
        ring = _ring(mask)
        outside = delta[ring] if np.any(ring) else delta[~binary]
        if inside.size:
            p95_values.append(float(np.percentile(inside, 95)))
            background = float(np.median(outside)) if outside.size else 0.01
            ratios.append(float(np.median(inside) / max(background, 1.0 / 255.0)))
    return float(np.mean(ratios)) if ratios else 0.0, float(
        np.mean(p95_values)
    ) if p95_values else 0.0
