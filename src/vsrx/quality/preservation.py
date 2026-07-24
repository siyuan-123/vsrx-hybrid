from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def preservation_error(
    source: Sequence[np.ndarray], output: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> tuple[float, float]:
    maximum = 0.0
    changed_fraction = 0.0
    count = 0
    for before, after, mask in zip(source, output, masks, strict=True):
        outside = mask == 0
        if not np.any(outside):
            continue
        delta = np.max(np.abs(before.astype(np.int16) - after.astype(np.int16)), axis=2)
        maximum = max(maximum, float(np.max(delta[outside])))
        changed_fraction += float(np.mean(delta[outside] > 1))
        count += 1
    return maximum, changed_fraction / max(count, 1)


def sharpness_ratio(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray], ring_width: int = 10
) -> float:
    values: list[float] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    for frame, mask in zip(frames, masks, strict=True):
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            continue
        ring = (cv2.dilate(binary, kernel) > 0) & (binary == 0)
        if not np.any(ring):
            continue
        lap = np.abs(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
        inside_value = float(np.mean(lap[binary > 0]))
        ring_value = float(np.mean(lap[ring]))
        values.append(inside_value / max(ring_value, 1e-3))
    return float(np.median(values)) if values else 1.0
