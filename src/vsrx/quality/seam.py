from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def boundary_seam_metrics(
    frames: Sequence[np.ndarray], masks: Sequence[np.ndarray], ring_width: int = 8
) -> tuple[float, float]:
    color_values: list[float] = []
    gradient_ratios: list[float] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    for frame, mask in zip(frames, masks, strict=True):
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            continue
        inner = (binary > 0) & (cv2.erode(binary, kernel) == 0)
        outer = (cv2.dilate(binary, kernel) > 0) & (binary == 0)
        if int(inner.sum()) < 16 or int(outer.sum()) < 16:
            continue
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        # OpenCV Lab Euclidean distance is used as an inexpensive, monotonic
        # seam proxy. The threshold remains configurable and benchmark-driven.
        inside_median = np.median(lab[inner], axis=0)
        outside_median = np.median(lab[outer], axis=0)
        color_values.append(float(np.linalg.norm(inside_median - outside_median) * 100.0 / 255.0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        inner_gradient = float(np.median(gradient[inner]))
        outer_gradient = float(np.median(gradient[outer]))
        gradient_ratios.append(
            max(inner_gradient, outer_gradient) / max(min(inner_gradient, outer_gradient), 2.0)
        )
    return float(np.median(color_values)) if color_values else 0.0, float(
        np.median(gradient_ratios)
    ) if gradient_ratios else 1.0
