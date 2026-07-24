from __future__ import annotations

import cv2
import numpy as np

from vsrx.utils.image import fill_holes


def text_probability_seed(
    image_bgr: np.ndarray, polygon_mask: np.ndarray, text_height: int
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel_size = max(3, min(21, int(round(text_height * 0.45)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    gradient = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    values = np.maximum.reduce([top_hat, black_hat, gradient])
    inside = values[polygon_mask > 0]
    if inside.size == 0:
        return np.zeros_like(polygon_mask)
    threshold = max(8.0, float(np.percentile(inside, 52)))
    seed = ((values >= threshold) & (polygon_mask > 0)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    minimum = max(8, int((polygon_mask > 0).sum() * 0.025))
    if int((seed > 0).sum()) < minimum:
        # A conservative fallback: erode the polygon so we do not erase the whole line box.
        radius = max(1, int(round(text_height * 0.12)))
        seed = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
        )
    return seed


def expand_text_effects(
    seed: np.ndarray, text_height: int, dilation_fraction: float, minimum: int, maximum: int
) -> np.ndarray:
    radius = int(np.clip(round(text_height * dilation_fraction), minimum, maximum))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    expanded = cv2.dilate(seed, kernel)
    # Common subtitle shadows are shifted down/right. Add a small asymmetric copy.
    shift = max(1, radius // 2)
    matrix = np.float32([[1, 0, shift], [0, 1, shift]])
    shadow = cv2.warpAffine(
        expanded,
        matrix,
        (expanded.shape[1], expanded.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    expanded = cv2.bitwise_or(expanded, shadow)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return fill_holes(expanded)


def remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    output = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
            output[labels == label] = 255
    return output
