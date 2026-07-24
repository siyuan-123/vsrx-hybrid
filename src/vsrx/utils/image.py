from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        maximum = 1.0 if float(np.nanmax(image)) <= 1.0 else 255.0
        return np.clip(image * (255.0 / maximum), 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def soft_alpha_from_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if radius <= 0:
        return binary.astype(np.float32)
    distance_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    alpha = np.clip(distance_inside / max(radius, 1), 0.0, 1.0)
    kernel = radius * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (kernel | 1, kernel | 1), radius / 2.0)
    alpha[binary == 0] = 0.0
    return alpha.astype(np.float32)


def composite(source: np.ndarray, replacement: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    result = source.astype(np.float32) * (1.0 - alpha) + replacement.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def fill_holes(binary: np.ndarray) -> np.ndarray:
    mask = (binary > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    flood = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)
