from __future__ import annotations

import cv2
import numpy as np


def srgb_to_linear(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image_linear: np.ndarray) -> np.ndarray:
    image = np.where(
        image_linear <= 0.0031308,
        image_linear * 12.92,
        1.055 * np.power(np.clip(image_linear, 0, None), 1 / 2.4) - 0.055,
    )
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def local_color_match(
    source: np.ndarray, replacement: np.ndarray, ring_mask: np.ndarray
) -> np.ndarray:
    valid = ring_mask > 0
    if int(valid.sum()) < 32:
        return replacement
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    repl_lab = cv2.cvtColor(replacement, cv2.COLOR_BGR2LAB).astype(np.float32)
    result = repl_lab.copy()
    for channel in range(3):
        src_values = source_lab[..., channel][valid]
        dst_values = repl_lab[..., channel][valid]
        src_std = max(float(src_values.std()), 1.0)
        dst_std = max(float(dst_values.std()), 1.0)
        result[..., channel] = (result[..., channel] - float(dst_values.mean())) * (
            src_std / dst_std
        ) + float(src_values.mean())
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
