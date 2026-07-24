from __future__ import annotations

import cv2
import numpy as np

from vsrx.utils.image import soft_alpha_from_mask


def _boundary_offset(
    base: np.ndarray, generated: np.ndarray, mask: np.ndarray, ring_width: int = 7
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(3, dtype=np.float32)
    inner = binary - cv2.erode(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width | 1, ring_width | 1))
    )
    outer = (
        cv2.dilate(
            binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width | 1, ring_width | 1))
        )
        - binary
    )
    if int(inner.sum()) < 20 or int(outer.sum()) < 20:
        return np.zeros(3, dtype=np.float32)
    inside_median = np.median(generated[inner > 0].astype(np.float32), axis=0)
    outside_median = np.median(base[outer > 0].astype(np.float32), axis=0)
    return np.clip(outside_median - inside_median, -18.0, 18.0)


def _grain_noise(
    base: np.ndarray, generated: np.ndarray, mask: np.ndarray, seed: int
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    ring = cv2.dilate(binary, np.ones((11, 11), np.uint8)) - binary
    if int(ring.sum()) < 64:
        return generated
    gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high_base = gray_base - cv2.GaussianBlur(gray_base, (0, 0), 1.2)
    high_generated = gray_generated - cv2.GaussianBlur(gray_generated, (0, 0), 1.2)
    source_sigma = float(np.std(high_base[ring > 0]))
    generated_sigma = (
        float(np.std(high_generated[binary > 0])) if int(binary.sum()) else source_sigma
    )
    missing = np.clip(source_sigma - generated_sigma, 0.0, 5.0)
    if missing < 0.25:
        return generated
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, missing, generated.shape[:2]).astype(np.float32)
    output = generated.astype(np.float32)
    output[binary > 0] += noise[binary > 0, None]
    return np.clip(output, 0, 255).astype(np.uint8)


def composite_exact(
    base: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    *,
    feather_radius: int = 4,
    color_match: bool = True,
    grain_match: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Composite while preserving every pixel outside the hard mask exactly."""

    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return base.copy()
    adjusted = generated.copy()
    if color_match:
        offset = _boundary_offset(base, adjusted, binary)
        adjusted = np.clip(adjusted.astype(np.float32) + offset.reshape(1, 1, 3), 0, 255).astype(
            np.uint8
        )
    if grain_match:
        adjusted = _grain_noise(base, adjusted, binary, seed)
    alpha = soft_alpha_from_mask(binary, feather_radius)[..., None]
    output = base.copy()
    select = binary > 0
    blended = base.astype(np.float32) * (1.0 - alpha) + adjusted.astype(np.float32) * alpha
    output[select] = np.clip(blended[select], 0, 255).astype(np.uint8)
    # An explicit assignment protects against accidental numerical changes.
    output[~select] = base[~select]
    return output
