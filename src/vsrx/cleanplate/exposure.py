from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ExposureModel:
    gain_bgr: tuple[float, float, float]
    offset_bgr: tuple[float, float, float]
    confidence: float

    @classmethod
    def identity(cls) -> ExposureModel:
        return cls((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0)


def unmasked_ring(mask: np.ndarray, width: int = 12) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.ones_like(binary, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    outer = cv2.dilate(binary, kernel)
    return (outer > 0) & (binary == 0)


def fit_per_channel_affine(
    reference_bgr: np.ndarray,
    target_bgr: np.ndarray,
    valid: np.ndarray,
    *,
    max_samples: int = 50_000,
) -> ExposureModel:
    select = valid.astype(bool)
    count = int(select.sum())
    if count < 96:
        return ExposureModel.identity()
    ys, xs = np.where(select)
    if count > max_samples:
        step = max(1, count // max_samples)
        ys, xs = ys[::step], xs[::step]
    source = reference_bgr[ys, xs].astype(np.float32)
    target = target_bgr[ys, xs].astype(np.float32)

    gains: list[float] = []
    offsets: list[float] = []
    residuals: list[float] = []
    for channel in range(3):
        x = source[:, channel]
        y = target[:, channel]
        # Remove saturated and nearly black samples, then trim gross outliers.
        keep = (x > 5) & (x < 250) & (y > 5) & (y < 250)
        if int(keep.sum()) < 64:
            gains.append(1.0)
            offsets.append(0.0)
            residuals.append(40.0)
            continue
        x = x[keep]
        y = y[keep]
        delta = y - x
        low, high = np.percentile(delta, [8, 92])
        keep2 = (delta >= low) & (delta <= high)
        x, y = x[keep2], y[keep2]
        design = np.column_stack([x, np.ones_like(x)])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            gain = float(np.clip(coefficients[0], 0.72, 1.38))
            offset = float(np.clip(coefficients[1], -48.0, 48.0))
        except np.linalg.LinAlgError:
            gain, offset = 1.0, float(np.median(y - x))
        prediction = x * gain + offset
        residual = float(np.median(np.abs(prediction - y)))
        gains.append(gain)
        offsets.append(offset)
        residuals.append(residual)
    confidence = float(np.clip(1.0 - np.mean(residuals) / 36.0, 0.0, 1.0))
    return ExposureModel(tuple(gains), tuple(offsets), confidence)


def apply_exposure(image_bgr: np.ndarray, model: ExposureModel) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    gain = np.asarray(model.gain_bgr, dtype=np.float32).reshape(1, 1, 3)
    offset = np.asarray(model.offset_bgr, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * gain + offset, 0, 255).astype(np.uint8)
