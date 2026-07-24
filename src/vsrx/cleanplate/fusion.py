from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_median_fusion(
    images: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    fallback: np.ndarray,
    *,
    row_block: int = 48,
) -> np.ndarray:
    """Memory-bounded per-channel weighted median."""

    if not images:
        return fallback.copy()
    value_stack = np.stack(images, axis=0).astype(np.uint8, copy=False)
    weight_stack = np.stack(weights, axis=0).astype(np.float32, copy=False)
    height, width = fallback.shape[:2]
    output = fallback.copy()
    for y1 in range(0, height, max(1, row_block)):
        y2 = min(height, y1 + max(1, row_block))
        block_weights = weight_stack[:, y1:y2]
        total = block_weights.sum(axis=0)
        available = total > 1e-6
        if not np.any(available):
            continue
        for channel in range(3):
            values = value_stack[:, y1:y2, :, channel]
            order = np.argsort(values, axis=0, kind="stable")
            sorted_values = np.take_along_axis(values, order, axis=0)
            sorted_weights = np.take_along_axis(block_weights, order, axis=0)
            cumulative = np.cumsum(sorted_weights, axis=0)
            median_index = np.argmax(cumulative >= (total[None, ...] * 0.5), axis=0)
            selected = np.take_along_axis(sorted_values, median_index[None, ...], axis=0)[0]
            destination = output[y1:y2, :, channel]
            destination[available] = selected[available]
    return output


def trimmed_mean_fusion(
    images: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    fallback: np.ndarray,
    trim_fraction: float = 0.2,
) -> np.ndarray:
    if not images:
        return fallback.copy()
    values = np.stack(images, axis=0).astype(np.float32)
    weight = np.stack(weights, axis=0).astype(np.float32)
    luminance = values.mean(axis=3)
    order = np.argsort(luminance, axis=0)
    n = len(images)
    trim = min(max(0, int(n * trim_fraction)), max(0, (n - 1) // 2))
    selected_indices = order[trim : n - trim if trim else n]
    expanded = selected_indices[..., None]
    selected_values = np.take_along_axis(values, expanded, axis=0)
    selected_weights = np.take_along_axis(weight, selected_indices, axis=0)
    denominator = selected_weights.sum(axis=0)
    numerator = (selected_values * selected_weights[..., None]).sum(axis=0)
    result = fallback.astype(np.float32)
    valid = denominator > 1e-6
    result[valid] = numerator[valid] / denominator[valid, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def best_reference_fusion(
    images: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    if not images:
        return fallback.copy()
    values = np.stack(images, axis=0)
    score = np.stack(weights, axis=0)
    index = np.argmax(score, axis=0)
    result = fallback.copy()
    for channel in range(3):
        result[..., channel] = np.take_along_axis(values[..., channel], index[None, ...], axis=0)[0]
    result[score.max(axis=0) <= 0] = fallback[score.max(axis=0) <= 0]
    return result
