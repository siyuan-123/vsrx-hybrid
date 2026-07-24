from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from vsrx.domain.contracts import CleanPlateResult, MaskFrame


def largest_component_area(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def aggregate_clean_plate_metrics(
    clean: Sequence[CleanPlateResult],
    masks: Sequence[MaskFrame],
) -> tuple[float, float, float, int]:
    # The intentionally explicit loop avoids indexing tricks and handles empty masks.
    coverage_values: list[float] = []
    confidence_values: list[float] = []
    residual_pixels = 0
    roi_pixels = 0
    largest = 0
    for result, mask in zip(clean, masks, strict=True):
        if not np.any(mask.hard_mask):
            continue
        coverage_values.append(result.mean_coverage_in_mask)
        confidence_values.append(result.mean_confidence_in_mask)
        residual_pixels += int(np.count_nonzero(result.residual_mask))
        roi_pixels += int(np.count_nonzero(mask.hard_mask))
        largest = max(largest, largest_component_area(result.residual_mask))
    return (
        float(np.mean(coverage_values)) if coverage_values else 1.0,
        float(np.mean(confidence_values)) if confidence_values else 1.0,
        residual_pixels / max(roi_pixels, 1),
        largest,
    )
