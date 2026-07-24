from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def clamp_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(x1, min(width, int(x2)))
    y2 = max(y1, min(height, int(y2)))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int], padding: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return clamp_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height)


def union_bboxes(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    items = list(boxes)
    if not items:
        return None
    return (
        min(x[0] for x in items),
        min(x[1] for x in items),
        max(x[2] for x in items),
        max(x[3] for x in items),
    )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w, y + h


def polygon_angle(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    edge = points[1] - points[0]
    return float(np.degrees(np.arctan2(edge[1], edge[0])))
