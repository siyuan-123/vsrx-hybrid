from __future__ import annotations

import cv2
import numpy as np

from vsrx.utils.geometry import clamp_bbox, expand_bbox


class BackgroundPanelDetector:
    """Conservative detector for subtitle background bars and rounded panels."""

    def detect(
        self,
        image_bgr: np.ndarray,
        text_bbox: tuple[int, int, int, int],
        *,
        minimum_score: float,
    ) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        text_height = max(1, text_bbox[3] - text_bbox[1])
        search = expand_bbox(text_bbox, max(8, int(text_height * 0.8)), width, height)
        x1, y1, x2, y2 = search
        patch = image_bgr[y1:y2, x1:x2]
        result = np.zeros((height, width), dtype=np.uint8)
        if patch.size == 0 or minimum_score < 0.70:
            return result
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        tx1, ty1, tx2, ty2 = text_bbox
        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < text_area * 1.05 or area > text_area * 7.0:
                continue
            global_box = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)
            contains = (
                global_box[0] <= tx1
                and global_box[1] <= ty1
                and global_box[2] >= tx2
                and global_box[3] >= ty2
            )
            if not contains:
                continue
            candidate = lab[by : by + bh, bx : bx + bw]
            if candidate.size == 0:
                continue
            variance = float(candidate[..., 0].std())
            fill_ratio = cv2.contourArea(contour) / max(area, 1)
            rectangularity = min(1.0, fill_ratio * 1.5)
            uniformity = float(np.clip(1.0 - variance / 34.0, 0.0, 1.0))
            score = 0.58 * uniformity + 0.42 * rectangularity
            if score >= minimum_score and (best is None or score > best[0]):
                best = (score, global_box)
        if best is not None:
            bx1, by1, bx2, by2 = clamp_bbox(best[1], width, height)
            cv2.rectangle(result, (bx1, by1), (max(bx1, bx2 - 1), max(by1, by2 - 1)), 255, -1)
        return result
