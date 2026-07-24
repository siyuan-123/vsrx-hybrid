from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from vsrx.domain.contracts import MaskFrame, VideoFrame
from vsrx.utils.config import Config
from vsrx.utils.geometry import expand_bbox, mask_bbox


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    index: int
    score: float
    temporal_weight: float
    screen_space_availability: float
    ring_similarity: float


class ReferenceSelector:
    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _ring(mask: np.ndarray, radius: int = 12) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return np.ones_like(binary, dtype=bool)
        outer = cv2.dilate(
            binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        )
        return (outer > 0) & (binary == 0)

    def select(
        self,
        target_index: int,
        frames: Sequence[VideoFrame],
        masks: Sequence[MaskFrame],
    ) -> list[ReferenceCandidate]:
        target = frames[target_index]
        target_mask = masks[target_index].hard_mask
        if not np.any(target_mask):
            return []
        max_seconds = float(
            self.config.get("clean_plate.reference_window_seconds.max_each_side", 1.5)
        )
        default_seconds = float(
            self.config.get("clean_plate.reference_window_seconds.default_each_side", 0.9)
        )
        tau = float(self.config.get("clean_plate.temporal_decay_tau_seconds", 0.85))
        max_each_side = int(self.config.get("clean_plate.max_reference_frames_each_side", 36))
        preferred = int(self.config.get("clean_plate.preferred_valid_references_per_pixel", 5))
        configured_total = int(self.config.get("clean_plate.max_selected_reference_frames", 8))
        max_total = max(4, min(max_each_side * 2, configured_total, max(preferred * 3, 4)))

        bbox = mask_bbox(target_mask)
        if bbox is None:
            return []
        height, width = target_mask.shape
        roi = expand_bbox(bbox, max(10, int(round(height / 1080 * 18))), width, height)
        x1, y1, x2, y2 = roi
        target_crop = target.image_bgr[y1:y2, x1:x2]
        target_mask_crop = target_mask[y1:y2, x1:x2]
        ring = self._ring(target_mask_crop)
        target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)

        candidates: list[ReferenceCandidate] = []
        for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
            if index == target_index or frame.ref.shot_id != target.ref.shot_id:
                continue
            distance_seconds = abs(frame.ref.pts_us - target.ref.pts_us) / 1_000_000.0
            if distance_seconds > max_seconds:
                continue
            source_mask_crop = mask.hard_mask[y1:y2, x1:x2]
            target_pixels = target_mask_crop > 0
            availability = (
                float(np.mean(source_mask_crop[target_pixels] == 0))
                if np.any(target_pixels)
                else 1.0
            )
            reference_gray = cv2.cvtColor(frame.image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            ring_valid = ring & (source_mask_crop == 0)
            if int(ring_valid.sum()) >= 64:
                difference = cv2.absdiff(target_gray, reference_gray).astype(np.float32) / 255.0
                ring_similarity = float(
                    np.clip(1.0 - np.median(difference[ring_valid]) / 0.32, 0.0, 1.0)
                )
            else:
                ring_similarity = 0.35
            temporal_weight = float(np.exp(-distance_seconds / max(tau, 0.05)))
            # Availability dominates because references that expose new subtitle
            # pixels are more useful than merely adjacent frames.
            score = availability * 0.50 + temporal_weight * 0.27 + ring_similarity * 0.23
            if distance_seconds <= default_seconds:
                score += 0.04
            candidates.append(
                ReferenceCandidate(index, score, temporal_weight, availability, ring_similarity)
            )

        before = sorted(
            (item for item in candidates if item.index < target_index),
            key=lambda item: item.score,
            reverse=True,
        )
        after = sorted(
            (item for item in candidates if item.index > target_index),
            key=lambda item: item.score,
            reverse=True,
        )
        selected: list[ReferenceCandidate] = []

        def take_diverse(items: list[ReferenceCandidate], quota: int) -> None:
            # Greedy score with a small spacing bonus.  This avoids spending all
            # registrations on nearly identical neighbouring frames.
            remaining = list(items)
            while (
                remaining
                and sum(
                    1
                    for item in selected
                    if (item.index < target_index) == (items[0].index < target_index)
                )
                < quota
            ):
                best = max(
                    remaining,
                    key=lambda item: (
                        item.score
                        + min(
                            0.10,
                            0.018
                            * min(
                                (abs(item.index - chosen.index) for chosen in selected), default=6
                            ),
                        )
                    ),
                )
                selected.append(best)
                remaining.remove(best)

        # Prefer evidence from both temporal directions when available.
        before_quota = max_total // 2
        after_quota = max_total - before_quota
        take_diverse(before, before_quota)
        take_diverse(after, after_quota)
        selected_ids = {item.index for item in selected}
        for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
            if len(selected) >= max_total:
                break
            if item.index not in selected_ids:
                selected.append(item)
                selected_ids.add(item.index)
        return sorted(selected[:max_total], key=lambda item: item.score, reverse=True)
