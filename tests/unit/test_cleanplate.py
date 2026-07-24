from __future__ import annotations

import cv2
import numpy as np
from conftest import make_frame, make_mask

from vsrx.cleanplate.engine import TemporalCleanPlateReconstructor


def test_cleanplate_recovers_pixels_visible_in_other_frames(fast_config) -> None:
    config = fast_config.with_overrides(
        {
            "clean_plate": {
                "minimum_valid_references_per_pixel": 1,
                "preferred_valid_references_per_pixel": 2,
                "max_selected_reference_frames": 4,
            },
            "motion_analysis": {"local_flow": {"forward_backward_check": False}},
        }
    )
    clean = np.full((72, 120, 3), 80, dtype=np.uint8)
    cv2.rectangle(clean, (35, 35), (85, 55), (30, 180, 220), -1)
    frames = []
    masks = []
    for index in range(5):
        image = clean.copy()
        mask = np.zeros(clean.shape[:2], dtype=np.uint8)
        if index == 2:
            mask[35:56, 35:86] = 255
            image[mask > 0] = 250
        frames.append(make_frame(index, image, fps=10))
        masks.append(make_mask(frames[-1], mask))
    result = TemporalCleanPlateReconstructor(config).reconstruct_sequence(
        frames, masks, target_indices=[2]
    )[0]
    region = masks[2].hard_mask > 0
    assert result.mean_coverage_in_mask > 0.2
    assert (
        float(np.mean(np.abs(result.image_bgr[region].astype(int) - clean[region].astype(int))))
        < 30
    )
