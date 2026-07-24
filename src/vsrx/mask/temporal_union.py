from __future__ import annotations

import cv2
import numpy as np


def _dis_flow(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    algorithm = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    return algorithm.calc(source, target, None)


def warp_mask_to_target(
    source_mask: np.ndarray, source_bgr: np.ndarray, target_bgr: np.ndarray
) -> np.ndarray:
    # DIS returns source -> target displacement. Inverse remapping is approximated
    # by computing target -> source, which directly samples the source mask.
    flow = _dis_flow(target_bgr, source_bgr)
    height, width = source_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(
        source_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def motion_compensated_union(
    frames_bgr: list[np.ndarray], masks: list[np.ndarray], radius: int = 1
) -> list[np.ndarray]:
    if radius <= 0 or len(masks) < 2:
        return [mask.copy() for mask in masks]
    output: list[np.ndarray] = []
    for index, current in enumerate(masks):
        merged = current.copy()
        for neighbor in range(max(0, index - radius), min(len(masks), index + radius + 1)):
            if neighbor == index or not np.any(masks[neighbor]):
                continue
            warped = warp_mask_to_target(masks[neighbor], frames_bgr[neighbor], frames_bgr[index])
            merged = cv2.bitwise_or(merged, warped)
        output.append(merged)
    return output
