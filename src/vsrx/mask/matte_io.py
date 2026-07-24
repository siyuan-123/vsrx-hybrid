from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vsrx.domain.contracts import FrameRef, MaskFrame


def export_masks(path: Path, masks: list[MaskFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"mask_{index:06d}": mask.hard_mask for index, mask in enumerate(masks)}
    metadata = [
        {
            "frame_index": mask.frame.frame_index,
            "pts_us": mask.frame.pts_us,
            "shot_id": mask.frame.shot_id,
            "source_track_ids": list(mask.source_track_ids),
            "confidence": mask.confidence,
        }
        for mask in masks
    ]
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return path


def import_masks(path: Path) -> list[MaskFrame]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    result: list[MaskFrame] = []
    for index, item in enumerate(metadata):
        hard = archive[f"mask_{index:06d}"].astype(np.uint8)
        result.append(
            MaskFrame(
                frame=FrameRef(int(item["frame_index"]), int(item["pts_us"]), int(item["shot_id"])),
                hard_mask=hard,
                soft_alpha=(hard > 0).astype(np.float32),
                source_track_ids=tuple(item["source_track_ids"]),
                confidence=float(item["confidence"]),
                mask_ratio_of_frame=float((hard > 0).mean()),
                expanded_bbox_xyxy=None,
            )
        )
    return result
