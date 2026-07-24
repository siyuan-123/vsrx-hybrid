from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from conftest import make_frame

from vsrx.mask.external import ExternalMaskProvider


def test_external_masks_by_index_pts_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    root.mkdir()
    image = np.zeros((30, 40), dtype=np.uint8)
    image[10:20, 5:25] = 255
    cv2.imwrite(str(root / "00000000.png"), image)
    cv2.imwrite(str(root / "pts_100000.png"), image)
    (root / "masks.json").write_text(json.dumps({"frames": {"frame_2": "00000000.png"}}))
    provider = ExternalMaskProvider(root)
    frames = [make_frame(i, np.zeros((30, 40, 3), dtype=np.uint8), fps=10) for i in range(3)]
    masks = provider.generate(frames)
    assert np.count_nonzero(masks[0].hard_mask) > 0
    assert np.count_nonzero(masks[1].hard_mask) > 0  # matched by PTS
    assert np.count_nonzero(masks[2].hard_mask) > 0  # matched by manifest
    assert provider.discover_rois(40, 30)
    assert len(provider.digest) == 64


def test_jpeg_noise_does_not_turn_entire_frame_into_mask(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    root.mkdir()
    image = np.zeros((40, 50), dtype=np.uint8)
    image[12:25, 10:35] = 255
    cv2.imwrite(str(root / "00000000.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    provider = ExternalMaskProvider(root)
    frame = make_frame(0, np.zeros((40, 50, 3), dtype=np.uint8))
    mask = provider.generate([frame])[0].hard_mask
    assert 100 < np.count_nonzero(mask) < 800


def test_external_mask_list_manifest(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    root.mkdir()
    image = np.zeros((24, 32), dtype=np.uint8)
    image[8:16, 7:22] = 255
    cv2.imwrite(str(root / "mask.png"), image)
    (root / "masks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_base": "microseconds",
                "frames": [{"frame_index": 0, "pts_us": 0, "path": "mask.png"}],
            }
        ),
        encoding="utf-8",
    )
    provider = ExternalMaskProvider(root)
    frame = make_frame(0, np.zeros((24, 32, 3), dtype=np.uint8))
    assert np.count_nonzero(provider.generate([frame])[0].hard_mask) > 0
