from __future__ import annotations

import cv2
import numpy as np
from conftest import make_frame

from vsrx.domain.contracts import Polygon, Shot, TextDetection
from vsrx.tracking.association import HungarianTrackBuilder


def detection(frame, x: int) -> TextDetection:
    return TextDetection(
        frame.ref,
        Polygon(((x, 60), (x + 45, 60), (x + 45, 76), (x, 76))),
        0.9,
        0.0,
    )


def test_hungarian_tracking_interpolates_missing_frame(fast_config) -> None:
    frames = []
    detections = []
    for index in range(5):
        image = np.zeros((90, 140, 3), dtype=np.uint8)
        cv2.rectangle(image, (35 + index, 60), (80 + index, 76), (255, 255, 255), -1)
        frame = make_frame(index, image, fps=10)
        frames.append(frame)
        if index != 2:
            detections.append(detection(frame, 35 + index))
    shot = Shot(0, 0, 500_000, 0, 5, 1.0)
    tracks = HungarianTrackBuilder(fast_config).build_raw(frames, detections, shot)
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 5
    assert any(item.propagated for item in tracks[0].detections)
