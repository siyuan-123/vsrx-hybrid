from __future__ import annotations

import cv2
import numpy as np


def fade_score(frames_bgr: list[np.ndarray]) -> float:
    if len(frames_bgr) < 3:
        return 0.0
    means = np.array(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() for frame in frames_bgr], dtype=np.float32
    )
    delta = np.diff(means)
    monotonic = max(float((delta >= 0).mean()), float((delta <= 0).mean()))
    span = float(means.max() - means.min()) / 255.0
    return float(np.clip(monotonic * span, 0.0, 1.0))
