from __future__ import annotations

import cv2
import numpy as np

from vsrx.motion.global_registration import GlobalRegistrar


def test_registration_recovers_translation(fast_config) -> None:
    target = np.zeros((120, 160, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x, y = rng.integers(10, 145), rng.integers(10, 105)
        cv2.circle(target, (int(x), int(y)), 3, tuple(int(v) for v in rng.integers(60, 255, 3)), -1)
    transform = np.array([[1, 0, 7], [0, 1, -4], [0, 0, 1]], dtype=np.float64)
    source = cv2.warpPerspective(target, np.linalg.inv(transform), (160, 120))
    registrar = GlobalRegistrar(fast_config)
    result = registrar.estimate(source, target, source_key=1, target_key=2, exclude_mask=None)
    assert result.valid_fraction > 0.75
    assert result.score > 0.5
    assert abs(float(result.transform[0, 2]) - 7) < 2.0
    assert abs(float(result.transform[1, 2]) + 4) < 2.0
