from __future__ import annotations

import numpy as np
from conftest import make_frame, make_mask

from vsrx.quality.gate import AutomaticQualityGate
from vsrx.quality.preservation import preservation_error


def test_preservation_detects_outside_modification() -> None:
    source = np.zeros((30, 40, 3), dtype=np.uint8)
    output = source.copy()
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    output[2, 2] = 255
    maximum, changed = preservation_error([source], [output], [mask])
    assert maximum == 255
    assert changed > 0


def test_quality_gate_passes_identical_empty_mask(fast_config) -> None:
    config = fast_config.with_overrides({"quality_control": {"enabled": True}})
    image = np.full((32, 48, 3), 70, dtype=np.uint8)
    frame = make_frame(0, image)
    mask = make_mask(frame, np.zeros(image.shape[:2], dtype=np.uint8))
    report = AutomaticQualityGate(config, detector=None).evaluate(
        [image], [image.copy()], [mask], "seg"
    )
    assert report.passed
