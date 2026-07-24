from __future__ import annotations

import numpy as np

from vsrx.domain.contracts import InpaintRequest
from vsrx.inpaint.composite import composite_exact
from vsrx.inpaint.lama_onnx import LamaOnnxInpainter
from vsrx.inpaint.migan_onnx import MiGanOnnxInpainter
from vsrx.inpaint.propainter_client import _ensure_minimum_roi
from vsrx.inpaint.telea import OpenCVTeleaInpainter


def test_composite_preserves_outside_mask_exactly() -> None:
    base = np.full((48, 64, 3), 60, dtype=np.uint8)
    generated = np.full_like(base, 200)
    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    mask[15:32, 18:45] = 255
    result = composite_exact(
        base, generated, mask, feather_radius=3, color_match=False, grain_match=False
    )
    assert np.array_equal(result[mask == 0], base[mask == 0])
    assert np.mean(result[mask > 0]) > np.mean(base[mask > 0])


def test_telea_preserves_outside_and_fills_mask() -> None:
    frame = np.full((50, 80, 3), 80, dtype=np.uint8)
    frame[:, :40] = 35
    mask = np.zeros((50, 80), dtype=np.uint8)
    mask[18:31, 31:49] = 255
    corrupted = frame.copy()
    corrupted[mask > 0] = 255
    request = InpaintRequest("seg", [corrupted], [mask], [0], (25, 10, 55, 40), {})
    result = OpenCVTeleaInpainter().inpaint(request).frames_bgr[0]
    assert np.array_equal(result[mask == 0], corrupted[mask == 0])
    assert float(np.mean(result[mask > 0])) < 245


def test_migan_prepares_official_pipeline_mask_semantics() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255

    image_tensor, mask_tensor = MiGanOnnxInpainter._prepare_input(image, mask, None, None)

    assert image_tensor.shape == (1, 3, 4, 5)
    assert tuple(image_tensor[0, :, 0, 0]) == (30, 20, 10)
    assert np.all(mask_tensor[0, 0, 1:3, 2:4] == 0)
    assert np.all(mask_tensor[0, 0, 0] == 255)


def test_propainter_roi_has_enough_resolution_for_raft() -> None:
    assert _ensure_minimum_roi((90, 60, 130, 85), 320, 180) == (46, 8, 174, 136)
    assert _ensure_minimum_roi((10, 10, 30, 30), 80, 60) == (0, 0, 80, 60)


def test_fixed_onnx_output_is_resized_back_to_original_roi(fast_config) -> None:
    class InputMeta:
        def __init__(self, name: str, channels: int) -> None:
            self.name = name
            self.type = "tensor(float)"
            self.shape = [1, channels, 512, 512]

    class Session:
        def get_inputs(self):
            return [InputMeta("image", 3), InputMeta("mask", 1)]

        def run(self, _outputs, _feeds):
            return [np.zeros((1, 3, 512, 512), dtype=np.float32)]

    engine = LamaOnnxInpainter(fast_config)
    engine._session = Session()
    image = np.zeros((160, 612, 3), dtype=np.uint8)
    mask = np.zeros((160, 612), dtype=np.uint8)

    assert engine._infer_patch(image, mask).shape == image.shape
