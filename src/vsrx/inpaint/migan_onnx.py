from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from vsrx.inpaint.lama_onnx import LamaOnnxInpainter
from vsrx.utils.config import Config


class MiGanOnnxInpainter(LamaOnnxInpainter):
    def __init__(
        self, config: Config, model_path: Path | None = None, device_index: int = 0
    ) -> None:
        configured = config.get("spatial_inpainting.migan.model_path")
        selected = Path(
            model_path
            or os.environ.get("VSRX_MIGAN_MODEL", "")
            or configured
            or "models/migan.onnx"
        )
        super().__init__(config, model_path=selected, device_index=device_index)

    @property
    def name(self) -> str:
        return "migan_onnx"

    @staticmethod
    def _prepare_input(
        image: np.ndarray, mask: np.ndarray, image_meta, mask_meta
    ) -> tuple[np.ndarray, np.ndarray]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = image_rgb.transpose(2, 0, 1)[None].astype(np.uint8)
        # 官方 MI-GAN ONNX Pipeline 使用 255 表示保留区，0 表示待修复区。
        mask_tensor = np.where(mask > 0, 0, 255).astype(np.uint8)[None, None]
        return image_tensor, mask_tensor
