from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vsrx.domain.contracts import InpaintRequest, InpaintResult
from vsrx.domain.errors import ModelUnavailableError
from vsrx.inpaint.base import BaseInpainter
from vsrx.inpaint.composite import composite_exact
from vsrx.utils.config import Config
from vsrx.utils.geometry import expand_bbox, mask_bbox
from vsrx.utils.onnxruntime import prepare_onnxruntime


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LamaOnnxInpainter(BaseInpainter):
    def __init__(
        self, config: Config, model_path: Path | None = None, device_index: int = 0
    ) -> None:
        self.config = config
        configured = config.get("spatial_inpainting.lama.model_path")
        self.model_path = Path(
            model_path or os.environ.get("VSRX_LAMA_MODEL", "") or configured or "models/lama.onnx"
        )
        self.device_index = max(0, int(device_index))
        self._session: Any = None
        self._model_hash: str | None = None

    @property
    def name(self) -> str:
        return "lama_onnx"

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return self.model_path.is_file()

    def _providers(self) -> list[Any]:
        ort = prepare_onnxruntime()
        available = set(ort.get_available_providers())
        preferred = self.config.get(
            "spatial_inpainting.lama.provider_order",
            ["onnxruntime_cuda", "openvino", "onnxruntime_cpu"],
        )
        providers: list[Any] = []
        for item in preferred:
            if item == "onnxruntime_cuda" and "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": self.device_index}))
            elif item == "openvino" and "OpenVINOExecutionProvider" in available:
                providers.append("OpenVINOExecutionProvider")
            elif item == "onnxruntime_cpu" and "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
        return providers or ["CPUExecutionProvider"]

    def _load(self) -> Any:
        if self._session is not None:
            return self._session
        if not self.available():
            raise ModelUnavailableError(
                "LaMa ONNX model is unavailable",
                details={"expected_path": str(self.model_path), "env": "VSRX_LAMA_MODEL"},
            )
        ort = prepare_onnxruntime()

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(self.config.get("ocr.inference_threads", 4)))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=self._providers()
        )
        self._model_hash = _file_hash(self.model_path)
        return self._session

    @staticmethod
    def _fixed_hw(input_meta: Any) -> tuple[int | None, int | None]:
        shape = input_meta.shape
        if len(shape) < 4:
            return None, None
        height = shape[-2] if isinstance(shape[-2], int) else None
        width = shape[-1] if isinstance(shape[-1], int) else None
        return height, width

    @staticmethod
    def _prepare_input(
        image: np.ndarray, mask: np.ndarray, image_meta: Any, mask_meta: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_type = str(image_meta.type)
        mask_type = str(mask_meta.type)
        if "uint8" in image_type:
            image_tensor = image_rgb.transpose(2, 0, 1)[None].astype(np.uint8)
        else:
            image_tensor = (image_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        mask_binary = (mask > 0).astype(np.uint8)
        if "uint8" in mask_type:
            mask_tensor = mask_binary[None, None]
        else:
            mask_tensor = mask_binary[None, None].astype(np.float32)
        return image_tensor, mask_tensor

    @staticmethod
    def _decode_output(output: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
        value = np.asarray(output)
        if value.ndim == 4:
            value = value[0]
        if value.ndim == 3 and value.shape[0] in {1, 3, 4}:
            value = value[:3].transpose(1, 2, 0)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=2)
        value = value.astype(np.float32)
        minimum, maximum = float(np.nanmin(value)), float(np.nanmax(value))
        if minimum >= -1.1 and maximum <= 1.1:
            value = (value + 1.0) * 127.5 if minimum < -0.05 else value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
        if value.shape[:2] != expected_shape:
            value = cv2.resize(
                value, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_CUBIC
            )
        return cv2.cvtColor(value, cv2.COLOR_RGB2BGR)

    def _infer_patch(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        session = self._load()
        inputs = session.get_inputs()
        if len(inputs) < 2:
            raise ModelUnavailableError(
                "unsupported LaMa ONNX graph: expected image and mask inputs"
            )
        image_meta = next((item for item in inputs if "mask" not in item.name.lower()), inputs[0])
        mask_meta = next((item for item in inputs if "mask" in item.name.lower()), inputs[1])
        fixed_h, fixed_w = self._fixed_hw(image_meta)
        original_shape = image.shape[:2]
        inference_image, inference_mask = image, mask
        resized_to_fixed = bool(fixed_h and fixed_w and (fixed_h, fixed_w) != original_shape)
        if resized_to_fixed:
            inference_image = cv2.resize(image, (fixed_w, fixed_h), interpolation=cv2.INTER_AREA)
            inference_mask = cv2.resize(mask, (fixed_w, fixed_h), interpolation=cv2.INTER_NEAREST)
        elif fixed_h is None or fixed_w is None:
            # Dynamic LaMa exports commonly require multiples of eight.
            padded_h = int(np.ceil(inference_image.shape[0] / 8.0) * 8)
            padded_w = int(np.ceil(inference_image.shape[1] / 8.0) * 8)
            inference_image = cv2.copyMakeBorder(
                inference_image,
                0,
                padded_h - inference_image.shape[0],
                0,
                padded_w - inference_image.shape[1],
                cv2.BORDER_REFLECT101,
            )
            inference_mask = cv2.copyMakeBorder(
                inference_mask,
                0,
                padded_h - inference_mask.shape[0],
                0,
                padded_w - inference_mask.shape[1],
                cv2.BORDER_CONSTANT,
                value=0,
            )
        image_tensor, mask_tensor = self._prepare_input(
            inference_image, inference_mask, image_meta, mask_meta
        )
        feeds = {image_meta.name: image_tensor, mask_meta.name: mask_tensor}
        output = session.run(None, feeds)[0]
        decoded = self._decode_output(output, inference_image.shape[:2])
        if resized_to_fixed:
            decoded = cv2.resize(
                decoded,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            decoded = decoded[: original_shape[0], : original_shape[1]]
        return decoded

    def _inpaint_frame(self, frame: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        bbox = mask_bbox(mask)
        if bbox is None:
            return frame.copy()
        height, width = frame.shape[:2]
        padding = int(
            round(
                int(self.config.get("spatial_inpainting.lama.roi_padding_px_1080p", 64))
                * height
                / 1080.0
            )
        )
        roi = expand_bbox(bbox, max(16, padding), width, height)
        x1, y1, x2, y2 = roi
        image_crop = frame[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        generated = self._infer_patch(image_crop, mask_crop)
        composed = composite_exact(
            image_crop, generated, mask_crop, feather_radius=max(2, height // 360), seed=seed
        )
        output = frame.copy()
        output[y1:y2, x1:x2] = composed
        return output

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        start = time.perf_counter()
        output = [
            self._inpaint_frame(frame, mask, index)
            for index, (frame, mask) in enumerate(
                zip(request.frames_bgr, request.masks, strict=True)
            )
        ]
        return InpaintResult(
            segment_id=request.segment_id,
            frames_bgr=output,
            engine=self.name,
            model_hash=self._model_hash or _file_hash(self.model_path),
            parameters={
                "model_path": str(self.model_path),
                "providers": self._session.get_providers() if self._session else [],
                "device_index": self.device_index,
            },
            peak_vram_mb=None,
            elapsed_seconds=time.perf_counter() - start,
        )
