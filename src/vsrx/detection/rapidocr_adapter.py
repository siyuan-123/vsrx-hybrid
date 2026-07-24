from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from vsrx.domain.contracts import Polygon, TextDetection, VideoFrame
from vsrx.domain.errors import ModelUnavailableError
from vsrx.utils.config import Config
from vsrx.utils.geometry import clamp_bbox, polygon_angle
from vsrx.utils.onnxruntime import prepare_onnxruntime

logger = logging.getLogger(__name__)


class RapidOCRTextDetector:
    """Lazy detection-only PP-OCRv6 adapter.

    The adapter deliberately instantiates RapidOCR's low-level TextDetector so
    recognition/classification sessions are not loaded into memory.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engines: dict[tuple[str, str], Any] = {}
        self._failed_engines: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def _backend_candidates(self) -> list[str]:
        order = list(self.config.get("ocr.backend_order", ["onnxruntime_cpu"]))
        result: list[str] = []
        for item in order:
            normalized = str(item).lower()
            if normalized == "onnxruntime_cuda":
                try:
                    ort = prepare_onnxruntime()
                    if "CUDAExecutionProvider" in ort.get_available_providers():
                        result.append("onnxruntime_cuda")
                except Exception:
                    continue
            elif normalized == "onnxruntime_cpu":
                result.append("onnxruntime_cpu")
            elif normalized == "openvino":
                try:
                    import openvino  # noqa: F401

                    result.append("openvino")
                except Exception:
                    continue
        return list(dict.fromkeys(result)) or ["onnxruntime_cpu"]

    @staticmethod
    def _tier_enum(tier: str):
        from rapidocr import ModelType

        mapping = {
            "tiny": ModelType.TINY,
            "small": ModelType.SMALL,
            "medium": ModelType.MEDIUM,
        }
        if tier not in mapping:
            raise ValueError(f"unsupported OCR tier: {tier}")
        return mapping[tier]

    def _build_engine(self, tier: str, backend: str):
        try:
            from rapidocr import EngineType, LangDet, OCRVersion
            from rapidocr.ch_ppocr_det.main import TextDetector
            from rapidocr.main import DEFAULT_CFG_PATH, root_dir
            from rapidocr.utils.parse_parameters import ParseParams
        except ImportError as exc:
            raise ModelUnavailableError(
                "RapidOCR is not installed; install vsrx-hybrid[ocr]"
            ) from exc

        engine_type = EngineType.OPENVINO if backend == "openvino" else EngineType.ONNXRUNTIME
        cfg = ParseParams.load(DEFAULT_CFG_PATH)
        if engine_type == EngineType.ONNXRUNTIME:
            cfg.EngineConfig.onnxruntime.use_cuda = backend == "onnxruntime_cuda"
            cfg.EngineConfig.onnxruntime.cuda_ep_cfg.device_id = 0
        params = {
            "Global.log_level": "ERROR",
            "Det.engine_type": engine_type,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": self._tier_enum(tier),
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.limit_side_len": int(self.config.get("ocr.input_limit_side_len", 736)),
            "Det.limit_type": str(self.config.get("ocr.input_limit_type", "min")),
            "Det.thresh": float(self.config.get("ocr.det_threshold", 0.30)),
            "Det.box_thresh": float(self.config.get("ocr.box_threshold", 0.50)),
            "Det.unclip_ratio": float(self.config.get("ocr.unclip_ratio", 1.60)),
            "Det.use_dilation": bool(self.config.get("ocr.use_detector_dilation", True)),
            "Det.score_mode": str(self.config.get("ocr.score_mode", "fast")),
        }
        cfg = ParseParams.update_batch(cfg, params)
        cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
        cfg.Det.model_root_dir = cfg.Global.model_root_dir or root_dir / "models"
        return TextDetector(cfg.Det)

    def _engine(self, tier: str):
        errors: list[str] = []
        with self._lock:
            for backend in self._backend_candidates():
                key = (tier, backend)
                if key in self._engines:
                    return self._engines[key]
                if key in self._failed_engines:
                    errors.append(f"{backend}: {self._failed_engines[key]}")
                    continue
                try:
                    engine = self._build_engine(tier, backend)
                    self._engines[key] = engine
                    logger.info("loaded OCR detector tier=%s backend=%s", tier, backend)
                    return engine
                except Exception as exc:
                    self._failed_engines[key] = str(exc)
                    errors.append(f"{backend}: {exc}")
            raise ModelUnavailableError(
                "failed to initialize PP-OCRv6 detector", details={"errors": errors}
            )

    def detect(
        self,
        frames: Sequence[VideoFrame],
        rois: Sequence[tuple[int, int, int, int]] | None = None,
        tier: str = "small",
    ) -> list[TextDetection]:
        engine = self._engine(tier)
        detections: list[TextDetection] = []
        for frame in frames:
            image = frame.image_bgr
            height, width = image.shape[:2]
            scan_rois = list(rois) if rois else [(0, 0, width, height)]
            for roi in scan_rois:
                x1, y1, x2, y2 = clamp_bbox(roi, width, height)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                patch = image[y1:y2, x1:x2]
                try:
                    result = engine(patch)
                except Exception as exc:
                    logger.warning("OCR detection failed at pts=%s: %s", frame.ref.pts_us, exc)
                    continue
                boxes = getattr(result, "boxes", None)
                scores = getattr(result, "scores", None)
                if boxes is None:
                    continue
                scores = scores or [1.0] * len(boxes)
                for box, score in zip(np.asarray(boxes), scores, strict=False):
                    points = np.asarray(box, dtype=np.float32)
                    points[:, 0] += x1
                    points[:, 1] += y1
                    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    detections.append(
                        TextDetection(
                            frame=frame.ref,
                            polygon=Polygon(tuple((float(px), float(py)) for px, py in points)),
                            confidence=float(score),
                            angle_degrees=polygon_angle(points),
                            detector_tier=tier,
                            source_roi=(x1, y1, x2, y2),
                        )
                    )
        return detections


class HeuristicTextDetector:
    """Dependency-free fallback for high-contrast subtitle-like overlays."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def detect(
        self,
        frames: Sequence[VideoFrame],
        rois: Sequence[tuple[int, int, int, int]] | None = None,
        tier: str = "heuristic",
    ) -> list[TextDetection]:
        detections: list[TextDetection] = []
        for frame in frames:
            image = frame.image_bgr
            height, width = image.shape[:2]
            scan_rois = list(rois) if rois else [(0, 0, width, height)]
            for roi in scan_rois:
                x1, y1, x2, y2 = clamp_bbox(roi, width, height)
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                kernel_width = max(9, int(round(patch.shape[1] * 0.02))) | 1
                gradient = cv2.morphologyEx(
                    gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                )
                _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
                )
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    area = bw * bh
                    if area < 80 or bw < 12 or bh < 5:
                        continue
                    aspect = bw / max(bh, 1)
                    if aspect < 1.2 and bh < 20:
                        continue
                    extent = cv2.contourArea(contour) / max(area, 1)
                    if extent < 0.08:
                        continue
                    points = (
                        (float(x1 + bx), float(y1 + by)),
                        (float(x1 + bx + bw), float(y1 + by)),
                        (float(x1 + bx + bw), float(y1 + by + bh)),
                        (float(x1 + bx), float(y1 + by + bh)),
                    )
                    detections.append(
                        TextDetection(
                            frame=frame.ref,
                            polygon=Polygon(points),
                            confidence=float(min(0.75, 0.35 + extent)),
                            angle_degrees=0.0,
                            detector_tier=tier,
                            source_roi=(x1, y1, x2, y2),
                        )
                    )
        return detections


class CascadedTextDetector:
    def __init__(self, config: Config) -> None:
        self.primary = RapidOCRTextDetector(config)
        self.fallback = HeuristicTextDetector(config)

    def detect(
        self, frames: Sequence[VideoFrame], rois=None, tier: str = "small"
    ) -> list[TextDetection]:
        try:
            return self.primary.detect(frames, rois, tier)
        except ModelUnavailableError as exc:
            logger.warning("RapidOCR unavailable, using heuristic detector: %s", exc)
            return self.fallback.detect(frames, rois, tier="heuristic")
