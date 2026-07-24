from __future__ import annotations

import time

import cv2
import numpy as np

from vsrx.domain.contracts import InpaintRequest, InpaintResult
from vsrx.inpaint.base import BaseInpainter
from vsrx.inpaint.composite import composite_exact


class OpenCVTeleaInpainter(BaseInpainter):
    def __init__(self, radius: float = 3.0, method: str = "telea") -> None:
        self.radius = float(radius)
        self.method = method

    @property
    def name(self) -> str:
        return f"opencv_{self.method}"

    def available(self) -> bool:
        return True

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        start = time.perf_counter()
        flag = cv2.INPAINT_NS if self.method == "ns" else cv2.INPAINT_TELEA
        output: list[np.ndarray] = []
        for index, (frame, mask) in enumerate(zip(request.frames_bgr, request.masks, strict=True)):
            binary = (mask > 0).astype(np.uint8) * 255
            if not np.any(binary):
                output.append(frame.copy())
                continue
            generated = cv2.inpaint(frame, binary, self.radius, flag)
            output.append(composite_exact(frame, generated, binary, feather_radius=2, seed=index))
        return InpaintResult(
            segment_id=request.segment_id,
            frames_bgr=output,
            engine=self.name,
            model_hash="opencv-runtime",
            parameters={"radius": self.radius, "method": self.method},
            peak_vram_mb=0,
            elapsed_seconds=time.perf_counter() - start,
        )
