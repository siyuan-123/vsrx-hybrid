from __future__ import annotations

from vsrx.domain.contracts import InpaintRequest, InpaintResult, RouteDecision
from vsrx.domain.enums import Route
from vsrx.domain.errors import ModelUnavailableError, OutOfMemoryError
from vsrx.inpaint.base import BaseInpainter
from vsrx.inpaint.lama_onnx import LamaOnnxInpainter
from vsrx.inpaint.migan_onnx import MiGanOnnxInpainter
from vsrx.inpaint.propainter_client import OfficialProPainterInpainter
from vsrx.inpaint.sttn_client import OfficialSttnInpainter
from vsrx.inpaint.telea import OpenCVTeleaInpainter
from vsrx.utils.config import Config


class InpaintRegistry:
    def __init__(self, config: Config, device_index: int = 0) -> None:
        self.config = config
        self.engines: dict[str, BaseInpainter] = {
            "telea": OpenCVTeleaInpainter(),
            "navier_stokes": OpenCVTeleaInpainter(method="ns"),
            "lama": LamaOnnxInpainter(config, device_index=device_index),
            "migan": MiGanOnnxInpainter(config, device_index=device_index),
            "propainter": OfficialProPainterInpainter(config, device_index=device_index),
            "sttn": OfficialSttnInpainter(config, device_index=device_index),
        }

    def model_status(self) -> dict[str, bool]:
        return {name: engine.available() for name, engine in self.engines.items()}

    def _candidates(self, route: Route) -> list[str]:
        if route == Route.TBE_TELEA:
            return ["telea", "navier_stokes"]
        if route == Route.TBE_LAMA:
            return ["lama", "telea", "navier_stokes"]
        if route == Route.TBE_MIGAN:
            return ["migan", "lama", "telea"]
        if route == Route.OFFICIAL_PROPAINTER:
            return ["propainter", "lama", "sttn", "telea"]
        if route == Route.STTN_FALLBACK:
            return ["sttn", "lama", "telea"]
        return []

    def execute(self, decision: RouteDecision, request: InpaintRequest) -> InpaintResult:
        candidates = self._candidates(decision.route)
        if not candidates:
            return InpaintResult(
                request.segment_id,
                [frame.copy() for frame in request.frames_bgr],
                "none",
                "none",
                {},
                0,
                0.0,
            )
        failures: list[dict[str, str]] = []
        for name in candidates:
            engine = self.engines[name]
            if not engine.available():
                failures.append({"engine": name, "reason": "unavailable"})
                continue
            try:
                result = engine.inpaint(request)
                if name != candidates[0]:
                    result.parameters = {
                        **dict(result.parameters),
                        "fallback_from": candidates[0],
                        "failures": failures,
                    }
                return result
            except (ModelUnavailableError, OutOfMemoryError) as exc:
                failures.append({"engine": name, "reason": exc.code, "message": str(exc)})
                continue
        # OpenCV is expected to remain available, but protect against broken builds.
        raise ModelUnavailableError(
            "no inpainting backend could complete the request",
            details={"route": decision.route.value, "failures": failures},
        )
