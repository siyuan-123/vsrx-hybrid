from .base import BaseInpainter
from .composite import composite_exact
from .lama_onnx import LamaOnnxInpainter
from .migan_onnx import MiGanOnnxInpainter
from .propainter_client import OfficialProPainterInpainter
from .registry import InpaintRegistry
from .sttn_client import OfficialSttnInpainter
from .telea import OpenCVTeleaInpainter

__all__ = [
    "BaseInpainter",
    "InpaintRegistry",
    "LamaOnnxInpainter",
    "MiGanOnnxInpainter",
    "OfficialProPainterInpainter",
    "OfficialSttnInpainter",
    "OpenCVTeleaInpainter",
    "composite_exact",
]
