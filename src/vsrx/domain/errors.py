from __future__ import annotations


class VSRXError(RuntimeError):
    code = "VSRX_ERROR"
    transient = False

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(VSRXError):
    code = "CONFIGURATION_ERROR"


class ExternalToolError(VSRXError):
    code = "EXTERNAL_TOOL_ERROR"


class MediaProbeError(VSRXError):
    code = "MEDIA_PROBE_ERROR"


class DecodeError(VSRXError):
    code = "DECODE_ERROR"


class EncodeError(VSRXError):
    code = "ENCODE_ERROR"


class ModelUnavailableError(VSRXError):
    code = "MODEL_UNAVAILABLE"


class ModelHashMismatchError(VSRXError):
    code = "MODEL_HASH_MISMATCH"


class OutOfMemoryError(VSRXError):
    code = "OUT_OF_MEMORY"
    transient = True


class CancelledError(VSRXError):
    code = "CANCELLED"


class QualityGateError(VSRXError):
    code = "QUALITY_GATE_FAILED"
