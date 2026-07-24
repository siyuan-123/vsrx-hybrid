from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GpuMemoryInfo:
    device_index: int
    total_mb: int
    used_mb: int
    free_mb: int
    name: str


def query_gpu_memory(device_index: int = 0) -> GpuMemoryInfo | None:
    """Query NVIDIA memory without making pynvml a mandatory dependency."""

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return GpuMemoryInfo(
            device_index,
            memory.total // 2**20,
            memory.used // 2**20,
            memory.free // 2**20,
            str(name),
        )
    except Exception:
        pass

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    command = [
        executable,
        f"--id={device_index}",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        name, total, used, free = [item.strip() for item in completed.stdout.strip().split(",", 3)]
        return GpuMemoryInfo(device_index, int(total), int(used), int(free), name)
    except Exception:
        return None


def predict_propainter_vram_mb(width: int, height: int, frame_count: int, fp16: bool = True) -> int:
    """Conservative empirical budget for the official ProPainter pipeline.

    It intentionally overestimates medium-size ROIs and includes flow/model
    workspace.  The isolated worker can update the calibration after real runs.
    """

    pixels = max(1, int(width) * int(height))
    frames = max(1, int(frame_count))
    memory = 900.0 + 0.00027 * pixels * frames + 0.0025 * pixels + 4.0e-9 * pixels * pixels
    if not fp16:
        memory *= 1.45
    return int(round(memory))


class VramCalibrator:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = os.fspath(path) if path is not None else None
        self.scale = 1.0
        if self.path and os.path.exists(self.path):
            try:
                self.scale = float(
                    json.loads(Path(self.path).read_text(encoding="utf-8")).get("scale", 1.0)
                )
            except Exception:
                self.scale = 1.0

    def predict(self, width: int, height: int, frame_count: int, fp16: bool = True) -> int:
        return int(round(predict_propainter_vram_mb(width, height, frame_count, fp16) * self.scale))

    def observe(self, predicted_mb: int, actual_mb: int) -> None:
        if predicted_mb <= 0 or actual_mb <= 0:
            return
        ratio = max(0.65, min(1.8, actual_mb / predicted_mb))
        self.scale = self.scale * 0.8 + ratio * 0.2
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            temporary = self.path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({"scale": self.scale}, handle)
            os.replace(temporary, self.path)
