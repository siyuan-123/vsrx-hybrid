from __future__ import annotations

import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

_dll_directory_handles: list[Any] = []
_prepared = False


def prepare_onnxruntime():
    global _prepared

    if os.name == "nt" and not _prepared:
        site_packages = Path(sys.prefix) / "Lib/site-packages"
        candidates = [site_packages / "torch/lib"]
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(sorted(nvidia_root.glob("*/bin")))
        directories = [str(path.resolve()) for path in candidates if path.is_dir()]
        if directories:
            current = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join([*directories, current])
            # cuDNN 会按文件名延迟加载 NVRTC，句柄需保持到进程结束。
            for directory in directories:
                try:
                    _dll_directory_handles.append(os.add_dll_directory(directory))
                except OSError:
                    continue
        _prepared = True

    import onnxruntime as ort

    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is not None:
        with suppress(Exception):
            preload_dlls()
    return ort
