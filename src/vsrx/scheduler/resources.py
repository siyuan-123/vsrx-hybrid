from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager
from pathlib import Path


class GpuLease(AbstractContextManager["GpuLease"]):
    """Cross-process one-heavy-worker-per-GPU lease using an advisory lock."""

    def __init__(self, root: Path, device_index: int = 0, timeout: float = 3600.0) -> None:
        self.path = root / f"gpu-{device_index}.lock"
        self.timeout = timeout
        self._handle = None
        self._windows_locked = False

    def __enter__(self) -> GpuLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        deadline = time.monotonic() + self.timeout
        if os.name == "nt":
            import msvcrt

            # Lock one byte. Ensure the byte exists before trying to lock it.
            self._handle.seek(0)
            if not self._handle.read(1):
                self._handle.seek(0)
                self._handle.write("0")
                self._handle.flush()
            while True:
                try:
                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                    self._windows_locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for GPU lease: {self.path}"
                        ) from None
                    time.sleep(0.2)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for GPU lease: {self.path}"
                        ) from None
                    time.sleep(0.2)
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                if self._windows_locked:
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            self._windows_locked = False
