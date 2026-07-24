from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from vsrx.domain.contracts import InpaintRequest, InpaintResult
from vsrx.domain.errors import ExternalToolError, ModelUnavailableError, OutOfMemoryError
from vsrx.inpaint.base import BaseInpainter
from vsrx.inpaint.composite import composite_exact
from vsrx.routing.budget import query_gpu_memory
from vsrx.utils.config import Config
from vsrx.utils.geometry import clamp_bbox


def _tree_identity(repo: Path) -> str:
    digest = hashlib.sha256()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        digest.update(commit.encode())
    except Exception:
        digest.update(str(repo.resolve()).encode())
    weights = repo / "weights"
    if weights.exists():
        for path in sorted(weights.rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(
                    f"{path.relative_to(repo)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                )
    return digest.hexdigest()


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ExternalToolError(f"failed to encode temporary image: {path}")
    path.write_bytes(encoded.tobytes())


def _ensure_minimum_roi(
    roi: tuple[int, int, int, int], width: int, height: int, min_side: int = 128
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_bbox(roi, width, height)
    target_width = min(width, max(min_side, x2 - x1))
    target_height = min(height, max(min_side, y2 - y1))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    x1 = min(max(0, round(center_x - target_width / 2.0)), width - target_width)
    y1 = min(max(0, round(center_y - target_height / 2.0)), height - target_height)
    return x1, y1, x1 + target_width, y1 + target_height


def _load_generated_frames(root: Path, expected: int) -> list[np.ndarray]:
    candidates: list[tuple[int, Path]] = []
    for directory in [root, *[item for item in root.rglob("*") if item.is_dir()]]:
        files = sorted([*directory.glob("*.png"), *directory.glob("*.jpg")])
        if files:
            candidates.append((len(files), directory))
    if candidates:
        _, directory = min(candidates, key=lambda item: abs(item[0] - expected))
        files = sorted([*directory.glob("*.png"), *directory.glob("*.jpg")])
        frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in files[:expected]]
        if len(frames) == expected and all(frame is not None for frame in frames):
            return frames  # type: ignore[return-value]
    videos = sorted([*root.rglob("*.mp4"), *root.rglob("*.mkv"), *root.rglob("*.avi")])
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        frames: list[np.ndarray] = []
        while len(frames) < expected:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if len(frames) == expected:
            return frames
    raise ExternalToolError(
        "ProPainter completed but generated frame output could not be located",
        details={"root": str(root), "expected_frames": expected},
    )


class OfficialProPainterInpainter(BaseInpainter):
    def __init__(
        self, config: Config, repo_path: Path | None = None, device_index: int = 0
    ) -> None:
        self.config = config
        configured = config.get("video_inpainting.propainter.repo_path")
        self.repo_path = Path(
            repo_path
            or os.environ.get("VSRX_PROPAINTER_REPO", "")
            or configured
            or "third_party/ProPainter"
        )
        self.device_index = device_index

    @property
    def name(self) -> str:
        return "official_propainter"

    @property
    def script(self) -> Path:
        return self.repo_path / "inference_propainter.py"

    def available(self) -> bool:
        return self.script.is_file() and (self.repo_path / "weights").exists()

    def _run_monitored(
        self, command: list[str], cwd: Path, timeout: float
    ) -> tuple[str, str, int | None]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.device_index)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        peak: int | None = None
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            info = query_gpu_memory(0 if "CUDA_VISIBLE_DEVICES" in env else self.device_index)
            if info is not None:
                peak = max(peak or 0, info.used_mb)
            if time.monotonic() > deadline:
                process.kill()
                stdout, stderr = process.communicate()
                raise ExternalToolError(
                    "ProPainter timed out",
                    details={"stdout": stdout[-2000:], "stderr": stderr[-4000:]},
                )
            time.sleep(0.25)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            combined = f"{stdout}\n{stderr}".lower()
            if "out of memory" in combined or "cuda oom" in combined:
                raise OutOfMemoryError(
                    "ProPainter exhausted GPU memory", details={"stderr": stderr[-5000:]}
                )
            raise ExternalToolError(
                f"ProPainter failed with exit code {process.returncode}",
                details={"stdout": stdout[-4000:], "stderr": stderr[-6000:], "command": command},
            )
        return stdout, stderr, peak

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        if not self.available():
            raise ModelUnavailableError(
                "official ProPainter repository/weights are unavailable",
                details={"expected_repo": str(self.repo_path), "env": "VSRX_PROPAINTER_REPO"},
            )
        start = time.perf_counter()
        first = request.frames_bgr[0]
        height, width = first.shape[:2]
        # RAFT 相关金字塔需要足够的空间分辨率，小 ROI 会在池化阶段变成空张量。
        roi = _ensure_minimum_roi(request.roi_xyxy, width, height)
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return InpaintResult(
                request.segment_id,
                [item.copy() for item in request.frames_bgr],
                self.name,
                _tree_identity(self.repo_path),
                {},
                0,
                0.0,
            )

        keep = bool(request.context.get("keep_runtime_artifacts", False))
        base_temp = request.context.get("runtime_dir")
        if base_temp:
            Path(base_temp).mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix="vsrx-propainter-", dir=str(base_temp) if base_temp else None)
        )
        frame_dir = temporary / "frames"
        mask_dir = temporary / "masks"
        output_dir = temporary / "output"
        frame_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        try:
            cropped_frames: list[np.ndarray] = []
            cropped_masks: list[np.ndarray] = []
            for index, (frame, mask) in enumerate(
                zip(request.frames_bgr, request.masks, strict=True)
            ):
                crop = frame[y1:y2, x1:x2]
                crop_mask = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
                cropped_frames.append(crop)
                cropped_masks.append(crop_mask)
                _write_png(frame_dir / f"{index:05d}.png", crop)
                _write_png(mask_dir / f"{index:05d}.png", crop_mask)

            cfg = self.config.get("video_inpainting.propainter", {})
            fps = float(request.context.get("fps", 25.0))
            command = [
                sys.executable,
                str(self.script),
                "-i",
                str(frame_dir),
                "-m",
                str(mask_dir),
                "-o",
                str(output_dir),
                "--mask_dilation",
                "0",
                "--ref_stride",
                str(int(cfg.get("ref_stride", 12))),
                "--neighbor_length",
                str(int(cfg.get("neighbor_length", 6))),
                "--subvideo_length",
                str(int(cfg.get("subvideo_length", 32))),
                "--save_fps",
                str(max(1, round(fps))),
                "--save_frames",
            ]
            if bool(cfg.get("fp16", True)):
                command.append("--fp16")
            timeout = max(300.0, len(cropped_frames) * 30.0)
            _, _, peak = self._run_monitored(command, self.repo_path, timeout)
            generated = _load_generated_frames(output_dir, len(cropped_frames))

            output: list[np.ndarray] = []
            for index, (full, source_crop, generated_crop, crop_mask) in enumerate(
                zip(request.frames_bgr, cropped_frames, generated, cropped_masks, strict=True)
            ):
                if generated_crop.shape[:2] != source_crop.shape[:2]:
                    generated_crop = cv2.resize(
                        generated_crop,
                        (source_crop.shape[1], source_crop.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                composed_crop = composite_exact(
                    source_crop,
                    generated_crop,
                    crop_mask,
                    feather_radius=max(2, source_crop.shape[0] // 240),
                    seed=index,
                )
                composed = full.copy()
                composed[y1:y2, x1:x2] = composed_crop
                output.append(composed)
            return InpaintResult(
                segment_id=request.segment_id,
                frames_bgr=output,
                engine=self.name,
                model_hash=_tree_identity(self.repo_path),
                parameters={
                    "roi_xyxy": roi,
                    "neighbor_length": int(cfg.get("neighbor_length", 6)),
                    "ref_stride": int(cfg.get("ref_stride", 12)),
                    "subvideo_length": int(cfg.get("subvideo_length", 32)),
                    "fp16": bool(cfg.get("fp16", True)),
                },
                peak_vram_mb=peak,
                elapsed_seconds=time.perf_counter() - start,
            )
        finally:
            if not keep:
                shutil.rmtree(temporary, ignore_errors=True)
