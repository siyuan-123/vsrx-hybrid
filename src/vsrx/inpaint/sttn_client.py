from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np

from vsrx.domain.contracts import InpaintRequest, InpaintResult
from vsrx.domain.errors import ExternalToolError, ModelUnavailableError, OutOfMemoryError
from vsrx.inpaint.base import BaseInpainter
from vsrx.inpaint.composite import composite_exact
from vsrx.inpaint.propainter_client import _load_generated_frames, _tree_identity, _write_png
from vsrx.utils.config import Config
from vsrx.utils.geometry import clamp_bbox


def _write_lossless_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("ffv1", rate=max(1, round(fps)))
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "bgr0"
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


class OfficialSttnInpainter(BaseInpainter):
    """Emergency adapter for the original STTN repository.

    The original repository has several forks with slightly different CLIs.
    ``VSRX_STTN_COMMAND`` can override invocation using placeholders:
    {python}, {script}, {video}, {mask}, {checkpoint}, {output}.
    """

    def __init__(
        self, config: Config, repo_path: Path | None = None, device_index: int = 0
    ) -> None:
        self.config = config
        self.repo_path = Path(
            repo_path
            or os.environ.get("VSRX_STTN_REPO", "")
            or config.get("video_inpainting.sttn.repo_path")
            or "third_party/STTN"
        )
        self.device_index = max(0, int(device_index))
        self.checkpoint = Path(
            os.environ.get("VSRX_STTN_CHECKPOINT", "")
            or config.get("video_inpainting.sttn.checkpoint")
            or self.repo_path / "checkpoints/sttn.pth"
        )

    @property
    def name(self) -> str:
        return "sttn_fallback"

    @property
    def script(self) -> Path:
        return self.repo_path / "test.py"

    def available(self) -> bool:
        return self.script.is_file() and self.checkpoint.is_file()

    def _runtime_script(self, temporary: Path) -> Path:
        source = self.script.read_text(encoding="utf-8")
        patched = source.replace(
            'torch.device("cuda:1" if torch.cuda.is_available() else "cpu")',
            'torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
        )
        if patched == source:
            return self.script
        script = temporary / "vsrx_sttn_test.py"
        script.write_text(patched, encoding="utf-8")
        return script

    def _command(self, video: Path, masks: Path, output: Path, script: Path) -> list[str]:
        template = os.environ.get("VSRX_STTN_COMMAND")
        values = {
            "python": sys.executable,
            "script": str(script),
            "video": str(video),
            "mask": str(masks),
            "checkpoint": str(self.checkpoint),
            "output": str(output),
        }
        if template:
            return [part.format(**values) for part in shlex.split(template)]
        command = [
            sys.executable,
            str(script),
            "--video",
            str(video),
            "--mask",
            str(masks),
            "--ckpt",
            str(self.checkpoint),
        ]
        try:
            help_text = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
            if "--output" in help_text:
                command.extend(["--output", str(output)])
            elif "--save_path" in help_text:
                command.extend(["--save_path", str(output)])
        except Exception:
            pass
        return command

    def inpaint(self, request: InpaintRequest) -> InpaintResult:
        if not self.available():
            raise ModelUnavailableError(
                "STTN repository/checkpoint is unavailable",
                details={"repo": str(self.repo_path), "checkpoint": str(self.checkpoint)},
            )
        start = time.perf_counter()
        height, width = request.frames_bgr[0].shape[:2]
        x1, y1, x2, y2 = clamp_bbox(request.roi_xyxy, width, height)
        runtime_dir = request.context.get("runtime_dir")
        if runtime_dir:
            Path(runtime_dir).mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="vsrx-sttn-", dir=runtime_dir))
        try:
            masks_dir = temp / "masks"
            output_dir = temp / "output"
            masks_dir.mkdir()
            output_dir.mkdir()
            crops = [frame[y1:y2, x1:x2] for frame in request.frames_bgr]
            crop_masks = [(mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255 for mask in request.masks]
            video = temp / "input.mkv"
            _write_lossless_video(video, crops, float(request.context.get("fps", 25.0)))
            for index, mask in enumerate(crop_masks):
                _write_png(masks_dir / f"{index:05d}.png", mask)
            script = self._runtime_script(temp)
            command = self._command(video, masks_dir, output_dir, script)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.device_index)
            env["PYTHONPATH"] = os.pathsep.join(
                item for item in (str(self.repo_path.resolve()), env.get("PYTHONPATH", "")) if item
            )
            process = subprocess.run(
                command,
                cwd=str(self.repo_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=max(300, len(crops) * 30),
            )
            if process.returncode != 0:
                combined = (process.stdout + process.stderr).lower()
                if "out of memory" in combined:
                    raise OutOfMemoryError(
                        "STTN exhausted memory", details={"stderr": process.stderr[-4000:]}
                    )
                raise ExternalToolError(
                    "STTN failed",
                    details={"stdout": process.stdout[-3000:], "stderr": process.stderr[-5000:]},
                )
            upstream_output = Path(f"{masks_dir}_result.mp4")
            if upstream_output.is_file():
                shutil.move(str(upstream_output), str(output_dir / upstream_output.name))
            generated = _load_generated_frames(
                output_dir if any(output_dir.iterdir()) else temp, len(crops)
            )
            output: list[np.ndarray] = []
            for index, (full, crop, generated_crop, mask) in enumerate(
                zip(request.frames_bgr, crops, generated, crop_masks, strict=True)
            ):
                generated_crop = cv2.resize(
                    generated_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                composed_crop = composite_exact(
                    crop, generated_crop, mask, feather_radius=3, seed=index
                )
                frame = full.copy()
                frame[y1:y2, x1:x2] = composed_crop
                output.append(frame)
            return InpaintResult(
                request.segment_id,
                output,
                self.name,
                _tree_identity(self.repo_path),
                {"roi_xyxy": (x1, y1, x2, y2), "device_index": self.device_index},
                None,
                time.perf_counter() - start,
            )
        finally:
            if not bool(request.context.get("keep_runtime_artifacts", False)):
                shutil.rmtree(temp, ignore_errors=True)
