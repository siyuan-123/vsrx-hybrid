from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from vsrx.app.config_loader import load_runtime_config
from vsrx.domain.contracts import FrameRef, MaskFrame, VideoFrame
from vsrx.utils.image import soft_alpha_from_mask


@pytest.fixture
def fast_config():
    return load_runtime_config(
        None,
        profile="fast",
        overrides=[
            "quality_control.enabled=false",
            "runtime.log_level=ERROR",
            "scene_detection.min_scene_len_frames=4",
            "routing.segment_max_frames=16",
            "video_inpainting.propainter.default_chunk_frames=16",
            "video_inpainting.propainter.max_chunk_frames=16",
        ],
    )


def make_frame(index: int, image: np.ndarray, fps: int = 12, shot_id: int = 0) -> VideoFrame:
    duration = int(round(1_000_000 / fps))
    return VideoFrame(
        FrameRef(index, index * duration, shot_id),
        np.ascontiguousarray(image),
        index,
        Fraction(1, fps),
        duration,
    )


def make_mask(frame: VideoFrame, mask: np.ndarray) -> MaskFrame:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    bbox = (
        None
        if not len(xs)
        else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    return MaskFrame(
        frame.ref,
        binary,
        soft_alpha_from_mask(binary, 2),
        ("test",) if bbox else (),
        1.0,
        float(np.mean(binary > 0)),
        bbox,
    )


def synthetic_frames(count: int = 12, width: int = 160, height: int = 90, fps: int = 12):
    clean: list[np.ndarray] = []
    burned: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in range(count):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(35 + x * 90 + index * 2, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(45 + y * 100, 0, 255).astype(np.uint8)
        image[..., 2] = np.clip(70 + (x + y) * 55, 0, 255).astype(np.uint8)
        cv2.circle(image, (25 + index * 4, 28), 10, (20, 180, 90), -1)
        clean.append(image.copy())
        overlay = image.copy()
        text = "TEST"
        origin = (45, 76)
        cv2.putText(
            overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            overlay, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[55:84, 39:112] = 255
        burned.append(overlay)
        masks.append(mask)
    return clean, burned, masks


def write_video(path: Path, frames: list[np.ndarray], fps: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w", format="matroska")
    stream = container.add_stream("ffv1", rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "bgr0"
    stream.time_base = Fraction(1, fps)
    for index, array in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="bgr24")
        frame.pts = index
        frame.time_base = Fraction(1, fps)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path
