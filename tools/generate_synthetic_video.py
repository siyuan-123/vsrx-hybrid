#!/usr/bin/env python3
"""Generate a deterministic clean video, burned-subtitle video and frame masks."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("ffv1", rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "bgr0"
    stream.time_base = Fraction(1, fps)
    for index, image in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = index
        frame.time_base = Fraction(1, fps)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _encode_png(path: Path, mask: np.ndarray) -> None:
    ok, payload = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    path.write_bytes(payload.tobytes())


def generate(output_dir: Path, width: int, height: int, frames: int, fps: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    clean_frames: list[np.ndarray] = []
    burned_frames: list[np.ndarray] = []
    manifest_frames: list[dict[str, int | str]] = []

    for index in range(frames):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(45 + 95 * x + 20 * np.sin(index / 8), 0, 255)
        image[..., 1] = np.clip(35 + 100 * y + 25 * np.cos(index / 10), 0, 255)
        image[..., 2] = np.clip(70 + 60 * (1 - x) + 25 * y, 0, 255)

        shift = int(round(12 * np.sin(index / 9)))
        cv2.rectangle(image, (30 + shift, 28), (105 + shift, 95), (40, 180, 220), -1)
        cv2.circle(image, (width - 70 - shift, 68), 28, (210, 85, 55), -1)
        # Foreground crosses the subtitle band in part of the clip.
        person_x = int(width * (index / max(frames - 1, 1)))
        cv2.rectangle(
            image, (person_x - 11, height - 73), (person_x + 11, height - 25), (90, 210, 80), -1
        )
        clean = image.copy()
        burned = image.copy()
        mask = np.zeros((height, width), dtype=np.uint8)

        text = "VSR-X TEST SUBTITLE" if index < frames // 2 else "HYBRID CLEAN PLATE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, width / 640.0)
        thickness = max(1, int(round(width / 320)))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        origin = ((width - text_w) // 2, height - max(14, height // 12))
        cv2.putText(burned, text, origin, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(burned, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(mask, text, origin, font, scale, 255, thickness + 7, cv2.LINE_AA)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        clean_frames.append(clean)
        burned_frames.append(burned)
        mask_name = f"{index:08d}.png"
        _encode_png(masks_dir / mask_name, mask)
        manifest_frames.append(
            {"frame_index": index, "pts_us": round(index * 1_000_000 / fps), "path": mask_name}
        )

    clean_path = output_dir / "clean.mkv"
    burned_path = output_dir / "burned.mkv"
    _write_video(clean_path, clean_frames, fps)
    _write_video(burned_path, burned_frames, fps)
    (masks_dir / "masks.json").write_text(
        json.dumps(
            {"schema_version": 1, "time_base": "microseconds", "frames": manifest_frames},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = {
        "clean": str(clean_path),
        "burned": str(burned_path),
        "masks": str(masks_dir),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {key: str(value) for key, value in metadata.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.fps) <= 0:
        parser.error("all numeric values must be positive")
    print(
        json.dumps(
            generate(args.output_dir.resolve(), args.width, args.height, args.frames, args.fps),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
