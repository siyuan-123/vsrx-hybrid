#!/usr/bin/env python3
"""Export deterministic rectangular per-frame masks for one or more ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        parts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2") from exc
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise argparse.ArgumentTypeError("ROI must be valid x1,y1,x2,y2")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--roi", action="append", type=parse_roi, required=True)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, int | str]] = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for index, frame in enumerate(container.decode(stream)):
            height, width = frame.height, frame.width
            mask = np.zeros((height, width), dtype=np.uint8)
            for x1, y1, x2, y2 in args.roi:
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(width, x2), min(height, y2)
                if cx2 > cx1 and cy2 > cy1:
                    mask[cy1:cy2, cx1:cx2] = 255
            name = f"{index:08d}.png"
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                raise RuntimeError(f"failed to encode {name}")
            (output / name).write_bytes(encoded.tobytes())
            pts_us = (
                round(float(frame.pts * time_base) * 1_000_000)
                if frame.pts is not None and time_base is not None
                else 0
            )
            manifest.append({"frame_index": index, "pts_us": pts_us, "path": name})
    (output / "masks.json").write_text(
        json.dumps(
            {"schema_version": 1, "time_base": "microseconds", "frames": manifest},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} masks to {output}")


if __name__ == "__main__":
    main()
