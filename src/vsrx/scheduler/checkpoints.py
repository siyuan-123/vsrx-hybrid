from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from vsrx.domain.contracts import FrameRef, ProbeResult, VideoFrame
from vsrx.media.encode import FFV1CheckpointWriter, write_json_atomic
from vsrx.utils.hash import sha256_file


class SegmentCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, segment_id: str) -> tuple[Path, Path]:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in segment_id
        )
        return self.root / f"{safe}.mkv", self.root / f"{safe}.json"

    def write(
        self,
        segment_id: str,
        frames: Sequence[VideoFrame],
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not frames:
            raise ValueError("cannot checkpoint an empty segment")
        video_path, metadata_path = self.paths(segment_id)
        temporary_video = video_path.with_suffix(".mkv.tmp")
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        container = av.open(str(temporary_video), mode="w", format="matroska")
        fps = 25
        if len(frames) > 1:
            median_duration = int(
                np.median([max(1, frame.duration_us or 40_000) for frame in frames])
            )
            fps = max(1, round(1_000_000 / median_duration))
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = frames[0].image_bgr.shape[1]
        stream.height = frames[0].image_bgr.shape[0]
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 1_000_000)
        for index, item in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(item.image_bgr), format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        metadata = {
            "segment_id": segment_id,
            "frames": [
                {
                    "frame_index": item.ref.frame_index,
                    "pts_us": item.ref.pts_us,
                    "shot_id": item.ref.shot_id,
                    "duration_us": item.duration_us,
                    "source_pts": item.source_pts,
                    "source_time_base": (
                        [item.source_time_base.numerator, item.source_time_base.denominator]
                        if item.source_time_base
                        else None
                    ),
                }
                for item in frames
            ],
        }
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_video, video_path)
        os.replace(temporary_metadata, metadata_path)
        return video_path

    def valid(self, segment_id: str) -> bool:
        video, metadata = self.paths(segment_id)
        if not video.is_file() or not metadata.is_file() or video.stat().st_size == 0:
            return False
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            return bool(payload.get("frames"))
        except Exception:
            return False

    def metadata(self, segment_id: str) -> dict[str, Any]:
        _, metadata_path = self.paths(segment_id)
        return dict(json.loads(metadata_path.read_text(encoding="utf-8")))

    def read(self, segment_id: str) -> list[VideoFrame]:
        video, metadata = self.paths(segment_id)
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        records = payload["frames"]
        container = av.open(str(video))
        decoded = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
        container.close()
        if len(decoded) != len(records):
            raise RuntimeError(
                f"checkpoint frame count mismatch for {segment_id}: {len(decoded)} != {len(records)}"
            )
        result: list[VideoFrame] = []
        for array, record in zip(decoded, records, strict=True):
            tb = record.get("source_time_base")
            result.append(
                VideoFrame(
                    ref=FrameRef(
                        int(record["frame_index"]), int(record["pts_us"]), int(record["shot_id"])
                    ),
                    image_bgr=np.ascontiguousarray(array),
                    source_pts=record.get("source_pts"),
                    source_time_base=Fraction(*tb) if tb else None,
                    duration_us=record.get("duration_us"),
                )
            )
        return result

    def digest(self, segment_id: str) -> str:
        video, metadata = self.paths(segment_id)
        return sha256_file(video) + ":" + sha256_file(metadata)

    def assemble(self, segment_ids: Sequence[str], output_path: Path, probe: ProbeResult) -> Path:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with FFV1CheckpointWriter(temporary, probe) as writer:
            last_pts = -1
            for segment_id in segment_ids:
                for frame in self.read(segment_id):
                    if frame.ref.pts_us <= last_pts:
                        continue
                    writer.write(frame)
                    last_pts = frame.ref.pts_us
        os.replace(temporary, output_path)
        manifest = output_path.with_suffix(output_path.suffix + ".json")
        write_json_atomic(
            manifest, {"segments": list(segment_ids), "sha256": sha256_file(output_path)}
        )
        return output_path
