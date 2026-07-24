from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction

import av
import cv2
import numpy as np

from vsrx.domain.contracts import FrameRef, ProbeResult, VideoFrame
from vsrx.domain.errors import DecodeError

logger = logging.getLogger(__name__)


def _pts_to_us(pts: int | None, time_base: Fraction | None, fallback_us: int) -> int:
    if pts is None or time_base is None:
        return fallback_us
    return int(round(float(pts * time_base) * 1_000_000))


def _rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


class PyAVFrameReader:
    """PTS-aware frame reader with time-range seeking.

    A new container is opened for each iterator so callers can process independent
    chunks concurrently without sharing decoder state.
    """

    def __init__(self, probe: ProbeResult) -> None:
        self.probe = probe

    def iter_frames(
        self,
        *,
        start_pts_us: int | None = None,
        end_pts_us: int | None = None,
        shot_id: int = 0,
        start_frame_index: int = 0,
    ) -> Iterator[VideoFrame]:
        path = str(self.probe.input_path)
        try:
            container = av.open(path)
        except av.AVError as exc:  # type: ignore[attr-defined]
            raise DecodeError(f"cannot open video: {path}") from exc
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise DecodeError(f"no video stream: {path}")
            stream.thread_type = "AUTO"
            time_base = (
                Fraction(stream.time_base)
                if stream.time_base
                else self.probe.video_time_base.as_fraction()
            )
            if start_pts_us is not None and start_pts_us > 0:
                seek_pts = int((start_pts_us / 1_000_000) / float(time_base))
                try:
                    container.seek(max(0, seek_pts), stream=stream, backward=True, any_frame=False)
                except (av.AVError, ValueError):  # type: ignore[attr-defined]
                    logger.warning(
                        "seek failed; decoding from beginning", extra={"stage": "decode"}
                    )
            frame_index = start_frame_index
            fallback_step = int(round(1_000_000 / max(self.probe.fps, 1e-6)))
            fallback_us = start_pts_us or 0
            last_pts_us: int | None = None
            pending: VideoFrame | None = None
            for packet in container.demux(stream):
                for frame in packet.decode():
                    pts_us = _pts_to_us(
                        frame.pts,
                        Fraction(frame.time_base) if frame.time_base else time_base,
                        fallback_us,
                    )
                    fallback_us = pts_us + fallback_step
                    if start_pts_us is not None and pts_us < start_pts_us:
                        continue
                    if end_pts_us is not None and pts_us >= end_pts_us:
                        if pending is not None:
                            if last_pts_us is not None:
                                pending.duration_us = max(1, pts_us - last_pts_us)
                            yield pending
                        return
                    image = frame.to_ndarray(format="bgr24")
                    image = _rotate(image, self.probe.rotation_degrees)
                    ref = FrameRef(frame_index=frame_index, pts_us=pts_us, shot_id=shot_id)
                    current = VideoFrame(
                        ref=ref,
                        image_bgr=np.ascontiguousarray(image),
                        source_pts=frame.pts,
                        source_time_base=Fraction(frame.time_base)
                        if frame.time_base
                        else time_base,
                    )
                    if pending is not None:
                        pending.duration_us = max(
                            1, pts_us - (last_pts_us or pts_us - fallback_step)
                        )
                        yield pending
                    pending = current
                    last_pts_us = pts_us
                    frame_index += 1
            if pending is not None:
                pending.duration_us = pending.duration_us or fallback_step
                yield pending
        except DecodeError:
            raise
        except Exception as exc:
            raise DecodeError(f"decode failed for {path}: {exc}") from exc
        finally:
            container.close()

    def read_range(
        self,
        start_pts_us: int,
        end_pts_us: int,
        *,
        shot_id: int = 0,
        start_frame_index: int = 0,
        max_frames: int | None = None,
    ) -> list[VideoFrame]:
        result: list[VideoFrame] = []
        for frame in self.iter_frames(
            start_pts_us=start_pts_us,
            end_pts_us=end_pts_us,
            shot_id=shot_id,
            start_frame_index=start_frame_index,
        ):
            result.append(frame)
            if max_frames is not None and len(result) >= max_frames:
                break
        return result

    def sample_at_pts(self, pts_values_us: Sequence[int], *, shot_id: int = 0) -> list[VideoFrame]:
        targets = sorted(set(max(0, int(value)) for value in pts_values_us))
        if not targets:
            return []
        result: list[VideoFrame] = []
        target_index = 0
        previous: VideoFrame | None = None
        for frame in self.iter_frames(start_pts_us=max(0, targets[0] - 2_000_000), shot_id=shot_id):
            target = targets[target_index]
            if frame.ref.pts_us >= target:
                if previous is None or abs(frame.ref.pts_us - target) <= abs(
                    previous.ref.pts_us - target
                ):
                    selected = frame
                else:
                    selected = previous
                selected = VideoFrame(
                    ref=FrameRef(len(result), selected.ref.pts_us, shot_id),
                    image_bgr=selected.image_bgr.copy(),
                    source_pts=selected.source_pts,
                    source_time_base=selected.source_time_base,
                    duration_us=selected.duration_us,
                )
                result.append(selected)
                target_index += 1
                if target_index >= len(targets):
                    break
            previous = frame
        return result


def iter_images(frames: Iterable[VideoFrame]) -> Iterator[np.ndarray]:
    for frame in frames:
        yield frame.image_bgr
