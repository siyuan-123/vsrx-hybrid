from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vsrx.domain.contracts import MaskFrame, Segment, Shot, VideoFrame
from vsrx.utils.config import Config


class TemporalSegmenter:
    """Split a shot on subtitle activity boundaries and bounded chunk sizes."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def split(
        self, shot: Shot, frames: Sequence[VideoFrame], masks: Sequence[MaskFrame]
    ) -> list[Segment]:
        if len(frames) != len(masks):
            raise ValueError("frames and masks must have identical lengths")
        if not frames:
            return []
        minimum = int(self.config.get("routing.segment_min_frames", 8))
        maximum = int(self.config.get("routing.segment_max_frames", 96))
        active = [bool(np.any(item.hard_mask)) for item in masks]
        boundaries = [0]
        for index in range(1, len(frames)):
            if (
                active[index] != active[index - 1]
                and index - boundaries[-1] >= minimum
                or index - boundaries[-1] >= maximum
            ):
                boundaries.append(index)
        if boundaries[-1] != len(frames):
            boundaries.append(len(frames))

        # Merge a tiny final segment backward when this does not violate maximum.
        if (
            len(boundaries) >= 3
            and boundaries[-1] - boundaries[-2] < minimum
            and boundaries[-1] - boundaries[-3] <= maximum
        ):
            del boundaries[-2]

        segments: list[Segment] = []
        for number, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=True)):
            first, last = frames[start], frames[end - 1]
            segments.append(
                Segment(
                    segment_id=f"shot-{shot.shot_id:06d}-segment-{number:04d}-{first.ref.pts_us}",
                    shot_id=shot.shot_id,
                    start_pts_us=first.ref.pts_us,
                    end_pts_us=last.ref.pts_us + (last.duration_us or 1),
                    start_frame_index=first.ref.frame_index,
                    end_frame_index=last.ref.frame_index + 1,
                    core_start_index=start,
                    core_end_index=end,
                )
            )
        return segments
