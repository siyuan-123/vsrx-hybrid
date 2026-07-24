from __future__ import annotations

from vsrx.domain.contracts import ProbeResult, Shot
from vsrx.utils.config import Config


class DiscoverySampler:
    def __init__(self, config: Config) -> None:
        self.config = config

    def sample_pts(self, probe: ProbeResult, shots: list[Shot]) -> list[int]:
        duration = probe.duration_us
        interval_us = int(
            float(self.config.get("subtitle_discovery.full_frame_sample_interval_seconds", 0.75))
            * 1_000_000
        )
        first_us = int(
            float(self.config.get("subtitle_discovery.sample_first_seconds", 20)) * 1_000_000
        )
        last_us = int(
            float(self.config.get("subtitle_discovery.sample_last_seconds", 20)) * 1_000_000
        )
        interval_us = max(interval_us, 1)
        points: set[int] = set()

        # For short videos the first-window scan already covers the whole file.
        # Re-adding middle/last/periodic windows only multiplies OCR calls.
        fully_covered = duration <= first_us + last_us
        for value in range(0, min(duration, first_us) + 1, interval_us):
            points.add(value)
        if not fully_covered and duration > last_us:
            start = max(0, duration - last_us)
            for value in range(start, duration + 1, interval_us):
                points.add(value)
        if not fully_covered:
            middle_windows = int(self.config.get("subtitle_discovery.sample_middle_windows", 3))
            for index in range(1, middle_windows + 1):
                center = int(duration * index / (middle_windows + 1))
                for offset in (-interval_us, 0, interval_us):
                    points.add(max(0, min(duration, center + offset)))
        for shot in shots:
            points.add(shot.start_pts_us)
            points.add((shot.start_pts_us + shot.end_pts_us) // 2)
        if not fully_covered:
            rescan = int(
                float(self.config.get("subtitle_discovery.periodic_full_frame_rescan_seconds", 8.0))
                * 1_000_000
            )
            for value in range(0, duration + 1, max(rescan, 1)):
                points.add(value)
        return sorted(point for point in points if 0 <= point < max(duration, 1))
