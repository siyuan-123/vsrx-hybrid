from __future__ import annotations

import logging

import cv2
import numpy as np

from vsrx.domain.contracts import ProbeResult, Shot
from vsrx.utils.config import Config

logger = logging.getLogger(__name__)


class AdaptiveSceneDetector:
    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _histogram_distance(before: np.ndarray, after: np.ndarray) -> float:
        before_hsv = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
        after_hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([before_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist_b = cv2.calcHist([after_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))

    def _confirm_cuts(self, probe: ProbeResult, shots: list[Shot]) -> list[Shot]:
        if len(shots) <= 1 or not self.config.get("scene_detection.confirm_with_histogram", True):
            return shots
        threshold = float(self.config.get("scene_detection.histogram_threshold", 0.42))
        capture = cv2.VideoCapture(str(probe.input_path))
        if not capture.isOpened():
            return shots
        accepted_boundaries: list[int] = [shots[0].start_frame_index]
        confidences: dict[int, float] = {}
        try:
            for shot in shots[:-1]:
                boundary = shot.end_frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, boundary - 1))
                ok_a, before = capture.read()
                ok_b, after = capture.read()
                if not (ok_a and ok_b):
                    accepted_boundaries.append(boundary)
                    confidences[boundary] = shot.cut_confidence
                    continue
                distance = self._histogram_distance(before, after)
                if distance >= threshold:
                    accepted_boundaries.append(boundary)
                    confidences[boundary] = min(1.0, distance / max(threshold, 1e-6))
            accepted_boundaries.append(shots[-1].end_frame_index)
        finally:
            capture.release()

        merged: list[Shot] = []
        fps = max(probe.fps, 1e-6)
        for index in range(len(accepted_boundaries) - 1):
            start_frame = accepted_boundaries[index]
            end_frame = accepted_boundaries[index + 1]
            merged.append(
                Shot(
                    shot_id=index,
                    start_pts_us=int(round(start_frame / fps * 1_000_000)),
                    end_pts_us=min(probe.duration_us, int(round(end_frame / fps * 1_000_000)))
                    if probe.duration_us
                    else int(round(end_frame / fps * 1_000_000)),
                    start_frame_index=start_frame,
                    end_frame_index=end_frame,
                    cut_confidence=confidences.get(end_frame, 1.0),
                    transition="cut",
                )
            )
        return merged or shots

    def detect(self, probe: ProbeResult) -> list[Shot]:
        if not self.config.get("scene_detection.enabled", True):
            estimated_frames = max(1, int(round(probe.duration_us / 1_000_000 * probe.fps)))
            return [Shot(0, 0, probe.duration_us, 0, estimated_frames, 1.0, "unknown")]
        try:
            from scenedetect import AdaptiveDetector, detect

            detector = AdaptiveDetector(
                adaptive_threshold=float(
                    self.config.get("scene_detection.adaptive_threshold", 3.0)
                ),
                min_scene_len=int(self.config.get("scene_detection.min_scene_len_frames", 10)),
                window_width=int(self.config.get("scene_detection.rolling_window_frames", 2)),
                min_content_val=float(self.config.get("scene_detection.min_content_val", 15.0)),
            )
            scenes = detect(str(probe.input_path), detector, show_progress=False)
            shots = [
                Shot(
                    shot_id=index,
                    start_pts_us=int(round(start.get_seconds() * 1_000_000)),
                    end_pts_us=int(round(end.get_seconds() * 1_000_000)),
                    start_frame_index=start.get_frames(),
                    end_frame_index=end.get_frames(),
                    cut_confidence=1.0,
                    transition="cut",
                )
                for index, (start, end) in enumerate(scenes)
            ]
        except Exception as exc:
            logger.warning("PySceneDetect failed; using a single shot: %s", exc)
            estimated_frames = max(1, int(round(probe.duration_us / 1_000_000 * probe.fps)))
            shots = [Shot(0, 0, probe.duration_us, 0, estimated_frames, 0.0, "unknown")]
        if not shots:
            estimated_frames = max(1, int(round(probe.duration_us / 1_000_000 * probe.fps)))
            shots = [Shot(0, 0, probe.duration_us, 0, estimated_frames, 1.0, "unknown")]
        return self._confirm_cuts(probe, shots)
