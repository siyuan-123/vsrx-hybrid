from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from vsrx.domain.contracts import Polygon, Shot, TextDetection, VideoFrame
from vsrx.tracking.kalman import BBoxKalman
from vsrx.utils.config import Config
from vsrx.utils.geometry import bbox_center, bbox_iou, clamp_bbox


@dataclass(slots=True)
class _ActiveTrack:
    track_id: str
    detections: list[TextDetection]
    kalman: BBoxKalman
    appearance_lab: np.ndarray
    missed: int = 0
    last_frame_index: int = 0

    @property
    def last(self) -> TextDetection:
        return self.detections[-1]


@dataclass(frozen=True, slots=True)
class RawTrack:
    track_id: str
    shot_id: int
    detections: tuple[TextDetection, ...]


class HungarianTrackBuilder:
    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _appearance(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros(3, dtype=np.float32)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def _cost(
        self,
        track: _ActiveTrack,
        detection: TextDetection,
        appearance: np.ndarray,
        width: int,
        height: int,
    ) -> float:
        previous = track.kalman.bbox
        current = detection.bbox_xyxy
        iou_error = 1.0 - bbox_iou(previous, current)
        pcx, pcy = bbox_center(previous)
        ccx, ccy = bbox_center(current)
        center_error = np.hypot((pcx - ccx) / max(width, 1), (pcy - ccy) / max(height, 1))
        ph = max(1, previous[3] - previous[1])
        ch = max(1, current[3] - current[1])
        height_error = abs(np.log(ch / ph))
        angle_error = min(abs(track.last.angle_degrees - detection.angle_degrees), 180.0) / 180.0
        appearance_error = float(np.linalg.norm(track.appearance_lab - appearance) / 255.0)
        return (
            float(self.config.get("tracking.iou_weight", 0.35)) * iou_error
            + float(self.config.get("tracking.center_distance_weight", 0.20)) * center_error
            + float(self.config.get("tracking.height_ratio_weight", 0.15)) * height_error
            + float(self.config.get("tracking.angle_weight", 0.10)) * angle_error
            + float(self.config.get("tracking.appearance_weight", 0.20)) * appearance_error
        )

    def _interpolate(
        self, detections: list[TextDetection], frames_by_index: dict[int, VideoFrame]
    ) -> list[TextDetection]:
        if len(detections) < 2:
            return detections
        output: list[TextDetection] = []
        for left, right in zip(detections, detections[1:], strict=False):
            output.append(left)
            gap = right.frame.frame_index - left.frame.frame_index
            if gap <= 1 or gap > int(self.config.get("tracking.max_gap_frames", 8)) + 1:
                continue
            left_points = np.asarray(left.polygon.points, dtype=np.float32)
            right_points = np.asarray(right.polygon.points, dtype=np.float32)
            if left_points.shape != right_points.shape:
                continue
            for step in range(1, gap):
                frame_index = left.frame.frame_index + step
                frame = frames_by_index.get(frame_index)
                if frame is None:
                    continue
                ratio = step / gap
                points = left_points * (1.0 - ratio) + right_points * ratio
                output.append(
                    TextDetection(
                        frame=frame.ref,
                        polygon=Polygon(tuple((float(x), float(y)) for x, y in points)),
                        confidence=min(left.confidence, right.confidence) * 0.72,
                        angle_degrees=left.angle_degrees * (1.0 - ratio)
                        + right.angle_degrees * ratio,
                        detector_tier=left.detector_tier,
                        source_roi=left.source_roi,
                        propagated=True,
                    )
                )
        output.append(detections[-1])
        return sorted(output, key=lambda item: item.frame.frame_index)

    def build_raw(
        self, frames: list[VideoFrame], detections: list[TextDetection], shot: Shot
    ) -> list[RawTrack]:
        frames_by_index = {frame.ref.frame_index: frame for frame in frames}
        detections_by_frame: dict[int, list[TextDetection]] = {}
        for detection in detections:
            if detection.frame.shot_id == shot.shot_id:
                detections_by_frame.setdefault(detection.frame.frame_index, []).append(detection)
        active: list[_ActiveTrack] = []
        completed: list[_ActiveTrack] = []
        max_gap = int(self.config.get("tracking.max_gap_frames", 8))
        max_center = float(self.config.get("tracking.max_center_distance_ratio", 0.12))
        max_height = float(self.config.get("tracking.max_height_change_ratio", 0.45))

        for frame in frames:
            frame_index = frame.ref.frame_index
            current = detections_by_frame.get(frame_index, [])
            appearances = [
                self._appearance(frame.image_bgr, detection.bbox_xyxy) for detection in current
            ]
            for track in active:
                steps = max(1, frame_index - track.last_frame_index)
                track.kalman.predict(steps)
            if active and current:
                costs = np.full((len(active), len(current)), 1e6, dtype=np.float64)
                for row, track in enumerate(active):
                    predicted = track.kalman.bbox
                    pcx, pcy = bbox_center(predicted)
                    ph = max(1, predicted[3] - predicted[1])
                    for column, detection in enumerate(current):
                        bbox = detection.bbox_xyxy
                        ccx, ccy = bbox_center(bbox)
                        center = np.hypot(
                            (pcx - ccx) / frame.image_bgr.shape[1],
                            (pcy - ccy) / frame.image_bgr.shape[0],
                        )
                        height_change = abs((bbox[3] - bbox[1]) / ph - 1.0)
                        if center > max_center or height_change > max_height:
                            continue
                        costs[row, column] = self._cost(
                            track,
                            detection,
                            appearances[column],
                            frame.image_bgr.shape[1],
                            frame.image_bgr.shape[0],
                        )
                rows, columns = linear_sum_assignment(costs)
                matched_tracks: set[int] = set()
                matched_detections: set[int] = set()
                for row, column in zip(rows, columns, strict=False):
                    if costs[row, column] >= 0.78:
                        continue
                    track = active[row]
                    detection = current[column]
                    track.kalman.update(detection.bbox_xyxy)
                    track.detections.append(detection)
                    track.appearance_lab = 0.75 * track.appearance_lab + 0.25 * appearances[column]
                    track.missed = 0
                    track.last_frame_index = frame_index
                    matched_tracks.add(row)
                    matched_detections.add(column)
                for index, track in enumerate(active):
                    if index not in matched_tracks:
                        track.missed += 1
                for index, detection in enumerate(current):
                    if index not in matched_detections:
                        active.append(
                            _ActiveTrack(
                                track_id=f"trk_{uuid4().hex[:12]}",
                                detections=[detection],
                                kalman=BBoxKalman(detection.bbox_xyxy),
                                appearance_lab=appearances[index],
                                last_frame_index=frame_index,
                            )
                        )
            else:
                for track in active:
                    track.missed += 1
                for index, detection in enumerate(current):
                    active.append(
                        _ActiveTrack(
                            track_id=f"trk_{uuid4().hex[:12]}",
                            detections=[detection],
                            kalman=BBoxKalman(detection.bbox_xyxy),
                            appearance_lab=appearances[index],
                            last_frame_index=frame_index,
                        )
                    )
            survivors: list[_ActiveTrack] = []
            for track in active:
                if track.missed > max_gap:
                    completed.append(track)
                else:
                    survivors.append(track)
            active = survivors
        completed.extend(active)
        minimum = int(self.config.get("tracking.min_track_frames", 3))
        raw: list[RawTrack] = []
        for track in completed:
            unique_frames = len({item.frame.frame_index for item in track.detections})
            if unique_frames < minimum:
                continue
            interpolated = self._interpolate(
                sorted(track.detections, key=lambda item: item.frame.frame_index), frames_by_index
            )
            raw.append(RawTrack(track.track_id, shot.shot_id, tuple(interpolated)))
        return raw
