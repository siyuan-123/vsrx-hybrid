from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from vsrx.domain.contracts import (
    Shot,
    SubtitleTrack,
    SubtitleTrackFeatures,
    VideoFrame,
)
from vsrx.domain.enums import TrackClassification
from vsrx.tracking.association import HungarianTrackBuilder, RawTrack
from vsrx.utils.config import Config
from vsrx.utils.geometry import bbox_center, union_bboxes


class SubtitleTrackClassifier:
    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _local_contrast(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        padding = max(4, int(round((y2 - y1) * 0.35)))
        ox1, oy1, ox2, oy2 = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(w, x2 + padding),
            min(h, y2 + padding),
        )
        patch = cv2.cvtColor(frame[oy1:oy2, ox1:ox2], cv2.COLOR_BGR2GRAY)
        inner = cv2.cvtColor(
            frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)], cv2.COLOR_BGR2GRAY
        )
        if patch.size == 0 or inner.size == 0:
            return 0.0
        return float(
            np.clip((inner.std() + abs(float(inner.mean()) - float(patch.mean()))) / 96.0, 0.0, 1.0)
        )

    def features(
        self, raw: RawTrack, frames: Sequence[VideoFrame], shot: Shot
    ) -> SubtitleTrackFeatures:
        frame_map = {frame.ref.frame_index: frame.image_bgr for frame in frames}
        detections = raw.detections
        boxes = [item.bbox_xyxy for item in detections]
        frame_height, frame_width = frames[0].image_bgr.shape[:2]
        centers = np.asarray([bbox_center(box) for box in boxes], dtype=np.float32)
        normalized_centers = centers / np.array([frame_width, frame_height], dtype=np.float32)
        stability = float(
            np.clip(1.0 - np.linalg.norm(normalized_centers.std(axis=0)) * 7.0, 0.0, 1.0)
        )
        detector_confidence = float(np.mean([item.confidence for item in detections]))
        first_pts = detections[0].frame.pts_us
        last_pts = detections[-1].frame.pts_us
        duration_seconds = max(0.0, (last_pts - first_pts) / 1_000_000)
        cadence = (
            float(np.exp(-(((duration_seconds - 2.7) / 3.5) ** 2)))
            if duration_seconds <= 12
            else 0.15
        )
        mean_center = normalized_centers.mean(axis=0)
        bottom = float(np.clip((mean_center[1] - 0.52) / 0.35, 0.0, 1.0))
        top = float(np.clip((0.25 - mean_center[1]) / 0.25, 0.0, 1.0))
        horizontal_center = float(np.clip(1.0 - abs(mean_center[0] - 0.5) * 2.3, 0.0, 1.0))
        layout = max(bottom * horizontal_center, top * horizontal_center * 0.72)
        contrasts = [
            self._local_contrast(frame_map[item.frame.frame_index], item.bbox_xyxy)
            for item in detections
            if item.frame.frame_index in frame_map
        ]
        local_contrast = float(np.mean(contrasts)) if contrasts else 0.0
        track_motion = (
            float(np.mean(np.linalg.norm(np.diff(normalized_centers, axis=0), axis=1)))
            if len(normalized_centers) > 1
            else 0.0
        )
        overlay_decoupling = float(np.clip(stability + (0.04 - track_motion) * 4.0, 0.0, 1.0))
        shot_duration = max(1, shot.end_frame_index - shot.start_frame_index)
        unique_frames = len({item.frame.frame_index for item in detections})
        persistence = unique_frames / shot_duration
        x, y = mean_center
        in_corner = (x < 0.2 or x > 0.8) and (y < 0.25 or y > 0.75)
        tiny_area = np.mean([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]) / (
            frame_width * frame_height
        )
        tiny_corner = float(in_corner and tiny_area < 0.012)
        logo_persistence = float(
            np.clip((persistence - 0.55) / 0.45, 0.0, 1.0) * (0.5 + 0.5 * tiny_corner)
        )
        scene_motion_coupling = float(np.clip(track_motion * 18.0, 0.0, 1.0) * (1.0 - stability))
        unchanged = float(np.clip((duration_seconds - 8.0) / 15.0, 0.0, 1.0) * stability)
        return SubtitleTrackFeatures(
            detector_confidence=detector_confidence,
            screen_coordinate_stability=stability,
            subtitle_cadence=cadence,
            layout_prior=layout,
            local_contrast=local_contrast,
            overlay_motion_decoupling=overlay_decoupling,
            optional_audio_vad_alignment=0.0,
            logo_persistence=logo_persistence,
            scene_motion_coupling=scene_motion_coupling,
            tiny_corner_mark=tiny_corner,
            long_unchanged_content=unchanged,
        )

    def classify(self, raw: RawTrack, frames: Sequence[VideoFrame], shot: Shot) -> SubtitleTrack:
        features = self.features(raw, frames, shot)
        weights = self.config.get("subtitle_track_classifier.weights", {})
        penalties = self.config.get("subtitle_track_classifier.penalties", {})
        score = (
            float(weights.get("detector_confidence", 0.18)) * features.detector_confidence
            + float(weights.get("screen_coordinate_stability", 0.18))
            * features.screen_coordinate_stability
            + float(weights.get("subtitle_cadence", 0.16)) * features.subtitle_cadence
            + float(weights.get("layout_prior", 0.12)) * features.layout_prior
            + float(weights.get("local_contrast", 0.10)) * features.local_contrast
            + float(weights.get("overlay_motion_decoupling", 0.16))
            * features.overlay_motion_decoupling
            + float(weights.get("optional_audio_vad_alignment", 0.10))
            * features.optional_audio_vad_alignment
            - float(penalties.get("logo_persistence", 0.22)) * features.logo_persistence
            - float(penalties.get("scene_motion_coupling", 0.28)) * features.scene_motion_coupling
            - float(penalties.get("tiny_corner_mark", 0.12)) * features.tiny_corner_mark
            - float(penalties.get("long_unchanged_content", 0.15)) * features.long_unchanged_content
        )
        score = float(np.clip(score, 0.0, 1.0))
        auto = float(self.config.get("subtitle_track_classifier.auto_remove_threshold", 0.62))
        review = float(self.config.get("subtitle_track_classifier.review_threshold", 0.45))
        if score >= auto:
            classification = TrackClassification.SUBTITLE
        elif score >= review:
            classification = TrackClassification.UNCERTAIN
        elif features.logo_persistence > 0.45 or features.tiny_corner_mark > 0.5:
            classification = TrackClassification.LOGO
        else:
            classification = TrackClassification.SCENE_TEXT
        bbox = union_bboxes(item.bbox_xyxy for item in raw.detections)
        assert bbox is not None
        widths = np.array([box[2] - box[0] for box in (item.bbox_xyxy for item in raw.detections)])
        heights = np.array([box[3] - box[1] for box in (item.bbox_xyxy for item in raw.detections)])
        is_vertical = bool(np.median(heights) > np.median(widths) * 1.25)
        movement = float(
            np.linalg.norm(
                np.asarray(bbox_center(raw.detections[-1].bbox_xyxy))
                - np.asarray(bbox_center(raw.detections[0].bbox_xyxy))
            )
        )
        frame_width = frames[0].image_bgr.shape[1]
        is_moving = movement > max(8.0, frame_width * 0.015)
        # Karaoke is conservatively approximated from high persistence plus changing boxes.
        area_values = np.array(
            [
                (item.bbox_xyxy[2] - item.bbox_xyxy[0]) * (item.bbox_xyxy[3] - item.bbox_xyxy[1])
                for item in raw.detections
            ],
            dtype=np.float32,
        )
        is_karaoke = bool(
            len(area_values) > 4
            and area_values.std() / max(area_values.mean(), 1.0) > 0.18
            and features.layout_prior > 0.45
        )
        return SubtitleTrack(
            track_id=raw.track_id,
            shot_id=raw.shot_id,
            detections=raw.detections,
            score=score,
            classification=classification,
            features=features,
            roi_xyxy=bbox,
            is_vertical=is_vertical,
            is_karaoke=is_karaoke,
            is_moving=is_moving,
        )


class TrackPipeline:
    def __init__(self, config: Config) -> None:
        self.builder = HungarianTrackBuilder(config)
        self.classifier = SubtitleTrackClassifier(config)

    def build(self, frames: list[VideoFrame], detections, shot: Shot) -> list[SubtitleTrack]:
        raw_tracks = self.builder.build_raw(frames, list(detections), shot)
        return [self.classifier.classify(raw, frames, shot) for raw in raw_tracks]
