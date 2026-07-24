from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from vsrx.domain.contracts import TextDetection
from vsrx.utils.config import Config
from vsrx.utils.geometry import bbox_center, bbox_iou, clamp_bbox, expand_bbox, union_bboxes


@dataclass(slots=True)
class _Cluster:
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    frame_ids: set[int] = field(default_factory=set)
    confidences: list[float] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        value = union_bboxes(self.boxes)
        assert value is not None
        return value


class ROIDiscoverer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def discover(
        self, detections: Sequence[TextDetection], width: int, height: int
    ) -> list[tuple[int, int, int, int]]:
        if not detections:
            bottom = float(self.config.get("subtitle_discovery.default_bottom_band_hint", 0.38))
            return [(0, int(height * (1.0 - bottom)), width, height)]
        clusters: list[_Cluster] = []
        for detection in sorted(detections, key=lambda item: item.frame.pts_us):
            box = detection.bbox_xyxy
            cx, cy = bbox_center(box)
            bh = max(1, box[3] - box[1])
            best: _Cluster | None = None
            best_score = -1.0
            for cluster in clusters:
                cluster_box = cluster.bbox
                ccx, ccy = bbox_center(cluster_box)
                center_distance = ((cx - ccx) / width) ** 2 + ((cy - ccy) / height) ** 2
                vertical_overlap = max(
                    0, min(box[3], cluster_box[3]) - max(box[1], cluster_box[1])
                ) / max(1, min(bh, cluster_box[3] - cluster_box[1]))
                score = max(
                    bbox_iou(box, cluster_box),
                    vertical_overlap * max(0.0, 1.0 - center_distance * 25.0),
                )
                if score > best_score and score >= 0.25:
                    best = cluster
                    best_score = score
            if best is None:
                best = _Cluster()
                clusters.append(best)
            best.boxes.append(box)
            best.frame_ids.add(detection.frame.frame_index)
            best.confidences.append(detection.confidence)

        total_frames = max(1, len({item.frame.frame_index for item in detections}))
        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        padding = int(
            round(
                min(width, height)
                * float(self.config.get("subtitle_discovery.roi_padding_ratio", 0.025))
            )
        )
        for cluster in clusters:
            box = expand_bbox(cluster.bbox, padding, width, height)
            x1, y1, x2, y2 = box
            persistence = len(cluster.frame_ids) / total_frames
            confidence = sum(cluster.confidences) / max(1, len(cluster.confidences))
            center_x, center_y = bbox_center(box)
            bottom_prior = max(0.0, (center_y / height - 0.45) / 0.55)
            center_prior = max(0.0, 1.0 - abs(center_x / width - 0.5) * 1.5)
            size_penalty = min(1.0, ((x2 - x1) * (y2 - y1)) / (width * height * 0.25))
            score = (
                0.42 * persistence
                + 0.28 * confidence
                + 0.18 * bottom_prior
                + 0.12 * center_prior
                - 0.12 * size_penalty
            )
            scored.append((score, box))
        scored.sort(reverse=True, key=lambda item: item[0])
        selected: list[tuple[int, int, int, int]] = []
        for _, box in scored:
            if any(bbox_iou(box, existing) > 0.55 for existing in selected):
                continue
            selected.append(clamp_bbox(box, width, height))
            if len(selected) >= int(self.config.get("subtitle_discovery.max_rois", 4)):
                break
        return selected or [(0, int(height * 0.62), width, height)]
