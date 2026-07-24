"""Stable data contracts shared by all VSR-X modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .enums import JobState, Route, TrackClassification


@dataclass(frozen=True, slots=True)
class Rational:
    num: int
    den: int

    def __post_init__(self) -> None:
        if self.den == 0:
            raise ValueError("denominator cannot be zero")

    @classmethod
    def parse(cls, value: str | None, default: Rational | None = None) -> Rational | None:
        if not value or value in {"0/0", "N/A"}:
            return default
        try:
            num, den = value.split("/", 1)
            return cls(int(num), int(den))
        except (ValueError, TypeError):
            return default

    def as_fraction(self) -> Fraction:
        return Fraction(self.num, self.den)

    def as_float(self) -> float:
        return self.num / self.den


@dataclass(frozen=True, slots=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None
    time_base: Rational | None
    tags: Mapping[str, str] = field(default_factory=dict)
    disposition: Mapping[str, int] = field(default_factory=dict)
    profile: str | None = None
    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    input_path: Path
    input_hash: str
    duration_us: int
    width: int
    height: int
    average_frame_rate: Rational | None
    nominal_frame_rate: Rational | None
    video_time_base: Rational
    is_vfr: bool
    is_hdr: bool
    is_interlaced: bool
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None
    pixel_format: str | None
    streams: tuple[StreamInfo, ...]
    chapters: tuple[Mapping[str, Any], ...] = ()
    rotation_degrees: int = 0
    start_time_us: int = 0
    raw_probe: Mapping[str, Any] = field(default_factory=dict)

    @property
    def video_stream(self) -> StreamInfo:
        for stream in self.streams:
            if stream.codec_type == "video":
                return stream
        raise LookupError("probe result contains no video stream")

    @property
    def subtitle_streams(self) -> tuple[StreamInfo, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == "subtitle")

    @property
    def has_soft_subtitles(self) -> bool:
        return bool(self.subtitle_streams)

    @property
    def fps(self) -> float:
        rate = self.average_frame_rate or self.nominal_frame_rate
        return rate.as_float() if rate else 25.0


@dataclass(frozen=True, slots=True)
class FrameRef:
    frame_index: int
    pts_us: int
    shot_id: int


@dataclass(slots=True)
class VideoFrame:
    ref: FrameRef
    image_bgr: np.ndarray
    source_pts: int | None
    source_time_base: Fraction | None
    duration_us: int | None = None


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def bbox(self) -> tuple[int, int, int, int]:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


@dataclass(frozen=True, slots=True)
class TextDetection:
    frame: FrameRef
    polygon: Polygon
    confidence: float
    angle_degrees: float
    probability_map_ref: str | None = None
    detector_tier: str = "small"
    source_roi: tuple[int, int, int, int] | None = None
    propagated: bool = False

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return self.polygon.bbox()


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: int
    start_pts_us: int
    end_pts_us: int
    start_frame_index: int
    end_frame_index: int
    cut_confidence: float
    transition: str = "cut"

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame_index - self.start_frame_index)


@dataclass(frozen=True, slots=True)
class SubtitleTrackFeatures:
    detector_confidence: float
    screen_coordinate_stability: float
    subtitle_cadence: float
    layout_prior: float
    local_contrast: float
    overlay_motion_decoupling: float
    optional_audio_vad_alignment: float
    logo_persistence: float
    scene_motion_coupling: float
    tiny_corner_mark: float
    long_unchanged_content: float


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    track_id: str
    shot_id: int
    detections: tuple[TextDetection, ...]
    score: float
    classification: TrackClassification
    features: SubtitleTrackFeatures
    roi_xyxy: tuple[int, int, int, int]
    is_vertical: bool = False
    is_karaoke: bool = False
    is_moving: bool = False


@dataclass(slots=True)
class MaskFrame:
    frame: FrameRef
    hard_mask: np.ndarray
    soft_alpha: np.ndarray
    source_track_ids: tuple[str, ...]
    confidence: float
    mask_ratio_of_frame: float
    expanded_bbox_xyxy: tuple[int, int, int, int] | None


@dataclass(slots=True)
class MotionField:
    source: FrameRef
    target: FrameRef
    flow_xy: np.ndarray
    confidence: np.ndarray
    occlusion: np.ndarray
    global_transform: np.ndarray | None


@dataclass(slots=True)
class CleanPlateResult:
    frame: FrameRef
    image_bgr: np.ndarray
    coverage: np.ndarray
    confidence: np.ndarray
    residual_mask: np.ndarray
    mean_coverage_in_mask: float
    mean_confidence_in_mask: float
    reference_frames: tuple[FrameRef, ...]


@dataclass(frozen=True, slots=True)
class Segment:
    segment_id: str
    shot_id: int
    start_pts_us: int
    end_pts_us: int
    start_frame_index: int
    end_frame_index: int
    core_start_index: int
    core_end_index: int


@dataclass(frozen=True, slots=True)
class SegmentFeatures:
    shot_id: int
    start_pts_us: int
    end_pts_us: int
    frame_count: int
    mask_ratio_of_frame: float
    mean_clean_plate_coverage: float
    mean_clean_plate_confidence: float
    mean_flow_confidence: float
    motion_score: float
    foreground_crossing_score: float
    flicker_risk: float
    largest_residual_component_px: int
    residual_mask_ratio_of_roi: float
    predicted_vram_mb: int | None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    reason_codes: tuple[str, ...]
    features: SegmentFeatures
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InpaintRequest:
    segment_id: str
    frames_bgr: Sequence[np.ndarray]
    masks: Sequence[np.ndarray]
    pts_us: Sequence[int]
    roi_xyxy: tuple[int, int, int, int]
    context: Mapping[str, Any]


@dataclass(slots=True)
class InpaintResult:
    segment_id: str
    frames_bgr: list[np.ndarray]
    engine: str
    model_hash: str
    parameters: Mapping[str, Any]
    peak_vram_mb: int | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class QualityMetric:
    name: str
    value: float
    threshold: float
    passed: bool
    frame_pts_us: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QualityReport:
    segment_id: str
    passed: bool
    metrics: tuple[QualityMetric, ...]
    retry_action: str | None
    review_reason: str | None


@dataclass(frozen=True, slots=True)
class ModelManifestEntry:
    name: str
    version: str
    sha256: str
    license_name: str
    source_url: str
    runtime: str
    minimum_runtime_version: str | None
    path: Path | None = None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return {"num": value.numerator, "den": value.denominator}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (JobState, Route, TrackClassification)):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class ProbeAdapter(Protocol):
    def probe(self, input_path: Path) -> ProbeResult: ...


class SceneDetector(Protocol):
    def detect(self, probe: ProbeResult) -> Sequence[Shot]: ...


class TextDetector(Protocol):
    def detect(
        self,
        frames: Sequence[VideoFrame],
        rois: Sequence[tuple[int, int, int, int]] | None = None,
        tier: str = "small",
    ) -> Sequence[TextDetection]: ...


class TrackBuilder(Protocol):
    def build(
        self, frames: Sequence[VideoFrame], detections: Sequence[TextDetection], shot: Shot
    ) -> Sequence[SubtitleTrack]: ...


class MaskGenerator(Protocol):
    def generate(
        self, frames: Sequence[VideoFrame], tracks: Sequence[SubtitleTrack]
    ) -> Sequence[MaskFrame]: ...


class CleanPlateReconstructor(Protocol):
    def reconstruct_sequence(
        self, frames: Sequence[VideoFrame], masks: Sequence[MaskFrame]
    ) -> Sequence[CleanPlateResult]: ...


class Router(Protocol):
    def decide(self, features: SegmentFeatures) -> RouteDecision: ...


class Inpainter(Protocol):
    @property
    def name(self) -> str: ...

    def available(self) -> bool: ...

    def inpaint(self, request: InpaintRequest) -> InpaintResult: ...


class QualityGate(Protocol):
    def evaluate(
        self,
        source_frames: Sequence[np.ndarray],
        output_frames: Sequence[np.ndarray],
        masks: Sequence[MaskFrame],
        segment_id: str,
    ) -> QualityReport: ...
