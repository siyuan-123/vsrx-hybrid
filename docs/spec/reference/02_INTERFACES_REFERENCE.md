---
doc_id: vsrx-reference-02_INTERFACES_REFERENCE
version: 1.1
language: zh-CN
format: markdown-only
---

# VSR-X 领域对象与插件接口（Python 源码）

> 这是实现无关的数据契约基线。生产代码可以分文件，但外部语义应保持兼容。

```python
"""Reference domain contracts for the VSR-X Hybrid architecture.

This file is intentionally implementation-neutral.  It defines the stable data
contracts between probing, detection, tracking, mask generation, temporal
reconstruction, model routing, inpainting, quality control, and encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class JobState(str, Enum):
    DISCOVERED = "discovered"
    PROBED = "probed"
    ANALYZED = "analyzed"
    MASKED = "masked"
    RECONSTRUCTING = "reconstructing"
    INPAINTING = "inpainting"
    QUALITY_CHECK = "quality_check"
    ENCODING = "encoding"
    DONE = "done"
    RETRY_PENDING = "retry_pending"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Route(str, Enum):
    COPY = "copy"
    TBE_ONLY = "tbe_only"
    TBE_TELEA = "tbe_telea"
    TBE_LAMA = "tbe_lama"
    TBE_MIGAN = "tbe_migan"
    OFFICIAL_PROPAINTER = "official_propainter"
    STTN_FALLBACK = "sttn_fallback"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class Rational:
    num: int
    den: int

    def __post_init__(self) -> None:
        if self.den == 0:
            raise ValueError("denominator cannot be zero")


@dataclass(frozen=True, slots=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str | None
    time_base: Rational | None
    tags: Mapping[str, str] = field(default_factory=dict)
    disposition: Mapping[str, int] = field(default_factory=dict)


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
    pixel_format: str | None
    streams: tuple[StreamInfo, ...]
    chapters: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class FrameRef:
    frame_index: int
    pts_us: int
    shot_id: int


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TextDetection:
    frame: FrameRef
    polygon: Polygon
    confidence: float
    angle_degrees: float
    probability_map_ref: str | None = None
    detector_tier: str = "small"


@dataclass(frozen=True, slots=True)
class Shot:
    shot_id: int
    start_pts_us: int
    end_pts_us: int
    start_frame_index: int
    end_frame_index: int
    cut_confidence: float
    transition: str = "cut"


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
    classification: str  # subtitle | overlay | scene_text | logo | uncertain
    features: SubtitleTrackFeatures
    roi_xyxy: tuple[int, int, int, int]
    is_vertical: bool = False
    is_karaoke: bool = False
    is_moving: bool = False


@dataclass(slots=True)
class MaskFrame:
    frame: FrameRef
    hard_mask: np.ndarray       # uint8 HxW, values 0 or 255
    soft_alpha: np.ndarray      # float32 HxW, range [0, 1]
    source_track_ids: tuple[str, ...]
    confidence: float
    mask_ratio_of_frame: float
    expanded_bbox_xyxy: tuple[int, int, int, int] | None


@dataclass(slots=True)
class MotionField:
    source: FrameRef
    target: FrameRef
    flow_xy: np.ndarray         # float32 HxWx2
    confidence: np.ndarray      # float32 HxW
    occlusion: np.ndarray       # bool HxW
    global_transform: np.ndarray | None


@dataclass(slots=True)
class CleanPlateResult:
    frame: FrameRef
    image_bgr: np.ndarray
    coverage: np.ndarray        # float32 HxW
    confidence: np.ndarray      # float32 HxW
    residual_mask: np.ndarray   # uint8 HxW
    mean_coverage_in_mask: float
    mean_confidence_in_mask: float
    reference_frames: tuple[FrameRef, ...]


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


class ProbeAdapter(Protocol):
    def probe(self, input_path: Path) -> ProbeResult: ...


class SceneDetector(Protocol):
    def detect(self, probe: ProbeResult) -> Sequence[Shot]: ...


class TextDetector(Protocol):
    def detect(self, frames: Sequence[np.ndarray], refs: Sequence[FrameRef]) -> Sequence[TextDetection]: ...


class TrackBuilder(Protocol):
    def build(self, detections: Sequence[TextDetection], shots: Sequence[Shot]) -> Sequence[SubtitleTrack]: ...


class MaskGenerator(Protocol):
    def generate(self, frames: Sequence[np.ndarray], tracks: Sequence[SubtitleTrack]) -> Sequence[MaskFrame]: ...


class CleanPlateReconstructor(Protocol):
    def reconstruct(
        self,
        target: np.ndarray,
        target_mask: MaskFrame,
        references: Sequence[np.ndarray],
        reference_masks: Sequence[MaskFrame],
    ) -> CleanPlateResult: ...


class Router(Protocol):
    def decide(self, features: SegmentFeatures) -> RouteDecision: ...


class Inpainter(Protocol):
    @property
    def name(self) -> str: ...

    def inpaint(self, request: InpaintRequest) -> InpaintResult: ...


class QualityGate(Protocol):
    def evaluate(
        self,
        source_frames: Sequence[np.ndarray],
        output_frames: Sequence[np.ndarray],
        masks: Sequence[MaskFrame],
        segment_id: str,
    ) -> QualityReport: ...


class Encoder(Protocol):
    def encode(self, job_id: str, output_path: Path) -> Path: ...
```
