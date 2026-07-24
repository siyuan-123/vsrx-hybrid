from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
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


class Route(StrEnum):
    COPY = "copy"
    TBE_ONLY = "tbe_only"
    TBE_TELEA = "tbe_telea"
    TBE_LAMA = "tbe_lama"
    TBE_MIGAN = "tbe_migan"
    OFFICIAL_PROPAINTER = "official_propainter"
    STTN_FALLBACK = "sttn_fallback"
    REVIEW = "review"


class TrackClassification(StrEnum):
    SUBTITLE = "subtitle"
    OVERLAY = "overlay"
    SCENE_TEXT = "scene_text"
    LOGO = "logo"
    UNCERTAIN = "uncertain"


class TransitionType(StrEnum):
    CUT = "cut"
    FADE = "fade"
    UNKNOWN = "unknown"
