from __future__ import annotations

from vsrx.domain.contracts import SegmentFeatures
from vsrx.domain.enums import Route
from vsrx.routing.router import AdaptiveRouter


def features(**kwargs) -> SegmentFeatures:
    values = dict(
        shot_id=0,
        start_pts_us=0,
        end_pts_us=1_000_000,
        frame_count=20,
        mask_ratio_of_frame=0.02,
        mean_clean_plate_coverage=0.95,
        mean_clean_plate_confidence=0.88,
        mean_flow_confidence=0.9,
        motion_score=0.1,
        foreground_crossing_score=0.0,
        flicker_risk=0.1,
        largest_residual_component_px=0,
        residual_mask_ratio_of_roi=0.0,
        predicted_vram_mb=1000,
    )
    values.update(kwargs)
    return SegmentFeatures(**values)


def test_router_copy_tbe_and_spatial(fast_config) -> None:
    router = AdaptiveRouter(fast_config)
    assert router.decide(features(mask_ratio_of_frame=0.0)).route == Route.COPY
    assert router.decide(features()).route == Route.TBE_ONLY
    decision = router.decide(
        features(
            mean_clean_plate_coverage=0.7,
            mean_clean_plate_confidence=0.65,
            residual_mask_ratio_of_roi=0.03,
            largest_residual_component_px=5000,
        )
    )
    assert decision.route == Route.TBE_LAMA


def test_router_short_segment_requires_review(fast_config) -> None:
    assert AdaptiveRouter(fast_config).decide(features(frame_count=3)).route == Route.REVIEW
