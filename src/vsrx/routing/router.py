from __future__ import annotations

import math

from vsrx.domain.contracts import RouteDecision, SegmentFeatures
from vsrx.domain.enums import Route
from vsrx.routing.budget import query_gpu_memory
from vsrx.utils.config import Config


class AdaptiveRouter:
    def __init__(self, config: Config, device_index: int = 0) -> None:
        self.config = config
        self.device_index = device_index

    def decide(self, features: SegmentFeatures) -> RouteDecision:
        reasons: list[str] = []
        params: dict[str, object] = {}
        if features.mask_ratio_of_frame <= 1e-7:
            return RouteDecision(Route.COPY, ("mask_empty",), features, {})

        review_cfg = self.config.get("routing.review_when", {})
        if features.mask_ratio_of_frame > float(review_cfg.get("mask_ratio_of_frame_above", 0.18)):
            reasons.append("very_large_mask")
        if features.frame_count < int(review_cfg.get("shot_frames_below", 5)):
            return RouteDecision(
                Route.REVIEW, ("shot_too_short_for_temporal_recovery",), features, {}
            )

        tbe_cfg = self.config.get("routing.tbe_only", {})
        if (
            features.mean_clean_plate_coverage >= float(tbe_cfg.get("min_coverage", 0.90))
            and features.mean_clean_plate_confidence
            >= float(tbe_cfg.get("min_mean_confidence", 0.76))
            and features.flicker_risk <= float(tbe_cfg.get("max_flicker_risk", 0.38))
            and features.residual_mask_ratio_of_roi <= 0.012
        ):
            return RouteDecision(Route.TBE_ONLY, ("clean_plate_sufficient",), features, {})

        spatial_cfg = self.config.get("routing.tbe_plus_spatial", {})
        telea_limit = int(self.config.get("spatial_inpainting.telea_max_component_px_1080p", 24))
        equivalent_diameter = math.sqrt(
            max(features.largest_residual_component_px, 0) * 4.0 / math.pi
        )
        if features.mean_clean_plate_coverage >= float(
            spatial_cfg.get("min_coverage", 0.55)
        ) and features.residual_mask_ratio_of_roi <= float(
            spatial_cfg.get("max_residual_mask_ratio_of_roi", 0.08)
        ):
            if equivalent_diameter <= telea_limit:
                return RouteDecision(Route.TBE_TELEA, ("small_residual_components",), features, {})
            return RouteDecision(
                Route.TBE_LAMA, ("bounded_residual_after_clean_plate",), features, {}
            )

        prop_cfg = self.config.get("routing.official_propainter", {})
        hard = (
            features.mean_clean_plate_coverage
            < float(prop_cfg.get("trigger_when_coverage_below", 0.55))
            or features.mean_flow_confidence
            < float(prop_cfg.get("trigger_when_flow_confidence_below", 0.52))
            or features.foreground_crossing_score
            > float(prop_cfg.get("trigger_when_foreground_crossing_score_above", 0.35))
            or features.mask_ratio_of_frame
            > float(prop_cfg.get("trigger_when_mask_ratio_of_frame_above", 0.06))
        )
        if hard:
            gpu = query_gpu_memory(self.device_index)
            margin = int(self.config.get("video_inpainting.propainter.vram_safety_margin_mb", 700))
            if gpu is None:
                reasons.extend(["hard_temporal_segment", "gpu_memory_unknown"])
                return RouteDecision(Route.OFFICIAL_PROPAINTER, tuple(reasons), features, params)
            params["gpu_free_mb"] = gpu.free_mb
            params["gpu_name"] = gpu.name
            if (
                features.predicted_vram_mb is None
                or features.predicted_vram_mb + margin <= gpu.free_mb
            ):
                reasons.extend(["hard_temporal_segment", "vram_budget_passed"])
                return RouteDecision(Route.OFFICIAL_PROPAINTER, tuple(reasons), features, params)
            reasons.extend(["hard_temporal_segment", "vram_budget_exceeded"])
            return RouteDecision(Route.STTN_FALLBACK, tuple(reasons), features, params)

        # Conservative middle path: let LaMa touch only the residual mask.
        reasons.append("intermediate_residual")
        return RouteDecision(Route.TBE_LAMA, tuple(reasons), features, params)
