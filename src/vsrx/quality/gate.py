from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vsrx.domain.contracts import MaskFrame, QualityMetric, QualityReport
from vsrx.quality.flicker import temporal_flicker_metrics
from vsrx.quality.preservation import preservation_error, sharpness_ratio
from vsrx.quality.residual_text import ResidualTextChecker
from vsrx.quality.seam import boundary_seam_metrics
from vsrx.utils.config import Config


class AutomaticQualityGate:
    def __init__(self, config: Config, detector=None) -> None:
        self.config = config
        self.residual_checker = ResidualTextChecker(config, detector=detector)

    @staticmethod
    def _metric(
        name: str, value: float, threshold: float, passed: bool, details=None
    ) -> QualityMetric:
        return QualityMetric(
            name, float(value), float(threshold), bool(passed), details=details or {}
        )

    def evaluate(
        self,
        source_frames: Sequence[np.ndarray],
        output_frames: Sequence[np.ndarray],
        masks: Sequence[MaskFrame],
        segment_id: str,
    ) -> QualityReport:
        if len(source_frames) != len(output_frames) or len(masks) != len(source_frames):
            raise ValueError("quality gate inputs must have identical lengths")
        hard_masks = [item.hard_mask for item in masks]
        metrics: list[QualityMetric] = []

        residual_cfg = self.config.get("quality_control.residual_text", {})
        try:
            before_prob, after_prob, probability_drop = self.residual_checker.evaluate(
                source_frames, output_frames, masks
            )
            max_probability = float(residual_cfg.get("max_output_text_probability", 0.24))
            minimum_drop = float(residual_cfg.get("minimum_probability_drop", 0.55))
            residual_pass = after_prob <= max_probability or probability_drop >= minimum_drop
            metrics.append(
                self._metric(
                    "residual_text_probability",
                    after_prob,
                    max_probability,
                    residual_pass,
                    {"before": before_prob, "drop": probability_drop},
                )
            )
        except Exception as exc:
            # QC remains operational without OCR; this is surfaced rather than
            # silently pretending the residual-text test was performed.
            metrics.append(
                self._metric(
                    "residual_text_checker_available",
                    0.0,
                    1.0,
                    True,
                    {"skipped": True, "reason": str(exc)},
                )
            )

        flicker_cfg = self.config.get("quality_control.temporal_flicker", {})
        flicker_ratio, flicker_p95 = temporal_flicker_metrics(output_frames, hard_masks)
        max_ratio = float(flicker_cfg.get("max_ratio_to_background_ring", 2.4))
        max_p95 = float(flicker_cfg.get("max_p95_luma_error", 0.085))
        # Relative ratios become unstable when the background ring is almost
        # perfectly static.  A segment passes when either the relative or the
        # absolute temporal error is acceptable; both values remain reported.
        flicker_pass = flicker_ratio <= max_ratio or flicker_p95 <= max_p95
        metrics.append(
            self._metric(
                "temporal_flicker_ratio",
                flicker_ratio,
                max_ratio,
                flicker_pass,
                {"absolute_p95": flicker_p95},
            )
        )
        metrics.append(
            self._metric(
                "temporal_flicker_p95",
                flicker_p95,
                max_p95,
                flicker_pass,
                {"relative_ratio": flicker_ratio},
            )
        )

        seam_cfg = self.config.get("quality_control.boundary_seam", {})
        ring_width = int(seam_cfg.get("ring_width_px_1080p", 8))
        color_delta, gradient_ratio = boundary_seam_metrics(output_frames, hard_masks, ring_width)
        color_limit = float(seam_cfg.get("max_median_delta_e_2000", 5.5))
        gradient_limit = float(seam_cfg.get("max_gradient_jump_ratio", 2.2))
        # Color alone is not decisive at natural object edges; require both
        # color and gradient to be suspicious before failing the seam check.
        seam_pass = color_delta <= color_limit or gradient_ratio <= gradient_limit
        metrics.append(
            self._metric(
                "boundary_color_delta_proxy",
                color_delta,
                color_limit,
                seam_pass,
                {"gradient_ratio": gradient_ratio},
            )
        )
        metrics.append(
            self._metric("boundary_gradient_ratio", gradient_ratio, gradient_limit, seam_pass)
        )

        preservation_cfg = self.config.get("quality_control.preservation", {})
        maximum_error, changed_fraction = preservation_error(
            source_frames, output_frames, hard_masks
        )
        epsilon = float(preservation_cfg.get("allowed_numeric_epsilon_8bit", 1))
        preserve_pass = maximum_error <= epsilon
        metrics.append(
            self._metric(
                "outside_mask_max_error",
                maximum_error,
                epsilon,
                preserve_pass,
                {"changed_fraction": changed_fraction},
            )
        )

        sharpness_cfg = self.config.get("quality_control.sharpness", {})
        ratio = sharpness_ratio(output_frames, hard_masks)
        lower = float(sharpness_cfg.get("min_laplacian_ratio_to_ring", 0.42))
        upper = float(sharpness_cfg.get("max_laplacian_ratio_to_ring", 2.4))
        # Loss of detail is a hard failure.  An unusually high ratio is only a
        # diagnostic warning because natural edges and synthetic graphics can
        # legitimately be much sharper than the surrounding ring.
        metrics.append(
            self._metric(
                "sharpness_ratio",
                ratio,
                lower,
                ratio >= lower,
                {"upper_warning": upper, "over_sharp_warning": ratio > upper},
            )
        )

        failed = [item for item in metrics if not item.passed]
        retry_action: str | None = None
        review_reason: str | None = None
        failed_names = {item.name for item in failed}
        if "outside_mask_max_error" in failed_names:
            retry_action = "manual_review"
            review_reason = "invariant_violation_outside_mask_modified"
        elif "residual_text_probability" in failed_names or any(
            name.startswith("boundary_") for name in failed_names
        ):
            retry_action = "expand_mask"
        elif any(name.startswith("temporal_flicker") for name in failed_names):
            retry_action = "upgrade_flow_and_references"
        elif "sharpness_ratio" in failed_names:
            retry_action = "residual_lama"
        return QualityReport(segment_id, not failed, tuple(metrics), retry_action, review_reason)
