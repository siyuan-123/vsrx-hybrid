from __future__ import annotations

from dataclasses import dataclass

from vsrx.domain.contracts import QualityReport
from vsrx.utils.config import Config


@dataclass(frozen=True, slots=True)
class RetryPlan:
    attempt: int
    action: str
    mask_expand_px: int = 0
    flow_preset: str | None = None
    force_route: str | None = None


class RetryPlanner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.ladder = list(
            config.get(
                "quality_control.retry_ladder",
                [
                    "expand_mask",
                    "upgrade_flow_and_references",
                    "residual_lama",
                    "official_propainter",
                    "manual_review",
                ],
            )
        )
        self.maximum = int(config.get("quality_control.max_automatic_attempts_per_segment", 4))

    def next(self, report: QualityReport, attempt: int) -> RetryPlan | None:
        if report.passed or attempt >= self.maximum:
            return None
        suggested = report.retry_action
        start = (
            self.ladder.index(suggested)
            if suggested in self.ladder
            else min(attempt, len(self.ladder) - 1)
        )
        action = self.ladder[min(max(start, attempt), len(self.ladder) - 1)]
        increment = int(self.config.get("mask_generation.dilation.retry_increment_px", 3))
        if action == "expand_mask":
            return RetryPlan(attempt + 1, action, mask_expand_px=increment * (attempt + 1))
        if action == "upgrade_flow_and_references":
            return RetryPlan(
                attempt + 1,
                action,
                mask_expand_px=increment,
                flow_preset=str(
                    self.config.get("motion_analysis.local_flow.quality_retry", "opencv_dis_medium")
                ),
            )
        if action == "residual_lama":
            return RetryPlan(attempt + 1, action, force_route="tbe_lama")
        if action == "official_propainter":
            return RetryPlan(attempt + 1, action, force_route="official_propainter")
        return None
