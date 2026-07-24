from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vsrx.media.encode import write_json_atomic


@dataclass(slots=True)
class SegmentAudit:
    segment_id: str
    shot_id: int
    start_pts_us: int
    end_pts_us: int
    route: str
    route_reasons: list[str]
    attempts: int
    qc_passed: bool
    review_required: bool
    review_reason: str | None
    engine: str
    elapsed_seconds: float
    peak_vram_mb: int | None
    feature_summary: dict[str, Any] = field(default_factory=dict)
    quality_metrics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class JobAudit:
    job_id: str
    input_path: str
    output_path: str
    input_hash: str
    config_hash: str
    profile: str
    started_at: str
    finished_at: str | None = None
    state: str = "discovered"
    soft_subtitle_fast_path: bool = False
    rois: list[tuple[int, int, int, int]] = field(default_factory=list)
    model_status: dict[str, bool] = field(default_factory=dict)
    segments: list[SegmentAudit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        job_id: str,
        input_path: Path,
        output_path: Path,
        input_hash: str,
        config_hash: str,
        profile: str,
    ) -> JobAudit:
        return cls(
            job_id,
            str(input_path),
            str(output_path),
            input_hash,
            config_hash,
            profile,
            datetime.now(UTC).isoformat(),
        )

    def finish(self, state: str) -> None:
        self.state = state
        self.finished_at = datetime.now(UTC).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "job_id": self.job_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "profile": self.profile,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "state": self.state,
            "soft_subtitle_fast_path": self.soft_subtitle_fast_path,
            "rois": self.rois,
            "model_status": self.model_status,
            "segments": [asdict(item) for item in self.segments],
            "warnings": self.warnings,
            "stage_timings": self.stage_timings,
            "summary": self.summary(),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        }

    def summary(self) -> dict[str, Any]:
        total = len(self.segments)
        heavy = sum(item.engine == "official_propainter" for item in self.segments)
        review = sum(item.review_required for item in self.segments)
        return {
            "segment_count": total,
            "qc_passed_count": sum(item.qc_passed for item in self.segments),
            "review_required_count": review,
            "review_rate": review / max(total, 1),
            "heavy_model_segment_count": heavy,
            "heavy_model_segment_rate": heavy / max(total, 1),
            "route_histogram": _histogram(item.route for item in self.segments),
            "engine_histogram": _histogram(item.engine for item in self.segments),
            "elapsed_seconds": sum(item.elapsed_seconds for item in self.segments),
        }

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "audit.json"
        markdown_path = directory / "audit.md"
        write_json_atomic(json_path, self.as_dict())
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, markdown_path

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            "# VSR-X 作业审计报告",
            "",
            f"- 作业 ID：`{self.job_id}`",
            f"- 状态：`{self.state}`",
            f"- 输入：`{self.input_path}`",
            f"- 输出：`{self.output_path}`",
            f"- 配置档位：`{self.profile}`",
            f"- 片段数：{summary['segment_count']}",
            f"- 需要复核：{summary['review_required_count']}",
            f"- 重模型片段占比：{summary['heavy_model_segment_rate']:.2%}",
            "",
            "## 片段明细",
            "",
            "| 时间范围 | 路由 | 实际引擎 | 尝试 | QC | 复核原因 |",
            "|---|---|---|---:|---|---|",
        ]
        for item in self.segments:
            start = _format_pts(item.start_pts_us)
            end = _format_pts(item.end_pts_us)
            lines.append(
                f"| {start}–{end} | `{item.route}` | `{item.engine}` | {item.attempts} | "
                f"{'通过' if item.qc_passed else '未通过'} | {item.review_reason or ''} |"
            )
        if self.warnings:
            lines.extend(["", "## 警告", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.extend(
            [
                "",
                "## 路由统计",
                "",
                "```json",
                json.dumps(summary["route_histogram"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        return "\n".join(lines)


def _histogram(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _format_pts(value: int) -> str:
    total_seconds = value / 1_000_000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
