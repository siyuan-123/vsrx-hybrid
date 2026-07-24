from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProcessOptions:
    output_path: Path | None = None
    work_dir: Path | None = None
    fixed_rois: list[tuple[int, int, int, int]] = field(default_factory=list)
    external_mask_path: Path | None = None
    force_hard_subtitle_scan: bool = False
    dry_run: bool = False
    overwrite: bool = False
    resume: bool = True
    codec: str = "auto"
    device_index: int = 0
    keep_intermediates: bool = False
    aggressive_uncertain_removal: bool = False
    priority: int = 0


@dataclass(frozen=True, slots=True)
class PipelineResult:
    job_id: str
    output_path: Path | None
    audit_json: Path
    audit_markdown: Path
    state: str
    review_required: bool
