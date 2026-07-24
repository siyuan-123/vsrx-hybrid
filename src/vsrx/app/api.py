from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vsrx.app.options import ProcessOptions
from vsrx.app.pipeline import VSRXPipeline
from vsrx.app.resources import model_manifest_path
from vsrx.scheduler import JobRepository
from vsrx.utils.config import Config


class JobSubmitRequest(BaseModel):
    input_path: str
    output_path: str | None = None
    fixed_rois: list[tuple[int, int, int, int]] = Field(default_factory=list)
    external_mask_path: str | None = None
    force_hard_subtitle_scan: bool = False
    overwrite: bool = False
    aggressive_uncertain_removal: bool = False
    codec: str = "auto"
    device_index: int = 0


class SubmissionStatus(BaseModel):
    submission_id: str
    state: str
    job_id: str | None = None
    output_path: str | None = None
    error: str | None = None


class ApiRuntime:
    def __init__(self, config: Config) -> None:
        self.config = config
        workers = max(1, int(config.get("scheduler.cpu_analysis_workers", 2)))
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vsrx-api")
        self.lock = threading.Lock()
        self.submissions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _validate_path(path: Path) -> Path:
        resolved = path.expanduser().resolve()
        roots = [
            Path(item).expanduser().resolve()
            for item in os.environ.get("VSRX_ALLOWED_ROOTS", "").split(os.pathsep)
            if item
        ]
        if roots and not any(resolved == root or root in resolved.parents for root in roots):
            raise PermissionError(f"path is outside VSRX_ALLOWED_ROOTS: {resolved}")
        return resolved

    def submit(self, request: JobSubmitRequest) -> str:
        submission_id = uuid.uuid4().hex
        with self.lock:
            self.submissions[submission_id] = {
                "state": "queued",
                "job_id": None,
                "output_path": None,
                "error": None,
            }

        def run() -> None:
            with self.lock:
                self.submissions[submission_id]["state"] = "running"
            try:
                pipeline = VSRXPipeline(self.config, model_manifest_path=model_manifest_path())
                input_path = self._validate_path(Path(request.input_path))
                output_path = (
                    self._validate_path(Path(request.output_path)) if request.output_path else None
                )
                mask_path = (
                    self._validate_path(Path(request.external_mask_path))
                    if request.external_mask_path
                    else None
                )
                result = pipeline.process(
                    input_path,
                    ProcessOptions(
                        output_path=output_path,
                        fixed_rois=request.fixed_rois,
                        external_mask_path=mask_path,
                        force_hard_subtitle_scan=request.force_hard_subtitle_scan,
                        overwrite=request.overwrite,
                        aggressive_uncertain_removal=request.aggressive_uncertain_removal,
                        codec=request.codec,
                        device_index=request.device_index,
                    ),
                )
                with self.lock:
                    self.submissions[submission_id].update(
                        state=result.state,
                        job_id=result.job_id,
                        output_path=str(result.output_path) if result.output_path else None,
                    )
            except Exception as exc:
                with self.lock:
                    self.submissions[submission_id].update(
                        state="failed", error=f"{type(exc).__name__}: {exc}"
                    )

        self.executor.submit(run)
        return submission_id

    def status(self, submission_id: str) -> dict[str, Any] | None:
        with self.lock:
            value = self.submissions.get(submission_id)
            return dict(value) if value else None


def create_app(config: Config) -> FastAPI:
    runtime = ApiRuntime(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        runtime.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="VSR-X Hybrid API", version="1.0.0", lifespan=lifespan)

    def repository() -> JobRepository:
        work = Path(str(config.get("runtime.work_dir", "./work"))).expanduser().resolve()
        db_config = Path(str(config.get("runtime.state_db", "vsrx.sqlite3")))
        db = db_config if db_config.is_absolute() else work / db_config.name
        return JobRepository(db)

    @app.get("/health")
    def health() -> dict[str, Any]:
        pipeline = VSRXPipeline(config, model_manifest_path=model_manifest_path())
        return {"status": "ok", "version": "1.0.0", "models": pipeline.inpainters.model_status()}

    @app.get("/v1/config")
    def get_config() -> dict[str, Any]:
        return {"hash": config.hash, "profile": config.get("profile"), "config": config.as_dict()}

    @app.post("/v1/jobs", response_model=SubmissionStatus, status_code=202)
    def submit_job(request: JobSubmitRequest) -> SubmissionStatus:
        try:
            submission_id = runtime.submit(request)
        except (PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SubmissionStatus(submission_id=submission_id, state="queued")

    @app.get("/v1/submissions/{submission_id}", response_model=SubmissionStatus)
    def get_submission(submission_id: str) -> SubmissionStatus:
        status = runtime.status(submission_id)
        if status is None:
            raise HTTPException(status_code=404, detail="submission not found")
        return SubmissionStatus(submission_id=submission_id, **status)

    @app.get("/v1/jobs")
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        repo = repository()
        try:
            return repo.list_jobs(limit=min(max(limit, 1), 1000))
        finally:
            repo.close()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        repo = repository()
        try:
            job = repo.get_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            job["segments"] = repo.list_segments(job_id)
            return job
        finally:
            repo.close()

    @app.get("/v1/jobs/{job_id}/audit")
    def get_job_audit(job_id: str, event_limit: int = 1000) -> dict[str, Any]:
        repo = repository()
        try:
            job = repo.get_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            return {
                "job": job,
                "probe": repo.get_probe(job_id),
                "segments": repo.list_segments(job_id),
                "quality_reports": repo.list_quality_reports(job_id),
                "model_runs": repo.list_model_runs(job_id),
                "artifacts": repo.list_artifacts(job_id),
                "events": repo.list_events(job_id, limit=event_limit),
            }
        finally:
            repo.close()

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str) -> dict[str, str]:
        repo = repository()
        try:
            if repo.get_job(job_id) is None:
                raise HTTPException(status_code=404, detail="job not found")
            repo.request_cancel(job_id)
            return {"job_id": job_id, "status": "cancel_requested"}
        finally:
            repo.close()

    return app
