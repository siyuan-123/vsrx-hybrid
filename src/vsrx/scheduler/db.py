from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vsrx.domain.contracts import (
    InpaintResult,
    ProbeResult,
    QualityReport,
    Segment,
    Shot,
    SubtitleTrack,
    jsonable,
)
from vsrx.domain.enums import JobState
from vsrx.scheduler.state_machine import assert_transition


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection
        return connection

    @contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self._connection().executescript(schema)

    def create_or_get_job(
        self,
        *,
        input_path: Path,
        input_hash: str,
        config_hash: str,
        model_manifest_hash: str,
        output_path: Path,
        priority: int = 0,
    ) -> tuple[str, bool]:
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT job_id FROM jobs WHERE input_hash=? AND config_hash=? AND model_manifest_hash=?",
                (input_hash, config_hash, model_manifest_hash),
            ).fetchone()
            if existing:
                job_id = str(existing["job_id"])
                connection.execute(
                    """
                    UPDATE jobs
                    SET input_path=?, output_path=?, priority=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (
                        str(input_path.resolve()),
                        str(output_path.resolve()),
                        int(priority),
                        now,
                        job_id,
                    ),
                )
                return job_id, False
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs(job_id,input_path,input_hash,config_hash,model_manifest_hash,state,priority,created_at,updated_at,output_path)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    str(input_path.resolve()),
                    input_hash,
                    config_hash,
                    model_manifest_hash,
                    JobState.DISCOVERED.value,
                    int(priority),
                    now,
                    now,
                    str(output_path.resolve()),
                ),
            )
            return job_id, True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._connection().execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, state: JobState | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if state is None:
            rows = (
                self._connection()
                .execute(
                    "SELECT * FROM jobs ORDER BY priority DESC, created_at DESC LIMIT ?", (limit,)
                )
                .fetchall()
            )
        else:
            rows = (
                self._connection()
                .execute(
                    "SELECT * FROM jobs WHERE state=? ORDER BY priority DESC, created_at LIMIT ?",
                    (state.value, limit),
                )
                .fetchall()
            )
        return [dict(row) for row in rows]

    def transition_job(
        self,
        job_id: str,
        target: JobState,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        review_required: bool | None = None,
        force: bool = False,
    ) -> None:
        with self.transaction() as connection:
            row = connection.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = JobState(str(row["state"]))
            if not force:
                assert_transition(current, target)
            values: list[Any] = [target.value, utc_now(), error_code, error_message]
            sql = "UPDATE jobs SET state=?,updated_at=?,error_code=?,error_message=?"
            if review_required is not None:
                sql += ",review_required=?"
                values.append(int(review_required))
            if target == JobState.RETRY_PENDING:
                sql += ",retry_count=retry_count+1"
            sql += " WHERE job_id=?"
            values.append(job_id)
            connection.execute(sql, values)
            self._insert_event(
                connection, job_id, None, "job_state", {"from": current.value, "to": target.value}
            )

    def save_probe(self, job_id: str, probe: ProbeResult) -> None:
        payload = json.dumps(jsonable(probe), ensure_ascii=False)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO media_probe(job_id,probe_json,duration_us,width,height,is_vfr,is_hdr,is_interlaced)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    payload,
                    probe.duration_us,
                    probe.width,
                    probe.height,
                    int(probe.is_vfr),
                    int(probe.is_hdr),
                    int(probe.is_interlaced),
                ),
            )

    def save_shots(self, job_id: str, shots: Sequence[Shot]) -> None:
        with self.transaction() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO shots(job_id,shot_id,start_pts_us,end_pts_us,start_frame_index,end_frame_index,transition,cut_confidence)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        job_id,
                        item.shot_id,
                        item.start_pts_us,
                        item.end_pts_us,
                        item.start_frame_index,
                        item.end_frame_index,
                        item.transition,
                        item.cut_confidence,
                    )
                    for item in shots
                ],
            )

    def save_tracks(self, job_id: str, tracks: Sequence[SubtitleTrack]) -> None:
        with self.transaction() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO subtitle_tracks(job_id,track_id,shot_id,classification,score,roi_json,features_json,is_vertical,is_karaoke,is_moving)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        job_id,
                        item.track_id,
                        item.shot_id,
                        item.classification.value,
                        item.score,
                        json.dumps(item.roi_xyxy),
                        json.dumps(jsonable(item.features), ensure_ascii=False),
                        int(item.is_vertical),
                        int(item.is_karaoke),
                        int(item.is_moving),
                    )
                    for item in tracks
                ],
            )

    def upsert_segment(self, job_id: str, segment: Segment, state: str = "pending") -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO segments(job_id,segment_id,shot_id,start_pts_us,end_pts_us,state,updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(job_id,segment_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    segment.segment_id,
                    segment.shot_id,
                    segment.start_pts_us,
                    segment.end_pts_us,
                    state,
                    utc_now(),
                ),
            )

    def update_segment(
        self,
        job_id: str,
        segment_id: str,
        *,
        state: str,
        route: str | None = None,
        reasons: Sequence[str] | None = None,
        features: Any = None,
        attempt: int | None = None,
        checkpoint_path: Path | None = None,
        output_checkpoint_path: Path | None = None,
    ) -> None:
        assignments = ["state=?", "updated_at=?"]
        values: list[Any] = [state, utc_now()]
        optional = {
            "route": route,
            "route_reason_json": json.dumps(list(reasons), ensure_ascii=False)
            if reasons is not None
            else None,
            "feature_json": json.dumps(jsonable(features), ensure_ascii=False)
            if features is not None
            else None,
            "attempt": attempt,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "output_checkpoint_path": str(output_checkpoint_path)
            if output_checkpoint_path
            else None,
        }
        for field, value in optional.items():
            if value is not None:
                assignments.append(f"{field}=?")
                values.append(value)
        values.extend([job_id, segment_id])
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE segments SET {','.join(assignments)} WHERE job_id=? AND segment_id=?",
                values,
            )

    def completed_segments(self, job_id: str) -> dict[str, str]:
        rows = (
            self._connection()
            .execute(
                "SELECT segment_id,output_checkpoint_path FROM segments WHERE job_id=? AND state='done' AND output_checkpoint_path IS NOT NULL",
                (job_id,),
            )
            .fetchall()
        )
        return {str(row["segment_id"]): str(row["output_checkpoint_path"]) for row in rows}

    def list_segments(self, job_id: str) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute("SELECT * FROM segments WHERE job_id=? ORDER BY start_pts_us", (job_id,))
            .fetchall()
        )
        return [dict(row) for row in rows]

    def get_probe(self, job_id: str) -> dict[str, Any] | None:
        row = (
            self._connection()
            .execute("SELECT * FROM media_probe WHERE job_id=?", (job_id,))
            .fetchone()
        )
        if row is None:
            return None
        value = dict(row)
        try:
            value["probe"] = json.loads(str(value.pop("probe_json")))
        except (TypeError, ValueError, json.JSONDecodeError):
            value["probe"] = None
        return value

    def list_quality_reports(self, job_id: str) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM quality_reports WHERE job_id=? ORDER BY segment_id,attempt",
                (job_id,),
            )
            .fetchall()
        )
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            try:
                value["report"] = json.loads(str(value.pop("report_json")))
            except (TypeError, ValueError, json.JSONDecodeError):
                value["report"] = None
            values.append(value)
        return values

    def list_model_runs(self, job_id: str) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM model_runs WHERE job_id=? ORDER BY segment_id,attempt,engine",
                (job_id,),
            )
            .fetchall()
        )
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            try:
                value["parameters"] = json.loads(str(value.pop("parameters_json")))
            except (TypeError, ValueError, json.JSONDecodeError):
                value["parameters"] = None
            values.append(value)
        return values

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute("SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at", (job_id,))
            .fetchall()
        )
        return [dict(row) for row in rows]

    def list_events(self, job_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute(
                "SELECT * FROM events WHERE job_id=? ORDER BY event_id LIMIT ?",
                (job_id, max(1, min(int(limit), 10000))),
            )
            .fetchall()
        )
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            try:
                value["payload"] = json.loads(str(value.pop("payload_json")))
            except (TypeError, ValueError, json.JSONDecodeError):
                value["payload"] = None
            values.append(value)
        return values

    def save_quality_report(self, job_id: str, report: QualityReport, attempt: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO quality_reports(job_id,segment_id,attempt,passed,report_json,created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    job_id,
                    report.segment_id,
                    attempt,
                    int(report.passed),
                    json.dumps(jsonable(report), ensure_ascii=False),
                    utc_now(),
                ),
            )

    def save_model_run(
        self, job_id: str, result: InpaintResult, attempt: int, peak_rss_mb: int | None = None
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_runs(job_id,segment_id,attempt,engine,model_name,model_version,model_sha256,parameters_json,elapsed_seconds,peak_vram_mb,peak_rss_mb,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    result.segment_id,
                    attempt,
                    result.engine,
                    result.engine,
                    "runtime",
                    result.model_hash,
                    json.dumps(jsonable(result.parameters), ensure_ascii=False),
                    result.elapsed_seconds,
                    result.peak_vram_mb,
                    peak_rss_mb,
                    utc_now(),
                ),
            )

    def add_artifact(
        self,
        artifact_id: str,
        job_id: str,
        artifact_type: str,
        path: Path,
        sha256: str,
        segment_id: str | None = None,
        retained: bool = False,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts(artifact_id,job_id,segment_id,artifact_type,path,sha256,byte_size,retained,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    artifact_id,
                    job_id,
                    segment_id,
                    artifact_type,
                    str(path),
                    sha256,
                    path.stat().st_size,
                    int(retained),
                    utc_now(),
                ),
            )

    def event(
        self,
        job_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        segment_id: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            self._insert_event(connection, job_id, segment_id, event_type, payload)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        job_id: str,
        segment_id: str | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events(job_id,segment_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
            (
                job_id,
                segment_id,
                event_type,
                json.dumps(jsonable(payload), ensure_ascii=False),
                utc_now(),
            ),
        )

    def request_cancel(self, job_id: str) -> None:
        self.event(job_id, "cancel_requested", {})

    def cancellation_requested(self, job_id: str) -> bool:
        row = (
            self._connection()
            .execute(
                "SELECT 1 FROM events WHERE job_id=? AND event_type='cancel_requested' ORDER BY event_id DESC LIMIT 1",
                (job_id,),
            )
            .fetchone()
        )
        return row is not None

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None
