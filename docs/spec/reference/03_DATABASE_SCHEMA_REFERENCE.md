---
doc_id: vsrx-reference-03_DATABASE_SCHEMA_REFERENCE
version: 1.1
language: zh-CN
format: markdown-only
---

# VSR-X SQLite 数据库模式（SQL 源码）

> 这是支持 WAL、断点续跑、片段重试和审计追踪的参考模式。

```sql
-- VSR-X Hybrid resumable job-state schema (reference)
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    input_path TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    model_manifest_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    output_path TEXT,
    error_code TEXT,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    review_required INTEGER NOT NULL DEFAULT 0,
    UNIQUE(input_hash, config_hash, model_manifest_hash)
);

CREATE TABLE IF NOT EXISTS media_probe (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    probe_json TEXT NOT NULL,
    duration_us INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    is_vfr INTEGER NOT NULL,
    is_hdr INTEGER NOT NULL,
    is_interlaced INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS shots (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL,
    start_pts_us INTEGER NOT NULL,
    end_pts_us INTEGER NOT NULL,
    start_frame_index INTEGER NOT NULL,
    end_frame_index INTEGER NOT NULL,
    transition TEXT NOT NULL,
    cut_confidence REAL NOT NULL,
    PRIMARY KEY(job_id, shot_id)
);

CREATE TABLE IF NOT EXISTS subtitle_tracks (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    track_id TEXT NOT NULL,
    shot_id INTEGER NOT NULL,
    classification TEXT NOT NULL,
    score REAL NOT NULL,
    roi_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    is_vertical INTEGER NOT NULL,
    is_karaoke INTEGER NOT NULL,
    is_moving INTEGER NOT NULL,
    PRIMARY KEY(job_id, track_id),
    FOREIGN KEY(job_id, shot_id) REFERENCES shots(job_id, shot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS segments (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL,
    shot_id INTEGER NOT NULL,
    start_pts_us INTEGER NOT NULL,
    end_pts_us INTEGER NOT NULL,
    state TEXT NOT NULL,
    route TEXT,
    route_reason_json TEXT,
    feature_json TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    checkpoint_path TEXT,
    output_checkpoint_path TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, segment_id),
    FOREIGN KEY(job_id, shot_id) REFERENCES shots(job_id, shot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quality_reports (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_id, segment_id, attempt),
    FOREIGN KEY(job_id, segment_id) REFERENCES segments(job_id, segment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    segment_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    retained INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    engine TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    peak_vram_mb INTEGER,
    peak_rss_mb INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_id, segment_id, attempt, engine)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    segment_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_priority ON jobs(state, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_segments_state ON segments(job_id, state, start_pts_us);
CREATE INDEX IF NOT EXISTS idx_events_job_time ON events(job_id, created_at);
```
