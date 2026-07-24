from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import make_frame

from vsrx.domain.enums import JobState
from vsrx.scheduler.checkpoints import SegmentCheckpointStore
from vsrx.scheduler.db import JobRepository


def test_job_identity_reuses_checkpoints_but_updates_output(tmp_path: Path) -> None:
    repo = JobRepository(tmp_path / "state.sqlite3")
    try:
        first, created = repo.create_or_get_job(
            input_path=tmp_path / "in.mkv",
            input_hash="i",
            config_hash="c",
            model_manifest_hash="m",
            output_path=tmp_path / "one.mkv",
        )
        assert created
        second, created = repo.create_or_get_job(
            input_path=tmp_path / "in.mkv",
            input_hash="i",
            config_hash="c",
            model_manifest_hash="m",
            output_path=tmp_path / "two.mkv",
            priority=7,
        )
        assert not created and first == second
        job = repo.get_job(first)
        assert job is not None
        assert Path(job["output_path"]).name == "two.mkv"
        assert job["priority"] == 7
        repo.transition_job(first, JobState.PROBED, force=True)
        assert repo.get_job(first)["state"] == JobState.PROBED.value
    finally:
        repo.close()


def test_checkpoint_roundtrip_and_metadata(tmp_path: Path) -> None:
    store = SegmentCheckpointStore(tmp_path / "segments")
    frames = [make_frame(i, np.full((32, 48, 3), i * 20, dtype=np.uint8), fps=10) for i in range(4)]
    store.write("seg/1", frames, extra_metadata={"modified": True, "route": "tbe_only"})
    assert store.valid("seg/1")
    metadata = store.metadata("seg/1")
    assert metadata["modified"] is True
    restored = store.read("seg/1")
    assert [frame.ref.pts_us for frame in restored] == [frame.ref.pts_us for frame in frames]
    assert all(
        np.array_equal(a.image_bgr, b.image_bgr) for a, b in zip(frames, restored, strict=True)
    )
