from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import pytest
from conftest import synthetic_frames, write_video

from vsrx.app.options import ProcessOptions
from vsrx.app.pipeline import VSRXPipeline
from vsrx.app.resources import model_manifest_path
from vsrx.media.probe import FFProbeAdapter


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg is required"
)
def test_external_mask_pipeline_and_resume(tmp_path: Path, fast_config) -> None:
    _, burned, masks = synthetic_frames(count=12)
    source = write_video(tmp_path / "input.mkv", burned)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for index, mask in enumerate(masks):
        assert cv2.imwrite(str(mask_dir / f"{index:08d}.png"), mask)

    work = tmp_path / "work"
    first_output = tmp_path / "first.mkv"
    pipeline = VSRXPipeline(fast_config, model_manifest_path=model_manifest_path())
    first = pipeline.process(
        source,
        ProcessOptions(
            output_path=first_output,
            work_dir=work,
            external_mask_path=mask_dir,
            overwrite=True,
        ),
    )
    assert first.state == "done"
    assert first_output.is_file()
    probe = FFProbeAdapter(fast_config).probe(first_output)
    assert probe.width == burned[0].shape[1] and probe.height == burned[0].shape[0]

    second_output = tmp_path / "second.mkv"
    second = VSRXPipeline(fast_config, model_manifest_path=model_manifest_path()).process(
        source,
        ProcessOptions(output_path=second_output, work_dir=work, external_mask_path=mask_dir),
    )
    assert second.job_id == first.job_id
    assert second_output.is_file()
    audit = json.loads(second.audit_json.read_text(encoding="utf-8"))
    assert audit["segments"]
    assert all("validated_checkpoint" in item["route_reasons"] for item in audit["segments"])
    manifest = json.loads(
        (work / "jobs" / first.job_id / "final_output.json").read_text(encoding="utf-8")
    )
    assert Path(manifest["output_path"]) == second_output
    assert manifest["modified"] is True
