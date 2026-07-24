from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import synthetic_frames, write_video

from vsrx.media.probe import FFProbeAdapter
from vsrx.media.stream_map import SoftSubtitleHandler


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg is required"
)
def test_soft_subtitle_removal_fast_path(tmp_path: Path, fast_config) -> None:
    _, burned, _ = synthetic_frames(count=8)
    video = write_video(tmp_path / "video.mkv", burned)
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\nsoft subtitle\n\n",
        encoding="utf-8",
    )
    source = tmp_path / "with_subtitle.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(subtitle),
            "-map",
            "0:v:0",
            "-map",
            "1:0",
            "-c",
            "copy",
            str(source),
        ],
        check=True,
    )
    prober = FFProbeAdapter(fast_config)
    before = prober.probe(source)
    assert before.subtitle_streams

    output = tmp_path / "without_subtitle.mkv"
    SoftSubtitleHandler(fast_config).remove(before, output)
    after = prober.probe(output)
    assert not after.subtitle_streams
    assert after.width == before.width and after.height == before.height
    assert abs(after.duration_us - before.duration_us) <= 100_000
