from __future__ import annotations

from pathlib import Path

from vsrx.domain.contracts import ProbeResult
from vsrx.domain.errors import EncodeError
from vsrx.utils.config import Config
from vsrx.utils.subprocess import run_command


class SoftSubtitleHandler:
    """Remove/extract subtitle streams without touching pixels when possible."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ffmpeg = str(config.get("probe.ffmpeg_path", "ffmpeg"))

    @staticmethod
    def _family(path: Path) -> str:
        if path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            return "mp4"
        if path.suffix.lower() == ".webm":
            return "webm"
        return "matroska"

    def _command(self, probe: ProbeResult, output_path: Path, *, copy_streams: bool) -> list[str]:
        family = self._family(output_path)
        command = [
            self.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(probe.input_path),
            "-map",
            "0:v?",
            "-map",
            "0:a?",
        ]
        if family == "matroska":
            command += ["-map", "0:d?", "-map", "0:t?"]
        command += ["-map_metadata", "0", "-map_chapters", "0", "-sn"]
        if copy_streams:
            command += ["-c", "copy"]
        elif family == "webm":
            command += [
                "-c:v",
                "libvpx-vp9",
                "-crf",
                "18",
                "-b:v",
                "0",
                "-c:a",
                "libopus",
                "-b:a",
                "160k",
            ]
        else:
            command += [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        if family == "mp4":
            command += ["-movflags", "+faststart"]
        command.append(str(output_path))
        return command

    def remove(self, probe: ProbeResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for copy_streams in (True, False):
            output_path.unlink(missing_ok=True)
            try:
                run_command(
                    self._command(probe, output_path, copy_streams=copy_streams), timeout=None
                )
                return output_path
            except Exception as exc:
                failures.append(f"copy={copy_streams}: {type(exc).__name__}: {exc}")
        raise EncodeError(
            f"failed to remove soft subtitles from {probe.input_path}",
            details={"attempts": failures, "output": str(output_path)},
        )

    def extract(self, probe: ProbeResult, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for ordinal, stream in enumerate(probe.subtitle_streams):
            extension = {"subrip": "srt", "ass": "ass", "ssa": "ssa", "webvtt": "vtt"}.get(
                stream.codec_name or "", "mks"
            )
            output = output_dir / f"subtitle_{ordinal:02d}_{stream.index}.{extension}"
            run_command(
                [
                    self.ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(probe.input_path),
                    "-map",
                    f"0:{stream.index}",
                    "-c",
                    "copy",
                    str(output),
                ],
                timeout=None,
            )
            outputs.append(output)
        return outputs
