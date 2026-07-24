from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from vsrx.domain.contracts import ProbeResult, VideoFrame
from vsrx.domain.errors import EncodeError
from vsrx.utils.config import Config
from vsrx.utils.subprocess import run_command

logger = logging.getLogger(__name__)


class FFV1CheckpointWriter:
    """Write lossless Matroska checkpoints while preserving microsecond PTS."""

    def __init__(self, output_path: Path, probe: ProbeResult) -> None:
        self.output_path = output_path
        self.probe = probe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(output_path), mode="w", format="matroska")
        rate = (
            Fraction(probe.average_frame_rate.as_fraction())
            if probe.average_frame_rate
            else Fraction(25, 1)
        )
        self._stream = self._container.add_stream("ffv1", rate=rate)
        self._stream.width = probe.width
        self._stream.height = probe.height
        self._stream.pix_fmt = "bgr0"
        self._stream.time_base = Fraction(1, 1_000_000)
        self._last_pts = -1
        self._closed = False

    def write(self, frame: VideoFrame | np.ndarray, pts_us: int | None = None) -> None:
        if self._closed:
            raise EncodeError("cannot write to closed checkpoint")
        if isinstance(frame, VideoFrame):
            image = frame.image_bgr
            pts = frame.ref.pts_us
        else:
            image = frame
            if pts_us is None:
                raise ValueError("pts_us is required when writing a raw ndarray")
            pts = pts_us
        pts = max(self._last_pts + 1, int(pts))
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = Fraction(1, 1_000_000)
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        self._last_pts = pts

    def close(self) -> Path:
        if not self._closed:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._closed = True
        return self.output_path

    def __enter__(self) -> FFV1CheckpointWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FinalEncoder:
    """Encode once at the end and remux original audio/metadata safely.

    Hardware encoders are attempted first.  Failures fall back to a software
    encoder, and incompatible copied audio falls back to a container-safe audio
    codec.  Data/attachment streams are retained only in Matroska containers.
    """

    _HARDWARE_SUFFIXES = ("_nvenc", "_qsv", "_vaapi", "_videotoolbox", "_amf")

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ffmpeg = str(config.get("probe.ffmpeg_path", "ffmpeg"))
        self._encoders: set[str] | None = None

    def _available_encoders(self) -> set[str]:
        if self._encoders is None:
            result = run_command([self.ffmpeg, "-hide_banner", "-encoders"], timeout=30)
            encoders: set[str] = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) >= 1 and parts[0][0] in {"V", "A", "S"}:
                    encoders.add(parts[1])
            self._encoders = encoders
        return self._encoders

    @staticmethod
    def _container_family(output_path: Path) -> str:
        suffix = output_path.suffix.lower()
        if suffix in {".mp4", ".m4v", ".mov"}:
            return "mp4"
        if suffix == ".webm":
            return "webm"
        return "matroska"

    def _normalize_codec(self, codec: str, output_path: Path) -> str:
        configured = str(self.config.get("encoding.video_codec", "auto"))
        selected = configured if configured != "auto" else codec
        family = self._container_family(output_path)
        if selected == "auto":
            selected = "av1" if family == "webm" else "h264"
        if family == "webm" and selected in {"h264", "h265"}:
            logger.warning(
                "WebM does not support the requested codec; using AV1", extra={"stage": "encode"}
            )
            selected = "av1"
        if selected not in {"h264", "h265", "av1"}:
            raise EncodeError(f"unsupported output codec: {selected}")
        return selected

    def _encoder_candidates(self, codec: str) -> list[tuple[str, list[str]]]:
        h264_q = str(self.config.get("encoding.cq_h264", 19))
        h265_q = str(self.config.get("encoding.cq_h265", 21))
        av1_q = str(self.config.get("encoding.cq_av1", 28))
        return {
            "h264": [
                (
                    str(self.config.get("encoding.h264_encoder", "h264_nvenc")),
                    ["-preset", "p5", "-cq", h264_q, "-b:v", "0"],
                ),
                ("h264_qsv", ["-global_quality", h264_q]),
                ("h264_videotoolbox", ["-q:v", "65"]),
                ("h264_amf", ["-quality", "quality", "-qp_i", h264_q, "-qp_p", h264_q]),
                ("libx264", ["-preset", "medium", "-crf", h264_q]),
            ],
            "h265": [
                (
                    str(self.config.get("encoding.h265_encoder", "hevc_nvenc")),
                    ["-preset", "p5", "-cq", h265_q, "-b:v", "0"],
                ),
                ("hevc_qsv", ["-global_quality", h265_q]),
                ("hevc_videotoolbox", ["-q:v", "65"]),
                ("hevc_amf", ["-quality", "quality", "-qp_i", h265_q, "-qp_p", h265_q]),
                ("libx265", ["-preset", "medium", "-crf", h265_q]),
            ],
            "av1": [
                (
                    str(self.config.get("encoding.av1_encoder", "av1_nvenc")),
                    ["-preset", "p5", "-cq", av1_q, "-b:v", "0"],
                ),
                ("av1_qsv", ["-global_quality", av1_q]),
                ("av1_amf", ["-quality", "quality", "-qp_i", av1_q, "-qp_p", av1_q]),
                ("libsvtav1", ["-preset", "8", "-crf", av1_q]),
                ("libaom-av1", ["-cpu-used", "6", "-crf", av1_q, "-b:v", "0"]),
            ],
        }[codec]

    def _choose_video_encoder(
        self, codec: str, *, software_only: bool = False
    ) -> tuple[str, list[str]]:
        available = self._available_encoders()
        for name, args in self._encoder_candidates(codec):
            hardware = name.endswith(self._HARDWARE_SUFFIXES)
            if software_only and hardware:
                continue
            if name in available:
                return name, args
        mode = "software " if software_only else ""
        raise EncodeError(f"no suitable {mode}encoder is available for codec {codec}")

    def _compatible_audio(self, family: str) -> tuple[str, list[str]]:
        available = self._available_encoders()
        if family == "webm":
            if "libopus" in available:
                return "libopus", ["-b:a", "160k"]
            if "opus" in available:
                return "opus", ["-b:a", "160k"]
        if "aac" in available:
            return "aac", ["-b:a", "192k"]
        if family == "matroska" and "libopus" in available:
            return "libopus", ["-b:a", "160k"]
        raise EncodeError(f"no container-compatible audio encoder is available for {family}")

    def _build_command(
        self,
        checkpoint: Path,
        source: ProbeResult,
        output_path: Path,
        *,
        encoder: str,
        quality_args: list[str],
        audio_codec: str,
        audio_args: list[str],
    ) -> list[str]:
        family = self._container_family(output_path)
        command: list[str] = [
            self.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(checkpoint),
            "-i",
            str(source.input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
        if family == "matroska":
            command += ["-map", "1:d?", "-map", "1:t?"]
        command += [
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:v",
            encoder,
            *quality_args,
            "-c:a",
            audio_codec,
            *audio_args,
            "-sn",
            "-fps_mode",
            "passthrough",
            "-max_muxing_queue_size",
            str(self.config.get("encoding.mux_queue_size", 4096)),
        ]
        if family == "matroska":
            command += ["-c:d", "copy", "-c:t", "copy"]
        if family == "mp4":
            command += ["-movflags", "+faststart"]

        pix_fmt = source.pixel_format or ""
        if "10" in pix_fmt and encoder in {
            "libx265",
            "hevc_nvenc",
            "av1_nvenc",
            "libsvtav1",
            "av1_qsv",
            "hevc_qsv",
        }:
            command += ["-pix_fmt", "p010le"]
        elif encoder in {"libx264", "h264_nvenc", "h264_qsv", "h264_videotoolbox", "h264_amf"}:
            command += ["-pix_fmt", "yuv420p"]
        for key, value in (
            ("color_primaries", source.color_primaries),
            ("color_trc", source.color_transfer),
            ("colorspace", source.color_space),
            ("color_range", source.color_range),
        ):
            if value:
                command += [f"-{key}", str(value)]
        command.append(str(output_path))
        return command

    def encode(
        self, checkpoint: Path, source: ProbeResult, output_path: Path, codec: str = "auto"
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_codec = self._normalize_codec(codec, output_path)
        family = self._container_family(output_path)
        configured_audio = str(self.config.get("encoding.audio_codec", "copy"))
        attempts: list[tuple[bool, str, list[str]]] = [(False, configured_audio, [])]
        attempts.append((True, configured_audio, []))
        fallback_audio, fallback_args = self._compatible_audio(family)
        if configured_audio != fallback_audio:
            attempts.append((True, fallback_audio, fallback_args))

        failures: list[str] = []
        seen: set[tuple[str, str]] = set()
        for software_only, audio_codec, audio_args in attempts:
            encoder, quality_args = self._choose_video_encoder(
                selected_codec, software_only=software_only
            )
            key = (encoder, audio_codec)
            if key in seen:
                continue
            seen.add(key)
            command = self._build_command(
                checkpoint,
                source,
                output_path,
                encoder=encoder,
                quality_args=quality_args,
                audio_codec=audio_codec,
                audio_args=audio_args,
            )
            output_path.unlink(missing_ok=True)
            try:
                run_command(command, timeout=None)
                return output_path
            except Exception as exc:
                failures.append(f"{encoder}/{audio_codec}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "encoder attempt failed; trying a safer fallback",
                    extra={"stage": "encode", "event": "encoder_fallback"},
                )
        raise EncodeError(
            f"final encoding failed: {output_path}",
            details={"attempts": failures, "codec": selected_codec, "container": family},
        )


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
