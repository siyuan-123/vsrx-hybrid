from __future__ import annotations

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from vsrx.domain.contracts import ProbeResult, Rational, StreamInfo
from vsrx.domain.errors import MediaProbeError
from vsrx.utils.config import Config
from vsrx.utils.hash import fast_file_hash, sha256_file
from vsrx.utils.subprocess import run_command

logger = logging.getLogger(__name__)

_HDR_TRANSFERS = {"smpte2084", "arib-std-b67", "smpte428"}
_INTERLACED_ORDERS = {"tt", "bb", "tb", "bt"}


def _to_us(value: str | int | float | None) -> int:
    if value in (None, "N/A", ""):
        return 0
    try:
        return int(round(float(value) * 1_000_000))
    except (TypeError, ValueError):
        return 0


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            try:
                return int(round(float(item["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    return 0


def parse_probe_json(path: Path, input_hash: str, raw: dict[str, Any]) -> ProbeResult:
    streams_raw = raw.get("streams") or []
    video_raw = next(
        (stream for stream in streams_raw if stream.get("codec_type") == "video"), None
    )
    if video_raw is None:
        raise MediaProbeError(f"no video stream found: {path}")

    streams: list[StreamInfo] = []
    for stream in streams_raw:
        streams.append(
            StreamInfo(
                index=int(stream.get("index", 0)),
                codec_type=str(stream.get("codec_type", "unknown")),
                codec_name=stream.get("codec_name"),
                time_base=Rational.parse(stream.get("time_base")),
                tags={str(k): str(v) for k, v in (stream.get("tags") or {}).items()},
                disposition={str(k): int(v) for k, v in (stream.get("disposition") or {}).items()},
                profile=stream.get("profile"),
                width=int(stream["width"]) if stream.get("width") is not None else None,
                height=int(stream["height"]) if stream.get("height") is not None else None,
                pixel_format=stream.get("pix_fmt"),
            )
        )

    avg = Rational.parse(video_raw.get("avg_frame_rate"))
    nominal = Rational.parse(video_raw.get("r_frame_rate"))
    time_base = Rational.parse(video_raw.get("time_base"), Rational(1, 1_000_000))
    assert time_base is not None
    avg_value = avg.as_float() if avg else 0.0
    nominal_value = nominal.as_float() if nominal else 0.0
    is_vfr = bool(
        avg_value
        and nominal_value
        and abs(avg_value - nominal_value) / max(avg_value, nominal_value) > 0.005
    )
    nb_frames = video_raw.get("nb_frames")
    duration_us = _to_us(video_raw.get("duration")) or _to_us(
        (raw.get("format") or {}).get("duration")
    )
    if nb_frames not in (None, "N/A") and duration_us > 0 and avg_value > 0:
        estimated = int(round(duration_us / 1_000_000 * avg_value))
        with suppress(TypeError, ValueError):
            is_vfr = is_vfr or abs(int(nb_frames) - estimated) > max(2, int(estimated * 0.01))

    transfer = video_raw.get("color_transfer")
    side_data = video_raw.get("side_data_list") or []
    is_hdr = transfer in _HDR_TRANSFERS or any(
        item.get("side_data_type")
        in {
            "Mastering display metadata",
            "Content light level metadata",
            "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)",
        }
        for item in side_data
    )
    field_order = str(video_raw.get("field_order", "progressive")).lower()
    is_interlaced = field_order in _INTERLACED_ORDERS

    return ProbeResult(
        input_path=path,
        input_hash=input_hash,
        duration_us=duration_us,
        width=int(video_raw.get("width") or 0),
        height=int(video_raw.get("height") or 0),
        average_frame_rate=avg,
        nominal_frame_rate=nominal,
        video_time_base=time_base,
        is_vfr=is_vfr,
        is_hdr=is_hdr,
        is_interlaced=is_interlaced,
        color_primaries=video_raw.get("color_primaries"),
        color_transfer=transfer,
        color_space=video_raw.get("color_space"),
        color_range=video_raw.get("color_range"),
        pixel_format=video_raw.get("pix_fmt"),
        streams=tuple(streams),
        chapters=tuple(raw.get("chapters") or ()),
        rotation_degrees=_rotation(video_raw),
        start_time_us=_to_us(video_raw.get("start_time")),
        raw_probe=raw,
    )


class FFProbeAdapter:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.ffprobe = str(config.get("probe.ffprobe_path", "ffprobe"))
        self.timeout = float(config.get("probe.timeout_seconds", 60))
        self.hash_mode = str(config.get("probe.hash_mode", "fast_then_full"))

    def _hash(self, path: Path) -> str:
        if self.hash_mode == "none":
            return f"path:{path.resolve()}"
        if self.hash_mode in {"fast", "fast_then_full"}:
            return fast_file_hash(path)
        if self.hash_mode == "full":
            return sha256_file(path)
        raise MediaProbeError(f"unsupported hash mode: {self.hash_mode}")

    def probe(self, input_path: Path) -> ProbeResult:
        path = input_path.expanduser().resolve()
        if not path.is_file():
            raise MediaProbeError(f"input is not a file: {path}")
        result = run_command(
            [
                self.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                str(path),
            ],
            timeout=self.timeout,
        )
        try:
            raw = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MediaProbeError(f"ffprobe returned invalid JSON for {path}") from exc
        probe = parse_probe_json(path, self._hash(path), raw)
        logger.info(
            "probed media",
            extra={"event": "media_probed", "stage": "probe", "job_id": probe.input_hash[:16]},
        )
        return probe
