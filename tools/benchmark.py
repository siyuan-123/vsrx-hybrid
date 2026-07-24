#!/usr/bin/env python3
"""Run one reproducible pipeline benchmark and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from vsrx.app.config_loader import load_runtime_config
from vsrx.app.options import ProcessOptions
from vsrx.app.pipeline import VSRXPipeline
from vsrx.app.resources import model_manifest_path
from vsrx.media import FFProbeAdapter
from vsrx.utils.hash import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--report", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--mask-path", type=Path)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    output = (
        args.video_output.expanduser().resolve()
        if args.video_output
        else report_path.with_name(f"{source.stem}.benchmark.mkv")
    )
    config = load_runtime_config(profile=args.profile, overrides=args.set)
    prober = FFProbeAdapter(config)
    input_probe = prober.probe(source)
    started = time.perf_counter()
    result = VSRXPipeline(config, model_manifest_path=model_manifest_path()).process(
        source,
        ProcessOptions(
            output_path=output,
            external_mask_path=args.mask_path,
            overwrite=args.overwrite,
            device_index=args.device,
        ),
    )
    elapsed = time.perf_counter() - started
    output_probe = prober.probe(output) if output.exists() else None
    audit = json.loads(result.audit_json.read_text(encoding="utf-8"))
    duration_seconds = input_probe.duration_us / 1_000_000
    payload = {
        "schema_version": 1,
        "input": str(source),
        "output": str(output),
        "profile": args.profile,
        "config_hash": config.hash,
        "elapsed_seconds": elapsed,
        "input_duration_seconds": duration_seconds,
        "processing_realtime_factor": elapsed / max(duration_seconds, 1e-9),
        "video_seconds_per_wall_second": duration_seconds / max(elapsed, 1e-9),
        "input_probe": {
            "width": input_probe.width,
            "height": input_probe.height,
            "duration_us": input_probe.duration_us,
            "is_vfr": input_probe.is_vfr,
            "is_hdr": input_probe.is_hdr,
        },
        "output_probe": (
            {
                "width": output_probe.width,
                "height": output_probe.height,
                "duration_us": output_probe.duration_us,
                "is_vfr": output_probe.is_vfr,
                "is_hdr": output_probe.is_hdr,
            }
            if output_probe
            else None
        ),
        "output_sha256": sha256_file(output) if output.exists() else None,
        "audit_summary": audit.get("summary", {}),
        "stage_timings": audit.get("stage_timings", {}),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
