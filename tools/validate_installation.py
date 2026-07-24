#!/usr/bin/env python3
"""Validate an installed VSR-X package without relying on the source tree."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from vsrx.app.config_loader import load_runtime_config
from vsrx.app.model_manager import ModelManager
from vsrx.app.resources import model_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("./models"))
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    config = load_runtime_config(profile=args.profile)
    manager = ModelManager(config, args.models_root, model_manifest_path())
    payload = {
        "python": sys.version,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "config_hash": config.hash,
        "status": manager.status(),
        "verification": manager.verify(),
        "temporary_directory_writable": False,
    }
    with tempfile.TemporaryDirectory(prefix="vsrx-validation-") as temporary:
        probe = Path(temporary) / "write-test"
        probe.write_text("ok", encoding="utf-8")
        payload["temporary_directory_writable"] = probe.read_text(encoding="utf-8") == "ok"
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ffmpeg"] or not payload["ffprobe"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
