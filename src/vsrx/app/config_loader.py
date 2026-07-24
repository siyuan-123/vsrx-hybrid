from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vsrx.app.resources import default_config_path, profile_overlay_path
from vsrx.utils.config import Config, parse_cli_overrides


def load_runtime_config(
    config_path: Path | None = None,
    *,
    profile: str = "balanced",
    overlay_paths: Sequence[Path] = (),
    overrides: Sequence[str] = (),
) -> Config:
    base = config_path or default_config_path()
    overlays = list(overlay_paths)
    profile_overlay = profile_overlay_path(profile)
    if profile_overlay is not None:
        overlays.insert(0, profile_overlay)
    override_mapping = parse_cli_overrides(list(overrides))
    override_mapping.setdefault("profile", profile)
    return Config.load(base, overlays, override_mapping)
