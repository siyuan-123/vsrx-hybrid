from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def bundled_resource(name: str) -> Path:
    resource = files("vsrx.resources").joinpath(name)
    # The wheel is installed unpacked in normal environments. Returning a Path
    # keeps the configuration API simple; zipped importers are handled by the
    # CLI context manager before use.
    return Path(str(resource))


def default_config_path() -> Path:
    return bundled_resource("balanced.yaml")


def profile_overlay_path(profile: str) -> Path | None:
    if profile == "balanced":
        return None
    candidate = bundled_resource(f"{profile}.yaml")
    return candidate if candidate.exists() else None


def model_manifest_path() -> Path:
    return bundled_resource("model_manifest.yaml")
