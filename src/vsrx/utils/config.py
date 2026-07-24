from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

from vsrx.domain.errors import ConfigurationError
from vsrx.utils.hash import stable_json_hash


def deep_merge(
    base: MutableMapping[str, Any], override: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class Config:
    def __init__(self, data: Mapping[str, Any], source_paths: tuple[Path, ...] = ()) -> None:
        self._data = copy.deepcopy(dict(data))
        self.source_paths = source_paths
        self.validate()

    @classmethod
    def load(
        cls,
        base_path: Path,
        overlay_paths: list[Path] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Config:
        if not base_path.exists():
            raise ConfigurationError(f"configuration file not found: {base_path}")
        data = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        sources = [base_path]
        for path in overlay_paths or []:
            if not path.exists():
                raise ConfigurationError(f"configuration overlay not found: {path}")
            deep_merge(data, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            sources.append(path)
        if overrides:
            deep_merge(data, overrides)
        return cls(data, tuple(sources))

    def validate(self) -> None:
        if self._data.get("schema_version") != 1:
            raise ConfigurationError("only schema_version 1 is supported")
        profile = self._data.get("profile")
        if profile not in {"fast", "balanced", "quality", "cpu_economy"}:
            raise ConfigurationError(f"unsupported profile: {profile}")
        if self.get("routing.segment_min_frames", 1) <= 0:
            raise ConfigurationError("routing.segment_min_frames must be positive")
        if self.get("routing.segment_max_frames", 0) < self.get("routing.segment_min_frames", 1):
            raise ConfigurationError("routing.segment_max_frames must be >= segment_min_frames")

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self._data
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigurationError(f"missing required configuration: {path}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def hash(self) -> str:
        return stable_json_hash(self._data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        data = self.as_dict()
        deep_merge(data, overrides)
        return Config(data, self.source_paths)


def set_dotted(data: MutableMapping[str, Any], key: str, value: Any) -> None:
    cursor = data
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def parse_cli_overrides(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigurationError(f"invalid override, expected key=value: {item}")
        key, raw = item.split("=", 1)
        set_dotted(result, key, yaml.safe_load(raw))
    return result
