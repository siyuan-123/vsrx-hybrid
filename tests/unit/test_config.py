from __future__ import annotations

import pytest

from vsrx.app.config_loader import load_runtime_config
from vsrx.domain.errors import ConfigurationError
from vsrx.utils.config import Config, deep_merge, parse_cli_overrides


def test_profile_merge_and_hash_are_deterministic() -> None:
    first = load_runtime_config(None, profile="fast", overrides=["routing.segment_max_frames=24"])
    second = load_runtime_config(None, profile="fast", overrides=["routing.segment_max_frames=24"])
    changed = load_runtime_config(None, profile="fast", overrides=["routing.segment_max_frames=25"])
    assert first.get("profile") == "fast"
    assert first.get("routing.segment_max_frames") == 24
    assert first.hash == second.hash
    assert first.hash != changed.hash


def test_deep_merge_and_cli_values() -> None:
    value = {"a": {"b": 1, "c": 2}}
    deep_merge(value, {"a": {"b": 4}, "x": [1, 2]})
    assert value == {"a": {"b": 4, "c": 2}, "x": [1, 2]}
    assert parse_cli_overrides(["a.b=5", "flag=true", "items=[1,2]"]) == {
        "a": {"b": 5},
        "flag": True,
        "items": [1, 2],
    }


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Config({"schema_version": 2, "profile": "fast"})
