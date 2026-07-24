from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

from vsrx.app.model_manager import ModelManager


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_propainter_requires_weights(tmp_path: Path, fast_config, monkeypatch) -> None:
    repo = tmp_path / "ProPainter"
    repo.mkdir()
    (repo / "inference_propainter.py").write_text("# test", encoding="utf-8")
    monkeypatch.setenv("VSRX_PROPAINTER_REPO", str(repo))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "propainter_official": {
                        "sha256": "REPLACE_ME",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = ModelManager(fast_config, tmp_path / "models", manifest).verify()
    assert result["propainter_official"]["verified"] is False
    assert result["propainter_official"]["script_present"] is True
    assert result["propainter_official"]["weight_file_count"] == 0
    json.dumps(result)


def test_verify_propainter_pinned_weight_files(tmp_path: Path, fast_config) -> None:
    repo = tmp_path / "models/ProPainter"
    weight = repo / "weights/model.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"pinned-weight")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "propainter_official": {
                        "files": {
                            "weights/model.pth": {
                                "bytes": weight.stat().st_size,
                                "sha256": _sha256(weight),
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = ModelManager(fast_config, tmp_path / "models", manifest).verify()

    assert result["propainter_official"]["verified"] is True
    assert result["propainter_official"]["files"]["weights/model.pth"]["verified"] is True


def test_status_resolves_optional_backends_from_models_root(
    tmp_path: Path, fast_config, monkeypatch
) -> None:
    for name in (
        "VSRX_LAMA_MODEL",
        "VSRX_MIGAN_MODEL",
        "VSRX_PROPAINTER_REPO",
        "VSRX_STTN_REPO",
        "VSRX_STTN_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    root = tmp_path / "models"
    root.mkdir()
    (root / "lama.onnx").write_bytes(b"model")
    (root / "migan.onnx").write_bytes(b"model")
    (root / "ProPainter/weights").mkdir(parents=True)
    (root / "ProPainter/inference_propainter.py").write_text("# test", encoding="utf-8")
    (root / "STTN/checkpoints").mkdir(parents=True)
    (root / "STTN/test.py").write_text("# test", encoding="utf-8")
    (root / "STTN/checkpoints/sttn.pth").write_bytes(b"model")

    status = ModelManager(fast_config, root).status()

    assert status["paths"] == {
        "lama": str(root / "lama.onnx"),
        "migan": str(root / "migan.onnx"),
        "propainter": str(root / "ProPainter"),
        "sttn": str(root / "STTN"),
        "sttn_checkpoint": str(root / "STTN/checkpoints/sttn.pth"),
    }
    onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
    assert status["inpainting_backends"] == {
        "telea": True,
        "navier_stokes": True,
        "lama": onnxruntime_available,
        "migan": onnxruntime_available,
        "propainter": True,
        "sttn": True,
    }
