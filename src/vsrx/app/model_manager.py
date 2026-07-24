from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from vsrx.inpaint import InpaintRegistry
from vsrx.routing import query_gpu_memory
from vsrx.utils.config import Config
from vsrx.utils.hash import sha256_file
from vsrx.utils.onnxruntime import prepare_onnxruntime


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    name: str
    available: bool
    details: str


class ModelManager:
    def __init__(self, config: Config, root: Path, manifest_path: Path | None = None) -> None:
        self.config = config
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = manifest_path

    def _model_paths(self) -> dict[str, str]:
        return {
            "lama": os.environ.get("VSRX_LAMA_MODEL", str(self.root / "lama.onnx")),
            "migan": os.environ.get("VSRX_MIGAN_MODEL", str(self.root / "migan.onnx")),
            "propainter": os.environ.get("VSRX_PROPAINTER_REPO", str(self.root / "ProPainter")),
            "sttn": os.environ.get("VSRX_STTN_REPO", str(self.root / "STTN")),
            "sttn_checkpoint": os.environ.get(
                "VSRX_STTN_CHECKPOINT", str(self.root / "STTN/checkpoints/sttn.pth")
            ),
        }

    def status(self) -> dict[str, Any]:
        paths = self._model_paths()
        model_config = self.config.with_overrides(
            {
                "spatial_inpainting": {
                    "lama": {"model_path": paths["lama"]},
                    "migan": {"model_path": paths["migan"]},
                },
                "video_inpainting": {
                    "propainter": {"repo_path": paths["propainter"]},
                    "sttn": {
                        "repo_path": paths["sttn"],
                        "checkpoint": paths["sttn_checkpoint"],
                    },
                },
            }
        )
        registry = InpaintRegistry(model_config)
        providers: list[str] = []
        try:
            ort = prepare_onnxruntime()

            providers = ort.get_available_providers()
        except ImportError:
            pass
        rapidocr_available = False
        try:
            import rapidocr  # noqa: F401

            rapidocr_available = True
        except ImportError:
            pass
        return {
            "root": str(self.root),
            "ffmpeg": shutil.which(str(self.config.get("probe.ffmpeg_path", "ffmpeg"))),
            "ffprobe": shutil.which(str(self.config.get("probe.ffprobe_path", "ffprobe"))),
            "rapidocr": rapidocr_available,
            "onnxruntime_providers": providers,
            "gpu": (
                lambda value: asdict(value) if value is not None and is_dataclass(value) else value
            )(query_gpu_memory()),
            "inpainting_backends": registry.model_status(),
            "paths": paths,
        }

    def verify(self) -> dict[str, Any]:
        payload = (
            yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path and self.manifest_path.exists()
            else {}
        )
        results: dict[str, Any] = {}
        for name, entry in (payload.get("models") or {}).items():
            expected = str(entry.get("sha256", ""))
            path: Path | None = None
            if name.startswith("ppocr_") and entry.get("filename"):
                try:
                    import rapidocr

                    path = Path(rapidocr.__file__).parent / "models" / str(entry["filename"])
                except ImportError:
                    path = None
            elif name == "lama":
                path = Path(os.environ.get("VSRX_LAMA_MODEL", self.root / "lama.onnx"))
            elif name == "propainter_official":
                path = Path(os.environ.get("VSRX_PROPAINTER_REPO", self.root / "ProPainter"))
            elif name == "mi_gan":
                path = Path(os.environ.get("VSRX_MIGAN_MODEL", self.root / "migan.onnx"))
            elif name == "sttn":
                path = Path(
                    os.environ.get("VSRX_STTN_CHECKPOINT", self.root / "STTN/checkpoints/sttn.pth")
                )
            if path is None:
                results[name] = {
                    "verified": False if name.startswith("ppocr_") else None,
                    "reason": (
                        "RapidOCR is not installed"
                        if name.startswith("ppocr_")
                        else "managed by external runtime"
                    ),
                }
            elif not path.exists():
                results[name] = {"verified": False, "path": str(path), "reason": "missing"}
            elif path.is_dir():
                expected_files = dict(entry.get("files") or {})
                if expected_files:
                    file_results: dict[str, Any] = {}
                    for relative, file_entry in expected_files.items():
                        file_path = path / relative
                        file_expected = str((file_entry or {}).get("sha256", ""))
                        if not file_path.is_file():
                            file_results[relative] = {"verified": False, "reason": "missing"}
                            continue
                        actual = sha256_file(file_path)
                        size = file_path.stat().st_size
                        expected_size = (file_entry or {}).get("bytes")
                        file_results[relative] = {
                            "verified": actual == file_expected
                            and (expected_size is None or size == int(expected_size)),
                            "bytes": size,
                            "expected": file_expected,
                            "actual": actual,
                        }
                    results[name] = {
                        "verified": bool(file_results)
                        and all(item["verified"] for item in file_results.values()),
                        "path": str(path),
                        "files": file_results,
                    }
                elif name == "propainter_official":
                    script = path / "inference_propainter.py"
                    weights = path / "weights"
                    weight_files = (
                        [item for item in weights.rglob("*") if item.is_file()]
                        if weights.is_dir()
                        else []
                    )
                    ready = script.is_file() and bool(weight_files)
                    results[name] = {
                        "verified": ready,
                        "path": str(path),
                        "script_present": script.is_file(),
                        "weight_file_count": len(weight_files),
                        "reason": (
                            "repository and weights present; runtime records tree identity"
                            if ready
                            else "repository is incomplete: inference script or official weights are missing"
                        ),
                    }
                else:
                    results[name] = {
                        "verified": True,
                        "path": str(path),
                        "reason": "repository present; individual weight hashes are recorded at runtime",
                    }
            elif expected and expected != "REPLACE_ME":
                actual = sha256_file(path)
                size = path.stat().st_size
                expected_size = entry.get("bytes")
                results[name] = {
                    "verified": actual == expected
                    and (expected_size is None or size == int(expected_size)),
                    "path": str(path),
                    "bytes": size,
                    "expected": expected,
                    "actual": actual,
                }
            else:
                results[name] = {
                    "verified": True,
                    "path": str(path),
                    "actual": sha256_file(path),
                    "reason": "manifest hash not pinned",
                }
        return results

    @staticmethod
    def _run(command: list[str], cwd: Path | None = None) -> None:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)

    def install_ocr(self, gpu: bool = False) -> None:
        package = "onnxruntime-gpu>=1.20" if gpu else "onnxruntime>=1.20"
        self._run([sys.executable, "-m", "pip", "install", "rapidocr>=3.9", package])

    def install_lama(self, source: str) -> Path:
        destination = self.root / "lama.onnx"
        temporary = destination.with_suffix(".onnx.tmp")
        if source.startswith(("http://", "https://")):
            urllib.request.urlretrieve(source, temporary)
        else:
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, temporary)
        if temporary.stat().st_size < 1024 * 1024:
            temporary.unlink(missing_ok=True)
            raise ValueError("LaMa model file is unexpectedly small")
        os.replace(temporary, destination)
        return destination

    def install_propainter(self) -> Path:
        destination = self.root / "ProPainter"
        if not destination.exists():
            self._run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/sczhou/ProPainter.git",
                    str(destination),
                ]
            )
        downloaders = [
            destination / "scripts/download_models.sh",
            destination / "scripts/download_model.sh",
            destination / "download_models.sh",
        ]
        for script in downloaders:
            if script.is_file():
                self._run(["bash", str(script)], cwd=destination)
                break
        else:
            # Newer/older revisions can expose links only in README. We do not
            # scrape arbitrary URLs; the doctor command will report missing
            # weights and preserve supply-chain transparency.
            raise RuntimeError(
                f"ProPainter repository cloned to {destination}, but no official download script was found. "
                "Follow that revision's README to place official weights under weights/."
            )
        return destination

    def install_sttn(self, checkpoint_source: str | None = None) -> Path:
        destination = self.root / "STTN"
        if not destination.exists():
            self._run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/researchmm/STTN.git",
                    str(destination),
                ]
            )
        checkpoint = destination / "checkpoints/sttn.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_source:
            if checkpoint_source.startswith(("http://", "https://")):
                urllib.request.urlretrieve(checkpoint_source, checkpoint)
            else:
                shutil.copy2(Path(checkpoint_source).expanduser().resolve(), checkpoint)
        return destination
