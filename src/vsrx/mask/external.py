from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vsrx.domain.contracts import MaskFrame, VideoFrame
from vsrx.mask.matte_io import import_masks
from vsrx.utils.geometry import expand_bbox, mask_bbox
from vsrx.utils.hash import sha256_file, stable_json_hash
from vsrx.utils.image import soft_alpha_from_mask


class ExternalMaskProvider:
    """Load deterministic user-provided masks by frame index or PTS.

    Supported inputs:
    - A ``.npz`` archive produced by :func:`vsrx.mask.matte_io.export_masks`.
    - A directory containing PNG/BMP/TIFF/JPEG/NPY masks.  Recommended names
      are ``00000042.png`` or ``frame_00000042.png``.  ``pts_1234567.png`` is
      also accepted.
    - A directory with ``masks.json`` whose ``frames`` mapping maps frame index
      or PTS keys to relative mask files.

    Missing frames are interpreted as empty masks.  Images are converted to a
    binary mask and resized with nearest-neighbour interpolation when needed.
    """

    _IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp", ".npy"}

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._index_paths: dict[int, Path] = {}
        self._pts_paths: dict[int, Path] = {}
        self._archive: dict[int, np.ndarray] = {}
        self._cache: dict[Path, np.ndarray] = {}
        self._scan()

    @staticmethod
    def _numeric_token(stem: str) -> tuple[str, int] | None:
        lowered = stem.lower()
        for prefix, kind in (("frame_", "frame"), ("mask_", "frame"), ("pts_", "pts")):
            if lowered.startswith(prefix) and lowered[len(prefix) :].isdigit():
                return kind, int(lowered[len(prefix) :])
        if lowered.isdigit():
            return "frame", int(lowered)
        return None

    def _scan_manifest(self, manifest_path: Path) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = payload.get("frames", payload)
        if isinstance(mapping, list):
            for item in mapping:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                target = (manifest_path.parent / str(item["path"])).resolve()
                if item.get("frame_index") is not None:
                    self._index_paths[int(item["frame_index"])] = target
                if item.get("pts_us") is not None:
                    self._pts_paths[int(item["pts_us"])] = target
            return
        if isinstance(mapping, dict):
            for raw_key, raw_value in mapping.items():
                if not isinstance(raw_value, str):
                    continue
                key = str(raw_key).lower()
                target = (manifest_path.parent / raw_value).resolve()
                if key.startswith("pts_"):
                    self._pts_paths[int(key[4:])] = target
                else:
                    self._index_paths[int(key.removeprefix("frame_"))] = target
            return
        raise ValueError(
            f"external mask manifest frames must be a mapping or list: {manifest_path}"
        )

    def _scan(self) -> None:
        if self.path.is_file():
            if self.path.suffix.lower() != ".npz":
                raise ValueError("external mask file must be a VSR-X .npz archive")
            for item in import_masks(self.path):
                self._archive[item.frame.frame_index] = item.hard_mask
            return
        manifest = self.path / "masks.json"
        if manifest.is_file():
            self._scan_manifest(manifest)
        for item in sorted(self.path.iterdir()):
            if not item.is_file() or item.suffix.lower() not in self._IMAGE_SUFFIXES:
                continue
            token = self._numeric_token(item.stem)
            if token is None:
                continue
            kind, value = token
            if kind == "pts":
                self._pts_paths.setdefault(value, item)
            else:
                self._index_paths.setdefault(value, item)

    @property
    def digest(self) -> str:
        if self.path.is_file():
            return sha256_file(self.path)
        entries: list[dict[str, Any]] = []
        paths = sorted(set(self._index_paths.values()) | set(self._pts_paths.values()))
        manifest = self.path / "masks.json"
        if manifest.is_file():
            paths.append(manifest)
        for item in sorted(set(paths)):
            entries.append(
                {
                    "relative": str(item.relative_to(self.path))
                    if self.path in item.parents
                    else str(item),
                    "size": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
        if self._archive:
            entries.append(
                {
                    "archive_frames": [
                        [index, tuple(mask.shape), int(np.count_nonzero(mask))]
                        for index, mask in sorted(self._archive.items())
                    ]
                }
            )
        return stable_json_hash(entries)

    def _read_path(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        if path.suffix.lower() == ".npy":
            raw = np.load(path, allow_pickle=False)
        else:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"failed to read external mask: {path}")
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                raw = np.maximum(cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY), raw[..., 3])
            else:
                raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
        values = np.asarray(raw)
        maximum = float(np.nanmax(values)) if values.size else 0.0
        threshold = 0.05 if maximum <= 1.0 else 8.0
        mask = np.where(values > threshold, 255, 0).astype(np.uint8)
        self._cache[path] = mask
        return mask

    def _raw_for(self, frame: VideoFrame) -> np.ndarray | None:
        archived = self._archive.get(frame.ref.frame_index)
        if archived is not None:
            return archived
        path = self._index_paths.get(frame.ref.frame_index)
        if path is None:
            path = self._pts_paths.get(frame.ref.pts_us)
        return self._read_path(path) if path is not None else None

    def generate(self, frames: Sequence[VideoFrame], retry_expand_px: int = 0) -> list[MaskFrame]:
        result: list[MaskFrame] = []
        for frame in frames:
            height, width = frame.image_bgr.shape[:2]
            raw = self._raw_for(frame)
            if raw is None:
                hard = np.zeros((height, width), dtype=np.uint8)
            else:
                hard = raw
                if hard.shape != (height, width):
                    hard = cv2.resize(hard, (width, height), interpolation=cv2.INTER_NEAREST)
                hard = np.where(hard > 0, 255, 0).astype(np.uint8)
            if retry_expand_px > 0 and np.any(hard):
                radius = max(1, int(retry_expand_px))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                )
                hard = cv2.dilate(hard, kernel)
            feather = max(1, int(round(height / 1080 * 5)))
            result.append(
                MaskFrame(
                    frame=frame.ref,
                    hard_mask=hard,
                    soft_alpha=soft_alpha_from_mask(hard, feather),
                    source_track_ids=("external_mask",) if np.any(hard) else (),
                    confidence=1.0,
                    mask_ratio_of_frame=float(np.mean(hard > 0)),
                    expanded_bbox_xyxy=mask_bbox(hard),
                )
            )
        return result

    def discover_rois(
        self, width: int, height: int, *, max_samples: int = 180
    ) -> list[tuple[int, int, int, int]]:
        masks: list[np.ndarray] = list(self._archive.values())
        paths = sorted(set(self._index_paths.values()) | set(self._pts_paths.values()))
        if len(paths) > max_samples:
            positions = np.linspace(0, len(paths) - 1, max_samples).round().astype(int)
            paths = [paths[index] for index in positions]
        masks.extend(self._read_path(path) for path in paths)
        boxes: list[tuple[int, int, int, int]] = []
        for raw in masks:
            mask = (
                raw
                if raw.shape == (height, width)
                else cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
            )
            bbox = mask_bbox(mask)
            if bbox is not None:
                boxes.append(bbox)
        if not boxes:
            return []
        union = (
            min(item[0] for item in boxes),
            min(item[1] for item in boxes),
            max(item[2] for item in boxes),
            max(item[3] for item in boxes),
        )
        return [expand_bbox(union, max(12, int(round(height * 0.025))), width, height)]
