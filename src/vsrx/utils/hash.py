from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path: Path, chunk_size: int = 4 * _CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fast_file_hash(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as stream:
        digest.update(stream.read(_CHUNK))
        if stat.st_size > _CHUNK:
            stream.seek(max(0, stat.st_size - _CHUNK))
            digest.update(stream.read(_CHUNK))
    return digest.hexdigest()


def stable_json_hash(data: object) -> str:
    import json

    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
