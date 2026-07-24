#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m compileall -q src tools
ruff format --check src tests tools
ruff check src tests tools
pytest -q
