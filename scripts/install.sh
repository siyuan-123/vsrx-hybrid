#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VSRX_VENV:-$ROOT/.venv}"
EXTRA="${VSRX_INSTALL_EXTRA:-ocr,test}"
PYTHON="${PYTHON:-python3}"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 2; }
command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 2; }

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$ROOT[$EXTRA]"
vsrx doctor --models-root "$ROOT/models"
echo "Installed. Activate with: source '$VENV/bin/activate'"
