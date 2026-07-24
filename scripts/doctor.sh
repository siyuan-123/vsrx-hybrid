#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec vsrx doctor --models-root "${VSRX_MODELS_ROOT:-$ROOT/models}" "$@"
