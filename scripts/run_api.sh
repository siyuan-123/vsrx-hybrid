#!/usr/bin/env bash
set -euo pipefail
: "${VSRX_ALLOWED_ROOTS:?Set VSRX_ALLOWED_ROOTS before starting the API}"
exec vsrx serve \
  --host "${VSRX_HOST:-127.0.0.1}" \
  --port "${VSRX_PORT:-8765}" \
  --profile "${VSRX_PROFILE:-balanced}"
