#!/usr/bin/env bash
# Evaluate downloaded weights on a local ACDC validation split.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=python3
if [[ -x "$DEPLOY_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$DEPLOY_DIR/.venv/bin/python"
fi

ARGS=(--out_json "${OUT_JSON:-acdc_val.json}")
if [[ -n "${WEIGHTS:-}" ]]; then
    ARGS+=(--weights "$WEIGHTS")
fi
if [[ -n "${ACDC_ROOT:-}" ]]; then
    ARGS+=(--acdc_root "$ACDC_ROOT")
fi
if [[ -n "${DEVICE:-}" ]]; then
    ARGS+=(--device "$DEVICE")
fi
if [[ -n "${REFERENCE:-}" ]]; then
    ARGS+=(--reference "$REFERENCE")
fi
exec "$PYTHON_BIN" "$DEPLOY_DIR/eval_acdc.py" "${ARGS[@]}" "$@"
