#!/usr/bin/env bash
# Reproduce the reference result of lidseg_pidnetl_acdc.pth on ACDC val in one step:
# check the files against SHA256SUMS, evaluate on the 406 val images in data/ACDC, and
# compare with reference_acdc_val.json. Ends with PASS or FAIL; writes golden_acdc_val.json.
#
#   bash run_eval.sh                          # first GPU if there is one, else the CPU
#   CUDA_VISIBLE_DEVICES=1 bash run_eval.sh   # a specific GPU
#   DEVICE=cpu bash run_eval.sh               # force the CPU (much slower)
#   ACDC_ROOT=/data/ACDC bash run_eval.sh     # ACDC stored somewhere else
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=python3
if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
fi
if ! "$PYTHON" -c "import torch, numpy, PIL" 2>/dev/null; then
    echo "PyTorch, NumPy or Pillow is missing for $PYTHON: run 'bash setup_env.sh' first" >&2
    exit 1
fi

echo "[1/2] checking files against SHA256SUMS"
sha256sum --check --quiet --ignore-missing SHA256SUMS
echo "      all files OK"

echo "[2/2] evaluating on ACDC val"
ARGS=(--out_json golden_acdc_val.json --reference reference_acdc_val.json)
if [[ -n "${DEVICE:-}" ]]; then
    ARGS+=(--device "$DEVICE")
fi
if [[ -n "${ACDC_ROOT:-}" ]]; then
    ARGS+=(--acdc_root "$ACDC_ROOT")
fi
"$PYTHON" eval_acdc.py "${ARGS[@]}"
