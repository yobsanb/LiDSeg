#!/usr/bin/env bash
# Create an inference environment used automatically by run_eval.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
