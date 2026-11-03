#!/usr/bin/env bash
# Only needed when PyTorch, NumPy or Pillow are missing: creates a virtual environment in
# .venv here and installs requirements.txt into it. run_eval.sh uses .venv automatically.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
