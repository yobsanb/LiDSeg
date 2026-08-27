#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:?usage: $0 <checkpoint> [backbone] [foggy_zurich|foggy_driving] [variant]}
BACKBONE=${2:-bisenetv2}
DATASET=${3:-foggy_zurich}
VARIANT=${4:-norecon}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

exec python tools/eval_zeroshot.py --checkpoint "$CKPT" --backbone "$BACKBONE" \
     --variant "$VARIANT" --datasets "$DATASET"
