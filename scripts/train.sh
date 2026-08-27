#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/train.sh <config> <output_dir> [backbone] [n_gpus] [extra args]}
OUTPUT=${2:?usage: scripts/train.sh <config> <output_dir> [backbone] [n_gpus] [extra args]}
BACKBONE=${3:-}
NGPUS=${4:-${NGPUS:-1}}
shift $(( $# < 4 ? $# : 4 ))

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARGS=(--config "$CONFIG" --output_root "$OUTPUT")
[[ -n "$BACKBONE" ]] && ARGS+=(--backbone "$BACKBONE")
ARGS+=("$@")
mkdir -p "$OUTPUT"

if [[ "$NGPUS" -gt 1 ]]; then
    echo "[train] $NGPUS GPUs -> $OUTPUT"
    torchrun --nproc_per_node="$NGPUS" --master_port=$(( 20000 + RANDOM % 20000 )) \
             train/train.py "${ARGS[@]}"
else
    echo "[train] 1 GPU -> $OUTPUT"
    python train/train.py "${ARGS[@]}"
fi
