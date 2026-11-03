#!/usr/bin/env bash
# Step 1: Foggy Cityscapes pretraining, resumable.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
GPUS=${GPUS:-0,1,2,3}
NGPUS=$(awk -F, '{print NF}' <<<"$GPUS")
if [[ "$NGPUS" != 4 ]]; then
    echo "configs/pidnetl_step1_foggy_cityscapes.yml splits 24 images/step over exactly 4 GPUs" >&2
    exit 1
fi
OUT=$REPO/output/cityscapes/pidnetl_guided
mkdir -p "$OUT"
unset PRETRAIN_FROM  # config pins PIDNet-L weights

echo "[step1] $(date -Is) training on GPUs $GPUS -> $OUT"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 20000)) \
    train/train.py --config configs/pidnetl_step1_foggy_cityscapes.yml --output_root "$OUT"

echo "[step1] $(date -Is) evaluating -> $OUT/eval_cs_foggy.json"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python evaluation/eval_cs_foggy.py --exp_dir "$OUT"
echo "[step1] $(date -Is) done"
