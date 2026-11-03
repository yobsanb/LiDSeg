#!/usr/bin/env bash
# Step 2: ACDC fine-tuning from step 1, resumable.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STEP1=$REPO/output/cityscapes/pidnetl_guided
if [[ ! -f "$STEP1/DONE" || ! -f "$STEP1/best.pth" ]]; then
    echo "step 1 is not finished: $STEP1/DONE and best.pth are required" >&2
    exit 1
fi
GPUS=${GPUS:-0,1}
NGPUS=$(awk -F, '{print NF}' <<<"$GPUS")
if [[ "$NGPUS" != 2 ]]; then
    echo "configs/pidnetl_step2_acdc_uber_bblr0.1.yml splits 16 images/step over exactly 2 GPUs" >&2
    exit 1
fi
OUT=$REPO/output/acdc/pidnetl_norecon_uber_bblr0.1
mkdir -p "$OUT"

echo "[step2] $(date -Is) training on GPUs $GPUS from $STEP1/best.pth -> $OUT"
CUDA_VISIBLE_DEVICES=$GPUS PRETRAIN_FROM=$STEP1/best.pth \
    torchrun --nproc_per_node=2 --master_port=$((20000 + RANDOM % 20000)) \
    train/train.py --config configs/pidnetl_step2_acdc_uber_bblr0.1.yml --output_root "$OUT"

echo "[step2] $(date -Is) evaluating -> $OUT/eval_acdc.json"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python evaluation/eval_acdc.py --exp_dir "$OUT" --split val \
    --batch_size 1
echo "[step2] $(date -Is) done"
