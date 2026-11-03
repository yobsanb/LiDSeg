#!/usr/bin/env bash
# Step 3: ACDC benchmark test predictions.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STEP2=$REPO/output/acdc/pidnetl_norecon_uber_bblr0.1
if [[ ! -f "$STEP2/DONE" || ! -f "$STEP2/best.pth" ]]; then
    echo "step 2 is not finished: $STEP2/DONE and best.pth are required" >&2
    exit 1
fi
GPUS=${GPUS:-0,1}
IFS=, read -r GPU_A GPU_B <<<"$GPUS"
if [[ -z "$GPU_A" || -z "$GPU_B" ]]; then
    echo "set GPUS to two GPU ids, e.g. GPUS=0,1" >&2
    exit 1
fi
NAME=lidseg_pidnetl_uber
BENCH=$REPO/output/acdc_benchmark
CKPT=(--checkpoint "$STEP2/best.pth")
mkdir -p "$BENCH"
rm -rf "${BENCH:?}/$NAME" "$BENCH/$NAME.zip"

echo "[step3] $(date -Is) val score -> $BENCH/${NAME}_val.json"
CUDA_VISIBLE_DEVICES=$GPU_A python tools/acdc_predict.py val "${CKPT[@]}" \
    --out_json "$BENCH/${NAME}_val.json"

echo "[step3] $(date -Is) test predictions -> $BENCH/$NAME/labelTrainIds"
CUDA_VISIBLE_DEVICES=$GPU_A python tools/acdc_predict.py test "${CKPT[@]}" \
    --conditions fog night --out_root "$BENCH/$NAME" --out_json "$BENCH/$NAME/report_fog_night.json" &
PID_A=$!
CUDA_VISIBLE_DEVICES=$GPU_B python tools/acdc_predict.py test "${CKPT[@]}" \
    --conditions rain snow --out_root "$BENCH/$NAME" --out_json "$BENCH/$NAME/report_rain_snow.json" &
PID_B=$!
wait "$PID_A"
wait "$PID_B"

python tools/acdc_pack_predictions.py --pred_dir "$BENCH/$NAME"
python tools/export_weights.py --checkpoint "$STEP2/best.pth" --out "$BENCH/${NAME}.pth"
echo "[step3] $(date -Is) done: $BENCH/$NAME.zip and $BENCH/${NAME}.pth"
