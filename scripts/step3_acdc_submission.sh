#!/usr/bin/env bash
# Step 3: ACDC test-server submission of the step-2 model (semantic segmentation, "all").
# Test-time augmentation: softmax averaged over scales 0.5-1.75 x horizontal flip at native
# resolution (val 74.49 single-scale -> 76.62). Writes, under output/acdc_submission/:
#   <NAME>_val_tta.json  val score of exactly this inference setting
#   <NAME>/labelTrainIds/<cond>/<seq>/<frame>_labelTrainIds.png   (2000 test predictions)
#   <NAME>.zip           validated against the site rules, < 200 MB
#   <NAME>_model.pth     weights-only model for the form's "Model File" field (< 600 MB)
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
NAME=lidseg_pidnetl_uber_tta
SUB=$REPO/output/acdc_submission
TTA=(--checkpoint "$STEP2/best.pth" --scales 0.5 0.75 1 1.25 1.5 1.75 --flip --amp)
mkdir -p "$SUB"
rm -rf "${SUB:?}/$NAME" "$SUB/$NAME.zip"

echo "[step3] $(date -Is) val score with TTA -> $SUB/${NAME}_val_tta.json"
CUDA_VISIBLE_DEVICES=$GPU_A python tools/acdc_tta.py val "${TTA[@]}" \
    --out_json "$SUB/${NAME}_val_tta.json"

echo "[step3] $(date -Is) test predictions -> $SUB/$NAME/labelTrainIds"
CUDA_VISIBLE_DEVICES=$GPU_A python tools/acdc_tta.py submit "${TTA[@]}" \
    --conditions fog night --out_root "$SUB/$NAME" --out_json "$SUB/$NAME/report_fog_night.json" &
PID_A=$!
CUDA_VISIBLE_DEVICES=$GPU_B python tools/acdc_tta.py submit "${TTA[@]}" \
    --conditions rain snow --out_root "$SUB/$NAME" --out_json "$SUB/$NAME/report_rain_snow.json" &
PID_B=$!
wait "$PID_A"
wait "$PID_B"

python tools/acdc_package_submission.py --sub_dir "$SUB/$NAME"
python tools/export_weights.py --checkpoint "$STEP2/best.pth" --out "$SUB/${NAME}_model.pth"
echo "[step3] $(date -Is) done: upload $SUB/$NAME.zip and $SUB/${NAME}_model.pth"
