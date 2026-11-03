#!/usr/bin/env bash
# Standalone hand-off folder of the step-2 ACDC model for fault-injection studies:
#   README.md, requirements.txt, run_eval.sh, setup_env.sh, lidseg_infer.py, eval_acdc.py
#                               from deploy/
#   models/                     the inference model code, byte-identical to this repo
#   lidseg_pidnetl_acdc.pth     weights-only export of the step-2 best.pth
#   data/ACDC/                  ACDC val split: rgb_anon images, labelTrainIds labels, and the
#                               dataset's own README.md and License.pdf
#   reference_acdc_val.json     golden run, made by the folder's own eval_acdc.py
#   layers.csv                  per-module table, made by the folder's own lidseg_infer.py
#   SHA256SUMS
# The folder is then checked the way the recipient will use it: `bash run_eval.sh` must
# print PASS. Writes output/handoff/lidseg_pidnetl_acdc/ and lidseg_pidnetl_acdc.tar.gz.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STEP2=$REPO/output/acdc/pidnetl_norecon_uber_bblr0.1
if [[ ! -f "$STEP2/DONE" || ! -f "$STEP2/best.pth" ]]; then
    echo "step 2 is not finished: $STEP2/DONE and best.pth are required" >&2
    exit 1
fi
GPU=${GPU:-0}
NAME=lidseg_pidnetl_acdc
HANDOFF=$REPO/output/handoff
OUT=$HANDOFF/$NAME
ACDC=$(python -c "from configs.paths import load; print(load()['ACDC_ROOT'])")
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/models/segmodels" "$OUT/data/ACDC"

cp models/{__init__,lidseg,dehazeformer,lgf}.py "$OUT/models/"
cp models/segmodels/{__init__,model_registry,feature_registry,pidnet}.py "$OUT/models/segmodels/"
cp deploy/{README.md,requirements.txt,run_eval.sh,setup_env.sh,lidseg_infer.py,eval_acdc.py} "$OUT/"
chmod +x "$OUT"/*.sh
python tools/export_weights.py --checkpoint "$STEP2/best.pth" --out "$OUT/$NAME.pth"

echo "[bundle] $(date -Is) copying the ACDC val split from $ACDC"
(cd "$ACDC" && cp --parents README.md License.pdf rgb_anon/*/val/*/*_rgb_anon.png \
    gt/*/val/*/*_gt_labelTrainIds.png "$OUT/data/ACDC/")
n_img=$(find "$OUT/data/ACDC/rgb_anon" -name '*_rgb_anon.png' | wc -l)
n_gt=$(find "$OUT/data/ACDC/gt" -name '*_gt_labelTrainIds.png' | wc -l)
if [[ "$n_img" != 406 || "$n_gt" != 406 ]]; then
    echo "expected 406 images and 406 labels, copied $n_img and $n_gt" >&2
    exit 1
fi

cd "$OUT"
echo "[bundle] $(date -Is) layer table and reference run on GPU $GPU, from inside the folder"
CUDA_VISIBLE_DEVICES=$GPU python lidseg_infer.py --layers layers.csv
CUDA_VISIBLE_DEVICES=$GPU python eval_acdc.py --out_json reference_acdc_val.json
find . -name __pycache__ -prune -exec rm -rf {} +
find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS

echo "[bundle] $(date -Is) recipient check: bash run_eval.sh"
check=$(CUDA_VISIBLE_DEVICES=$GPU bash run_eval.sh)
echo "$check"
grep -q "^PASS" <<<"$check"
rm -f golden_acdc_val.json
find . -name __pycache__ -prune -exec rm -rf {} +

cd "$HANDOFF"
# Python's tarfile, not GNU tar: on the CIFS share GNU tar reports unchanged directories as
# "changed as we read it" and exits 1.
python -m tarfile -c "$NAME.tar.gz" "$NAME"
echo "[bundle] $(date -Is) -> $OUT ($(du -sh "$OUT" | cut -f1)), $NAME.tar.gz ($(du -h "$NAME.tar.gz" | cut -f1))"
