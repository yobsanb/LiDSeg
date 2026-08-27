#!/usr/bin/env bash
set -euo pipefail

BACKBONE=${1:-bisenetv2}
TARGET=${2:-uber}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${OUTPUT_PATH:-./output}

case "$TARGET" in
    uber)                CONDITIONS=(fog night rain snow) ;;
    fog|night|rain|snow) CONDITIONS=("$TARGET") ;;
    *)
        echo "usage: $0 <backbone> [uber|fog|night|rain|snow]" >&2
        exit 1
        ;;
esac

exec "$REPO/scripts/train.sh" configs/acdc.yml \
     "$OUT/acdc/${BACKBONE}_norecon_${TARGET}" "$BACKBONE" "${NGPUS:-1}" \
     --conditions "${CONDITIONS[@]}"
