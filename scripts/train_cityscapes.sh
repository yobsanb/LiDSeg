#!/usr/bin/env bash
set -euo pipefail

BACKBONE=${1:-bisenetv2}
VARIANT=${2:-guided}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${OUTPUT_PATH:-./output}

case "$VARIANT" in
    guided|baseline|norecon)
        exec "$REPO/scripts/train.sh" configs/cityscapes_foggy.yml \
             "$OUT/cityscapes/${BACKBONE}_${VARIANT}" "$BACKBONE" "${NGPUS:-1}" \
             --variant "$VARIANT"
        ;;
    clear)
        exec "$REPO/scripts/train.sh" configs/cityscapes_clear.yml \
             "$OUT/pretrain/${BACKBONE}" "$BACKBONE" "${NGPUS:-1}"
        ;;
    *)
        echo "usage: $0 <backbone> [guided|baseline|norecon|clear]" >&2
        exit 1
        ;;
esac
