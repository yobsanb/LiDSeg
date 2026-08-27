"""Evaluate a single checkpoint on one dataset condition."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from configs.paths import assert_exists  # noqa: E402
from evaluation._common import (  # noqa: E402
    NUM_CLASSES,
    VARIANT_NAMES,
    architecture_for_report,
    build_acdc_val_loader,
    build_cs_loader,
    build_model_from_checkpoint,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument(
        "--variant", required=True, choices=list(VARIANT_NAMES),
    )
    p.add_argument(
        "--dataset", choices=["cityscapes", "acdc"], required=True,
    )
    p.add_argument(
        "--condition", required=True,
        help=("Cityscapes: 'clear' or a fog beta {0.005, 0.01, 0.02}. "
              "ACDC: one of {fog, night, rain, snow}."),
    )
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--crop_eval", action="store_true",
        help="Score a centre crop instead of the whole image (default: "
             "full resolution).",
    )
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    assert_exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, architecture, _ = build_model_from_checkpoint(
        args.checkpoint, expected_backbone=args.backbone,
        expected_variant=args.variant, expected_n_classes=NUM_CLASSES)
    model = model.to(device)

    crop = (args.crop_h, args.crop_w)
    if args.dataset == "cityscapes":
        loader = build_cs_loader(
            args.condition, crop, args.batch_size, args.num_workers,
            full_res=not args.crop_eval,
        )
    else:
        loader = build_acdc_val_loader(
            args.condition, crop, args.batch_size, args.num_workers,
            full_res=not args.crop_eval,
        )
    rep = evaluate(model, loader, NUM_CLASSES, device,
                   desc=f"{args.dataset}|{args.condition}")
    print(
        f"mIoU={rep['miou_percent']:6.2f}  "
        f"pAcc={rep['pixel_acc_percent']:6.2f}  "
        f"n={rep['num_images']}"
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).parent
        / f"test_{args.dataset}_{args.condition}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "variant": args.variant,
        "architecture": architecture_for_report(architecture),
        "dataset": args.dataset,
        "condition": args.condition,
        **rep,
    }, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
