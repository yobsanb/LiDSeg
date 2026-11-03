"""ACDC inference with test-time augmentation: val scoring and benchmark submission.

Both modes share one inference path, so what is scored on val is exactly what writes the
submission. Test-time augmentation averages the softmax over input scales and horizontal
flips at native 1080x1920 resolution.

    # score a TTA setting on the labelled val set (406 images)
    python tools/acdc_tta.py val --checkpoint CKPT --scales 0.5 0.75 1 1.25 1.5 1.75 --flip
    # ACDC semantic-segmentation submission: DIR/labelTrainIds/<cond>/<seq>/<frame>_labelTrainIds.png
    python tools/acdc_tta.py submit --checkpoint CKPT --scales ... --flip --out_root DIR

``--conditions`` restricts the run to some conditions, so the four can run on four GPUs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from configs.paths import load as load_paths  # noqa: E402
from evaluation._common import (  # noqa: E402
    ACDC_CONDITIONS,
    NUM_CLASSES,
    build_model_from_checkpoint,
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE = 255


class _AcdcImages(Dataset):
    """Native-resolution ACDC images of one condition and split, labels when present."""

    def __init__(self, root: str, condition: str, split: str):
        self.files = sorted(glob.glob(
            os.path.join(root, "rgb_anon", condition, split, "*", "*_rgb_anon.png")))
        if not self.files:
            raise FileNotFoundError(f"no ACDC {split} images for {condition} under {root}")
        self.root, self.condition, self.split = root, condition, split

    def __len__(self) -> int:
        return len(self.files)

    def label_path(self, img_path: str) -> str:
        seq = os.path.basename(os.path.dirname(img_path))
        stem = os.path.basename(img_path).replace("_rgb_anon.png", "")
        return os.path.join(self.root, "gt", self.condition, self.split, seq,
                            f"{stem}_gt_labelTrainIds.png")

    def __getitem__(self, i: int) -> dict:
        p = self.files[i]
        img = (np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 - _MEAN) / _STD
        item = {"image": torch.from_numpy(img.transpose(2, 0, 1).copy()), "path": p}
        if self.split != "test":
            item["label"] = torch.from_numpy(
                np.asarray(Image.open(self.label_path(p)), dtype=np.int64))
        return item


@torch.no_grad()
def tta_probs(model, x: torch.Tensor, scales, flip: bool, amp: bool) -> torch.Tensor:
    """Softmax averaged over scales x {identity, h-flip}, at the input resolution."""
    h, w = x.shape[-2:]
    acc, n = None, 0
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(
            x, size=(int(round(h * s)), int(round(w * s))), mode="bilinear",
            align_corners=False)
        for flipped in ((False, True) if flip else (False,)):
            xi = xs.flip(-1) if flipped else xs
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model.predict(xi)
            logits = logits.float()
            if flipped:
                logits = logits.flip(-1)
            if logits.shape[-2:] != (h, w):
                logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                                       align_corners=False)
            p = logits.softmax(1)
            acc = p if acc is None else acc + p
            n += 1
    return acc / n


def _hist(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    k = label != IGNORE
    return np.bincount(NUM_CLASSES * label[k] + pred[k],
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def _miou(hist: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist))
    return float(np.nanmean(iou) * 100.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["val", "submit"])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--flip", action="store_true")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (faster)")
    p.add_argument("--conditions", nargs="+", default=list(ACDC_CONDITIONS))
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if args.mode == "submit" and args.out_root is None:
        p.error("submit needs --out_root")

    root = load_paths()["ACDC_ROOT"]
    device = torch.device("cuda")
    model, _, _ = build_model_from_checkpoint(str(args.checkpoint),
                                              expected_n_classes=NUM_CLASSES)
    model = model.to(device).eval()
    split = "val" if args.mode == "val" else "test"
    report = {"mode": args.mode, "checkpoint": str(args.checkpoint),
              "scales": args.scales, "flip": args.flip, "amp": args.amp,
              "split": split, "per_condition": {}}
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    t0 = time.time()
    for cond in args.conditions:
        ds = _AcdcImages(root, cond, split)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        written = 0
        for batch in dl:
            probs = tta_probs(model, batch["image"].to(device, non_blocking=True),
                              args.scales, args.flip, args.amp)
            pred = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
            if args.mode == "val":
                hist += _hist(batch["label"][0].numpy(), pred.astype(np.int64))
                continue
            src = batch["path"][0]
            if pred.shape != (1080, 1920):
                raise RuntimeError(f"bad prediction shape for {src}: {pred.shape}")
            seq = os.path.basename(os.path.dirname(src))
            stem = os.path.basename(src).replace("_rgb_anon.png", "")
            out = args.out_root / "labelTrainIds" / cond / seq / f"{stem}_labelTrainIds.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred).save(out, optimize=True)
            written += 1
        row = {"num_images": len(ds)}
        if args.mode == "val":
            row["miou"] = _miou(hist)
            pooled += hist
            print(f"{cond:>5}: mIoU {row['miou']:6.2f}  n={len(ds)}", flush=True)
        else:
            if written != len(ds):
                raise RuntimeError(f"{cond}: wrote {written} of {len(ds)}")
            row["written"] = written
            print(f"{cond:>5}: wrote {written}", flush=True)
        report["per_condition"][cond] = row
    if args.mode == "val" and len(args.conditions) > 1:
        report["pooled_miou"] = _miou(pooled)
        print(f"  all: mIoU {report['pooled_miou']:6.2f}", flush=True)
    if args.mode == "val":
        report["hist"] = pooled.tolist()
    report["seconds"] = round(time.time() - t0, 1)
    out_json = args.out_json or (args.out_root / f"{args.mode}_report.json" if args.out_root else None)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=1))
        print(f"-> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
