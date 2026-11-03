"""Evaluate LiDSeg PIDNet-L on the ACDC validation set.

One forward pass per image at the native 1080x1920 resolution, fp32 with TF32 disabled and
deterministic cuDNN kernels, so repeated runs on one machine give bit-identical predictions.
Per condition, a 19x19 confusion matrix is accumulated over all its images (pixels labelled
255 are skipped); IoU = TP / (TP + FP + FN) per class and mIoU is the mean over the 19
classes. "all" pools the confusion matrices of the four conditions. Per image the report
also holds the pixel accuracy, the mIoU over the classes present in the label or the
prediction, and an MD5 of the predicted label map.

Run from the repository root with downloaded weights and a local ACDC copy.
Use --reference to compare with an existing report (PASS / FAIL):

    python deploy/eval_acdc.py --weights W.pth --acdc_root /path/to/ACDC
    python deploy/eval_acdc.py --weights W.pth --acdc_root /path/to/ACDC [--out_json F]
        [--reference reference_acdc_val.json] [--save_pred DIR]
        [--conditions night] [--limit 10]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lidseg_infer import (CLASSES, IGNORE_INDEX, load_model, sha256_file,  # noqa: E402
                          to_input)

CONDITIONS = ("fog", "night", "rain", "snow")
VAL_SIZE = {"fog": 100, "night": 106, "rain": 100, "snow": 100}
N = len(CLASSES)
TOLERANCE = 0.05  # mIoU points allowed between a run and the reference


class AcdcVal(Dataset):
    """ACDC val images (rgb_anon) with their trainId labels, sorted by path."""

    def __init__(self, root: str, conditions, limit=None):
        self.root = root
        self.items = []
        for cond in conditions:
            files = sorted(glob.glob(os.path.join(root, "rgb_anon", cond, "val", "*",
                                                  "*_rgb_anon.png")))
            if limit is None and len(files) != VAL_SIZE[cond]:
                raise FileNotFoundError(
                    f"{root}/rgb_anon/{cond}/val has {len(files)} images, expected "
                    f"{VAL_SIZE[cond]}: point --acdc_root (ACDC_ROOT for run_eval.sh) to ACDC")
            self.items += [(cond, f) for f in files[:limit]]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        cond, img = self.items[i]
        seq = os.path.basename(os.path.dirname(img))
        stem = os.path.basename(img)[: -len("_rgb_anon.png")]
        gt = os.path.join(self.root, "gt", cond, "val", seq, f"{stem}_gt_labelTrainIds.png")
        return {"image": to_input(np.asarray(Image.open(img).convert("RGB"))),
                "label": torch.from_numpy(np.asarray(Image.open(gt), dtype=np.int64)),
                "id": f"{cond}/{seq}/{stem}", "condition": cond}


def confusion(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """N x N confusion matrix (rows = ground truth) over pixels not labelled 255."""
    keep = label != IGNORE_INDEX
    return np.bincount(N * label[keep] + pred[keep], minlength=N * N).reshape(N, N)


def iou_per_class(hist: np.ndarray) -> np.ndarray:
    """IoU per class in percent; NaN where the class is absent from labels and predictions."""
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(0) + hist.sum(1) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, 100.0 * inter / union, np.nan)


def summary(hist: np.ndarray, n_images: int) -> dict:
    iou = iou_per_class(hist)
    return {"num_images": n_images, "miou": round(float(np.nanmean(iou)), 4),
            "pixel_acc": round(float(100.0 * np.trace(hist) / hist.sum()), 4),
            "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 4))
                              for c, v in zip(CLASSES, iou)}}


def strict_fp32() -> None:
    """Deterministic cuDNN kernels and full-precision fp32 (no TF32) for convs and matmuls."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cudnn, "conv"):  # fp32_precision API, torch >= 2.9
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def compare(report: dict, reference: Path) -> bool:
    """mIoU of every fully evaluated condition against the reference, and per-image MD5s."""
    ref = json.loads(reference.read_text())
    rows = [(c, report["per_condition"][c], ref["per_condition"].get(c))
            for c in report["per_condition"]]
    if set(report["per_condition"]) == set(CONDITIONS):
        rows.append(("all", report["all"], ref["all"]))
    rows = [r for r in rows if r[2] is not None and r[1]["num_images"] == r[2]["num_images"]]
    ok = True
    print(f"comparison with {reference.name}:")
    for name, got, want in rows:
        diff = got["miou"] - want["miou"]
        ok = ok and abs(diff) <= TOLERANCE
        print(f"{name:>7}: {got['miou']:6.2f} vs {want['miou']:6.2f} ({diff:+.2f})")
    ref_md5 = {i["id"]: i["pred_md5"] for i in ref["images"]}
    same = sum(ref_md5.get(i["id"]) == i["pred_md5"] for i in report["images"])
    print(f"identical label maps: {same} of {len(report['images'])} "
          "(all on the reference GPU; elsewhere a few pixels may differ)")
    if not rows:
        print("no condition was evaluated in full, so no mIoU was compared")
    elif ok:
        print(f"PASS: every mIoU is within {TOLERANCE} of the reference")
    else:
        print(f"FAIL: an mIoU differs from the reference by more than {TOLERANCE}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path,
                   default=HERE.parent / "pretrained" / "lidseg_pidnetl_uber_tta_model.pth")
    p.add_argument("--acdc_root", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None,
                   help="reference report to compare with, e.g. reference_acdc_val.json")
    p.add_argument("--out_json", type=Path, default=Path("acdc_val.json"))
    p.add_argument("--save_pred", type=Path, default=None,
                   help="also write each predicted label map as a trainId PNG here")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--limit", type=int, default=None, help="first N images per condition")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    strict_fp32()
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    ds = AcdcVal(str(args.acdc_root), args.conditions, args.limit)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    hists = {c: np.zeros((N, N), np.int64) for c in args.conditions}
    counts = dict.fromkeys(args.conditions, 0)
    images = []
    forward_s = 0.0
    t0 = time.time()
    for batch in dl:
        x = batch["image"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        logits = model.predict(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s += time.perf_counter() - t
        pred = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        label = batch["label"][0].numpy()
        if pred.shape != label.shape:
            raise RuntimeError(f"{batch['id'][0]}: prediction {pred.shape} vs label {label.shape}")
        hist = confusion(label, pred.astype(np.int64))
        cond = batch["condition"][0]
        hists[cond] += hist
        counts[cond] += 1
        images.append({"id": batch["id"][0], "pixel_acc": round(float(100.0 * np.trace(hist) / hist.sum()), 4),
                       "miou": round(float(np.nanmean(iou_per_class(hist))), 4),
                       "pred_md5": hashlib.md5(pred.tobytes()).hexdigest()})
        if args.save_pred:
            out = args.save_pred / f"{batch['id'][0]}_labelTrainIds.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred).save(out)

    report = {
        "model": {"weights": args.weights.name, "sha256": sha256_file(args.weights)},
        "protocol": {"dataset": "ACDC", "split": "val",
                     "images": "rgb_anon/<condition>/val/<seq>/*_rgb_anon.png",
                     "labels": "gt/<condition>/val/<seq>/*_gt_labelTrainIds.png",
                     "input": "1080x1920 RGB, ImageNet mean/std", "forward_passes": 1,
                     "precision": "fp32, TF32 off, deterministic cuDNN",
                     "ignore_index": IGNORE_INDEX,
                     "metric": "per-class IoU from the accumulated confusion matrix; "
                               "mIoU = mean over 19 classes; 'all' pools the 4 conditions"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": (torch.cuda.get_device_name(device)
                                                               if device.type == "cuda" else platform.processor())},
        "per_condition": {c: summary(hists[c], counts[c]) for c in args.conditions},
        "all": summary(sum(hists.values()), len(images)),
        "timing": {"seconds_total": round(time.time() - t0, 1),
                   "ms_per_forward": round(1000.0 * forward_s / len(images), 1),
                   "peak_gpu_memory_mb": (round(torch.cuda.max_memory_allocated(device) / 2**20)
                                          if device.type == "cuda" else None)},
        "images": images,
    }
    for c in args.conditions:
        r = report["per_condition"][c]
        print(f"{c:>5}: mIoU {r['miou']:6.2f}  ({r['num_images']} images)")
    print(f"  all: mIoU {report['all']['miou']:6.2f}  ({len(images)} images, "
          f"{report['timing']['ms_per_forward']} ms/forward)")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1))
    print(f"-> {args.out_json}")
    if args.reference and not compare(report, args.reference):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
