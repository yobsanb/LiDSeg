"""Evaluate one experiment on ACDC."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.paths import assert_exists, paths  # noqa: E402
from evaluation._common import (  # noqa: E402
    ACDC_CONDITIONS,
    CS_CLASSES,
    NUM_CLASSES,
    architecture_for_report,
    build_acdc_val_loader,
    build_model_from_checkpoint,
    evaluate,
    final_stage_checkpoint,
    parse_exp_name,
    repo_relative,
)
from dataset.loader import collate  # noqa: E402
from train.core.metrics import jsonable_iou, mean_iou, per_class_iou  # noqa: E402


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _AcdcTestSet(Dataset):
    """Image-only ACDC test loader."""

    def __init__(self, root: str, condition: str, size: Tuple[int, int],
                 *, full_res: bool):
        self.size = size
        self.full_res = bool(full_res)
        layout_a = os.path.join(
            root, "rgb_anon", condition, "test", "*", "*_rgb_anon.png"
        )
        files = sorted(glob.glob(layout_a))
        if not files:
            layout_b = os.path.join(
                root, "rgb_anon", "test", condition, "*_rgb_anon.png"
            )
            files = sorted(glob.glob(layout_b))
        self.files = files
        self.condition = condition

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        p = self.files[i]
        img = np.array(Image.open(p).convert("RGB"),
                       dtype=np.float32) / 255.0
        h0, w0 = img.shape[:2]
        if self.full_res:
            img_p = img
        else:
            ph, pw = self.size
            img_p = (
                np.array(
                    Image.fromarray((img * 255).astype(np.uint8)).resize(
                        (pw, ph), Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        img_p = (img_p - _MEAN) / _STD
        img_p = torch.from_numpy(img_p.transpose(2, 0, 1)).float()
        return {
            "image": img_p,
            "orig_h": h0,
            "orig_w": w0,
            "path": p,
        }


def _save_acdc_test_png(
    pred: np.ndarray, src_path: str, out_root: Path,
) -> Path:
    """Write a uint8 trainId map for one ACDC test image."""
    p = Path(src_path)
    stem = p.stem.replace("_rgb_anon", "")
    parents = list(p.parents)
    if len(parents) >= 3 and parents[0].name in ACDC_CONDITIONS:
        cond = parents[0].name
        out_path = (
            out_root / "test" / cond
            / f"{stem}_gt_labelTrainIds.png"
        )
    elif len(parents) >= 4 and parents[1].name == "test":
        scene = parents[0].name
        cond = parents[2].name
        out_path = (
            out_root / cond / "test" / scene
            / f"{stem}_gt_labelTrainIds.png"
        )
    else:
        out_path = out_root / f"{stem}_gt_labelTrainIds.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8)).save(out_path)
    return out_path


@torch.no_grad()
def predict_test(
    model: torch.nn.Module,
    condition: str,
    *,
    crop: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    out_root: Path,
    full_res: bool,
) -> int:
    ds = _AcdcTestSet(
        paths["ACDC_ROOT"], condition, size=crop, full_res=full_res)
    if len(ds) == 0:
        raise FileNotFoundError(
            f"no ACDC test images found for condition={condition!r}")
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate,
    )
    model.eval()
    n_written = 0
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        out = model.predict(x) if hasattr(model, "predict") else model(x)
        if isinstance(out, dict):
            out = out["logits"]
        if isinstance(out, (tuple, list)):
            out = out[0]
        for i in range(x.size(0)):
            h0 = int(batch["orig_h"][i])
            w0 = int(batch["orig_w"][i])
            li = out[i:i + 1]
            if li.shape[-2:] != (h0, w0):
                li = F.interpolate(
                    li, size=(h0, w0), mode="bilinear", align_corners=False,
                )
            pred = li.argmax(1)[0].cpu().numpy().astype(np.uint8)
            _save_acdc_test_png(
                pred, batch["path"][i], out_root,
            )
            n_written += 1
    if n_written != len(ds):
        raise RuntimeError(
            f"ACDC test inference wrote {n_written} predictions for {len(ds)} "
            f"{condition!r} test images")
    return n_written


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate one ACDC experiment on val (mIoU) and/or "
                    "test (predictions for the ACDC benchmark).",
    )
    p.add_argument(
        "--exp_dir", required=True,
        help="Experiment dir, e.g. output/acdc/<bb>_<variant>_uber/",
    )
    p.add_argument(
        "--split", choices=["val", "test", "both"], default="both",
        help=("Which ACDC split to evaluate. Default: both. "
              "test has no public labels — predictions are written for "
              "evaluation on the ACDC benchmark server."),
    )
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--crop_eval", action="store_true",
        help="Use a centre crop / resized test input instead of native resolution. "
             "Default: native-resolution validation and test inference.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON (default: <exp_dir>/eval_acdc.json).",
    )
    args = p.parse_args()
    assert_exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir).resolve()
    parsed = parse_exp_name(exp_dir.name, dataset="acdc")
    if parsed is None:
        raise ValueError(
            f"can't parse backbone/variant/suffix from {exp_dir.name!r}; "
            f"expected '<bb>_<variant>_{{uber,fog,night,rain,snow}}'"
        )
    backbone, variant, suffix = parsed
    print(f"[exp={exp_dir.name}]  backbone={backbone}  "
          f"variant={variant}  suffix={suffix}")

    final = final_stage_checkpoint(exp_dir)
    if final is None:
        raise FileNotFoundError(f"no checkpoint found under {exp_dir}")
    final_tag, final_path = final
    print(f"[ckpt={final_tag}]  {final_path}")
    model, architecture, meta = build_model_from_checkpoint(
        str(final_path), expected_backbone=backbone,
        expected_variant=variant, expected_n_classes=NUM_CLASSES)
    model = model.to(device)

    out: dict = {
        "experiment": exp_dir.name,
        "backbone": backbone,
        "variant": variant,
        "suffix": suffix,
        "checkpoint": repo_relative(final_path),
        "checkpoint_tag": final_tag,
        "architecture": architecture_for_report(architecture),
        "checkpoint_meta": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in meta.items()
            if k in ("stage", "epoch", "miou", "best")
        },
    }

    conds = (
        list(ACDC_CONDITIONS)
        if suffix in (None, "uber") else [suffix]
    )

    if args.split in ("val", "both"):
        out["val"] = {}
        pooled_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for c in conds:
            loader = build_acdc_val_loader(
                c, (args.crop_h, args.crop_w),
                args.batch_size, args.num_workers,
                full_res=not args.crop_eval,
            )
            rep = evaluate(
                model, loader, NUM_CLASSES, device,
                desc=f"{exp_dir.name} | val | {c}",
                return_hist=True,
            )
            print(
                f"  val | {c:>5}: "
                f"mIoU={rep['miou_percent']:6.2f}  "
                f"pAcc={rep['pixel_acc_percent']:6.2f}  "
                f"n={rep['num_images']}"
            )
            pooled_hist += np.asarray(rep.pop("hist"), dtype=np.int64)
            out["val"][c] = rep
        if len(conds) > 1:
            if pooled_hist.sum() < 1:
                raise RuntimeError(
                    "ACDC validation completed without any valid labelled pixels")
            iou = per_class_iou(pooled_hist)
            pooled_miou = mean_iou(pooled_hist) * 100.0
            pooled_pacc = float(
                np.diag(pooled_hist).sum() / max(1, pooled_hist.sum()) * 100.0
            )
            out["val"]["pooled_all"] = {
                "desc": f"{exp_dir.name} | val | pooled_all",
                "num_images": int(sum(out["val"][c]["num_images"] for c in conds)),
                "miou_percent": pooled_miou,
                "pixel_acc_percent": pooled_pacc,
                "per_class_iou_percent": {
                    CS_CLASSES[i]: v if v is None else v * 100.0
                    for i, v in enumerate(jsonable_iou(iou))
                },
                "note": "Single confusion matrix over the selected ACDC conditions.",
            }
            out["val"]["mean_of_condition_mious"] = float(np.mean(
                [out["val"][c]["miou_percent"] for c in conds]
            ))
            print(
                f"  val |   all: mIoU={pooled_miou:6.2f}  "
                f"pAcc={pooled_pacc:6.2f}  "
                f"n={out['val']['pooled_all']['num_images']}"
            )

    if args.split in ("test", "both"):
        preds_root = exp_dir / "acdc_test_predictions"
        out["test"] = {
            "predictions_root": repo_relative(preds_root),
            "counts": {},
            "note": ("ACDC test labels are not public; the saved predictions "
                     "are scored by the ACDC benchmark server."),
        }
        for c in conds:
            n = predict_test(
                model, c,
                crop=(args.crop_h, args.crop_w),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                out_root=preds_root,
                full_res=not args.crop_eval,
            )
            if n < 1:
                raise RuntimeError(
                    f"ACDC test inference generated zero predictions for {c!r}")
            target = preds_root / c / "test"
            print(
                f"  test | {c:>5}: wrote {n} prediction(s) under {target}"
            )
            out["test"]["counts"][c] = n

    out_path = Path(args.out) if args.out else exp_dir / "eval_acdc.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
