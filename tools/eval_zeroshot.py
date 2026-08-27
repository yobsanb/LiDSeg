"""Zero-shot evaluation on Foggy Zurich v2 and Foggy Driving."""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
BMVC = HERE.parent
sys.path.insert(0, str(BMVC))
from configs import paths  # noqa: E402
from dataset.loader import VALID_HW_KEY, build_acdc, collate  # noqa: E402
from evaluation._common import (  # noqa: E402
    ACDC_CONDITIONS,
    architecture_for_report,
    build_model_from_checkpoint,
    final_stage_checkpoint,
)
from train.core.recon_metrics import (
    recon_psnr_ssim_batch,
)
from train.core.metrics import jsonable_iou, mean_iou, per_class_iou  # noqa: E402

CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
NUM_CLASSES = 19
IGNORE_LABEL = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _ImgLabelDataset(Dataset):
    """Minimal eval dataset: loads (image, semantic) at full resolution."""

    def __init__(self, pairs: List[Tuple[str, str]], crop_h: int, crop_w: int,
                 *, full_res: bool):
        self.pairs = pairs
        self.size = (crop_h, crop_w)
        self.full_res = bool(full_res)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, lbl_p = self.pairs[idx]
        img = np.array(Image.open(img_p).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(lbl_p), dtype=np.int64)
        if not self.full_res:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
                (self.size[1], self.size[0]), Image.BILINEAR), dtype=np.float32) / 255.0
            lbl = np.array(Image.fromarray(lbl.astype(np.uint8)).resize(
                (self.size[1], self.size[0]), Image.NEAREST), dtype=np.int64)
        img = (img - MEAN) / STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        lbl = torch.from_numpy(lbl).long()
        return {"image": img, "semantic": lbl}


def _list_foggy_driving(root: str) -> List[Tuple[str, str]]:
    img_root = os.path.join(root, "leftImg8bit", "test")
    pairs = []
    for img_p in sorted(glob.glob(os.path.join(img_root, "*", "*_leftImg8bit.png"))):
        rel = os.path.relpath(img_p, img_root)
        lbl_p = os.path.join(root, "gtFine", "test",
                             rel.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png"))
        if os.path.exists(lbl_p):
            pairs.append((img_p, lbl_p))
    return pairs


def _list_foggy_zurich(root: str) -> List[Tuple[str, str]]:
    img_root = os.path.join(root, "RGB")
    lbl_root = os.path.join(root, "gt_labelTrainIds")
    pairs = []
    if not os.path.isdir(img_root):
        return pairs
    for seq in sorted(os.listdir(img_root)):
        seq_img = os.path.join(img_root, seq)
        seq_lbl = os.path.join(lbl_root, seq)
        if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl):
            continue
        for img_p in sorted(glob.glob(os.path.join(seq_img, "*.png"))):
            lbl_p = os.path.join(seq_lbl, os.path.basename(img_p))
            if os.path.exists(lbl_p):
                pairs.append((img_p, lbl_p))
    return pairs


def resolve_experiment_checkpoint(root: Path) -> Path:
    """Select the highest final-stage checkpoint."""
    root = root.resolve()
    final = final_stage_checkpoint(root)
    if final is not None:
        return final[1]
    raise FileNotFoundError(
        f"No final-stage checkpoint under {root}")


def _recon_metrics_at_valid_extents(
    recon: torch.Tensor,
    clear: torch.Tensor,
    paired_mask,
    valid_hw: np.ndarray,
) -> Tuple[float, float, int]:
    """Reconstruction metrics over valid (unpadded) extents."""
    mask = torch.as_tensor(paired_mask, device=recon.device).bool().reshape(-1)
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    for i in range(int(recon.shape[0])):
        if not bool(mask[i]):
            continue
        h, w = (int(valid_hw[i, 0]), int(valid_hw[i, 1]))
        if h < 1 or w < 1:
            raise RuntimeError(f"invalid zero-shot reconstruction extent {(h, w)}")
        rp, rs = recon_psnr_ssim_batch(
            recon[i:i + 1, ..., :h, :w], clear[i:i + 1, ..., :h, :w], None)
        if rp is not None and rs is not None:
            psnr_sum += float(rp)
            ssim_sum += float(rs)
            count += 1
    return psnr_sum, ssim_sum, count


@torch.inference_mode()
def evaluate(
    model, loader, device, track_recon: bool = False,
) -> Tuple[float, dict, Optional[dict], int]:
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    recon_acc = {"psnr": 0.0, "ssim": 0.0, "n": 0.0}
    n_images = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        labels = batch.get("semantic")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("zero-shot semantic labels must be tensors")
        y = labels.cpu().numpy()
        if y.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"zero-shot image/label row mismatch: {x.shape[0]} vs {y.shape[0]}")
        valid_hw = batch.get(VALID_HW_KEY)
        if valid_hw is None:
            valid_hw_np = np.tile(
                np.asarray(y.shape[-2:], dtype=np.int64), (x.shape[0], 1))
        else:
            if not isinstance(valid_hw, torch.Tensor):
                valid_hw = torch.as_tensor(valid_hw)
            valid_hw_np = valid_hw.cpu().numpy()
            if valid_hw_np.shape != (x.shape[0], 2):
                raise RuntimeError(
                    "zero-shot valid-H/W provenance must have shape "
                    f"({x.shape[0]}, 2), got {valid_hw_np.shape}")
        unwrapped = model.module if hasattr(model, "module") else model
        if (
            track_recon
            and "clear_image" in batch
            and getattr(unwrapped, "use_decoder", False)
        ):
            out = unwrapped.predict(x, return_recon=True)
            if isinstance(out, tuple) and len(out) == 2:
                out, recon = out
            else:
                out = out[0] if isinstance(out, (tuple, list)) else out
                recon = None
            if recon is not None:
                clear_img = batch["clear_image"].to(device, non_blocking=True)
                mask = batch.get("has_paired_clear")
                if mask is None:
                    raise RuntimeError(
                        "zero-shot clear_image is missing has_paired_clear provenance")
                psnr_sum, ssim_sum, recon_count = _recon_metrics_at_valid_extents(
                    recon, clear_img, mask, valid_hw_np)
                recon_acc["psnr"] += psnr_sum
                recon_acc["ssim"] += ssim_sum
                recon_acc["n"] += float(recon_count)
        else:
            out = model(x) if not hasattr(model, "predict") else model.predict(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[-2:] != y.shape[-2:]:
            out = F.interpolate(out, size=y.shape[-2:], mode="bilinear",
                                align_corners=False)
        pred = out.argmax(1).cpu().numpy()
        for i in range(pred.shape[0]):
            h, w = (int(valid_hw_np[i, 0]), int(valid_hw_np[i, 1]))
            if not 0 < h <= y.shape[-2] or not 0 < w <= y.shape[-1]:
                raise RuntimeError(
                    f"invalid zero-shot extent {(h, w)} for padded label shape "
                    f"{y.shape[-2:]}")
            yi = y[i, :h, :w]
            pi = pred[i, :h, :w]
            k = (yi >= 0) & (yi < NUM_CLASSES) & (yi != IGNORE_LABEL)
            hist += np.bincount(
                NUM_CLASSES * yi[k].astype(np.int64) + pi[k].astype(np.int64),
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            n_images += 1
    if n_images < 1:
        raise RuntimeError("zero-shot evaluation yielded zero images")
    if hist.sum() < 1:
        raise RuntimeError(
            "zero-shot evaluation yielded no valid labelled pixels; check label "
            "encoding and ignore-index handling")
    iou = per_class_iou(hist)
    miou = mean_iou(hist) * 100
    per_class = {CLASSES[i]: (None if v is None else v * 100)
                 for i, v in enumerate(jsonable_iou(iou))}
    extra = None
    if recon_acc["n"] > 0:
        n = max(1.0, recon_acc["n"])
        extra = {"psnr": recon_acc["psnr"] / n, "ssim": recon_acc["ssim"] / n}
    return miou, per_class, extra, n_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Explicit .pth path (overrides --experiment_dir)")
    p.add_argument("--experiment_dir", default=None,
                   help="e.g. output/cityscapes/bisenetv2_guided — auto-pick the highest final-stage checkpoint")
    p.add_argument("--backbone", required=True,
                   help="model_hub key, e.g. bisenetv2 / stdc / ddrnet / pidnet_s")
    p.add_argument("--variant", choices=["baseline", "guided", "norecon"],
                   default="guided")
    p.add_argument("--datasets", nargs="+",
                   choices=["acdc", "foggy_driving", "foggy_zurich"],
                   default=["acdc", "foggy_driving", "foggy_zurich"])
    p.add_argument("--conditions", nargs="+",
                   default=["fog", "night", "rain", "snow"],
                   help="ACDC conditions when --datasets includes acdc")
    p.add_argument("--acdc_root", default=None,
                   help="defaults to ACDC_ROOT from configs/paths.env")
    p.add_argument("--foggy_driving_root", default=None,
                   help="defaults to FOGGY_DRIVING_ROOT from configs/paths.env")
    p.add_argument("--foggy_zurich_root", default=None,
                   help="defaults to FOGGY_ZURICH_ROOT from configs/paths.env")
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_json", default=None)
    p.add_argument(
        "--crop_eval", action="store_true",
        help="Resize evaluation images to --crop_h x --crop_w instead of the "
             "default native-resolution protocol.",
    )
    p.add_argument(
        "--track_recon",
        action="store_true",
        help="If batches include clear_image and the model has use_decoder, "
             "log PSNR/SSIM (off by default; zero-shot sets usually lack pairs).",
    )
    args = p.parse_args()
    if not args.checkpoint and not args.experiment_dir:
        p.error("Provide --checkpoint or --experiment_dir")
    for key, env in (("acdc_root", "ACDC_ROOT"),
                     ("foggy_driving_root", "FOGGY_DRIVING_ROOT"),
                     ("foggy_zurich_root", "FOGGY_ZURICH_ROOT")):
        if getattr(args, key) is None:
            setattr(args, key, paths.paths[env])
    return args


def main():
    args = parse_args()
    paths.assert_exists()
    invalid_conditions = sorted(set(args.conditions).difference(ACDC_CONDITIONS))
    if invalid_conditions:
        raise ValueError(
            f"unknown ACDC conditions {invalid_conditions}; expected "
            f"{list(ACDC_CONDITIONS)}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt_path = str(Path(args.checkpoint).resolve())
    else:
        ckpt_path = str(resolve_experiment_checkpoint(Path(args.experiment_dir)))
    print(f"backbone={args.backbone}  variant={args.variant}  ckpt={ckpt_path}")
    model, architecture, _ = build_model_from_checkpoint(
        ckpt_path, expected_backbone=args.backbone,
        expected_variant=args.variant, expected_n_classes=NUM_CLASSES)
    if args.track_recon and not architecture["use_decoder"]:
        raise ValueError(
            "--track_recon requires a checkpoint whose architecture has "
            "use_decoder=True")
    model = model.to(device).eval()

    out = {"checkpoint": ckpt_path, "backbone": args.backbone,
           "variant": args.variant,
           "architecture": architecture_for_report(architecture),
           "resolution_protocol": (
               "deterministic_resized_crop" if args.crop_eval else "native"),
           "datasets": {}}

    if "acdc" in args.datasets:
        out["datasets"]["acdc"] = {}
        for c in args.conditions:
            ds = build_acdc(args.acdc_root, [c], (args.crop_h, args.crop_w),
                            is_train=False, load_clear=False,
                            eval_full_res=not args.crop_eval)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=collate)
            miou, per_class, recon, n_images = evaluate(
                model, loader, device, track_recon=args.track_recon)
            msg = f"  ACDC/{c:5s}    mIoU = {miou:6.2f}  (n={n_images})"
            if recon:
                msg += f"  PSNR={recon['psnr']:.2f}  SSIM={recon['ssim']:.4f}"
            print(msg)
            row = {"mIoU": miou, "per_class": per_class, "n": n_images}
            if recon:
                row["psnr"] = recon["psnr"]
                row["ssim"] = recon["ssim"]
            out["datasets"]["acdc"][c] = row

    if "foggy_driving" in args.datasets:
        pairs = _list_foggy_driving(args.foggy_driving_root)
        if not pairs:
            raise FileNotFoundError(
                "Foggy Driving has no labelled image/label pairs; refusing to "
                "silently skip a requested zero-shot dataset")
        ds = _ImgLabelDataset(
            pairs, args.crop_h, args.crop_w, full_res=not args.crop_eval)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)
        miou, per_class, recon, n_images = evaluate(
            model, loader, device, track_recon=args.track_recon)
        msg = f"  Foggy Driving  mIoU = {miou:6.2f}  (n={n_images})"
        if recon:
            msg += f"  PSNR={recon['psnr']:.2f}  SSIM={recon['ssim']:.4f}"
        print(msg)
        row = {"mIoU": miou, "per_class": per_class, "n": n_images}
        if recon:
            row["psnr"] = recon["psnr"]
            row["ssim"] = recon["ssim"]
        out["datasets"]["foggy_driving"] = row

    if "foggy_zurich" in args.datasets:
        pairs = _list_foggy_zurich(args.foggy_zurich_root)
        if not pairs:
            raise FileNotFoundError(
                "Foggy Zurich has no labelled image/label pairs; refusing to "
                "silently skip a requested zero-shot dataset")
        ds = _ImgLabelDataset(
            pairs, args.crop_h, args.crop_w, full_res=not args.crop_eval)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)
        miou, per_class, recon, n_images = evaluate(
            model, loader, device, track_recon=args.track_recon)
        msg = f"  Foggy Zurich   mIoU = {miou:6.2f}  (n={n_images})"
        if recon:
            msg += f"  PSNR={recon['psnr']:.2f}  SSIM={recon['ssim']:.4f}"
        print(msg)
        row = {"mIoU": miou, "per_class": per_class, "n": n_images}
        if recon:
            row["psnr"] = recon["psnr"]
            row["ssim"] = recon["ssim"]
        out["datasets"]["foggy_zurich"] = row

    if not out["datasets"]:
        raise RuntimeError("zero-shot evaluation completed without any requested result")

    out_json = args.output_json or str(
        Path(ckpt_path).with_name("zero_shot.json"))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))
    print("-> wrote", out_json)


if __name__ == "__main__":
    main()
