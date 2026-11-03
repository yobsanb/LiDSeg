"""Per-condition mIoU validation with DDP histogram."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from train.core.metrics import fast_hist, mean_iou
from train.core.recon_metrics import recon_psnr_ssim_batch
from train.core.utils import _unwrap, is_ddp
from dataset.loader import EVAL_PADDING_KEY, VALID_HW_KEY


def _recon_metrics_at_valid_extents(
    recon: torch.Tensor,
    clear: torch.Tensor,
    paired_mask: torch.Tensor,
    valid_hw: np.ndarray,
) -> tuple[float, float, int]:
    """Score paired rows over their valid (unpadded) extents."""
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    for i in range(int(recon.shape[0])):
        if not bool(paired_mask[i]):
            continue
        h, w = (int(valid_hw[i, 0]), int(valid_hw[i, 1]))
        if h < 1 or w < 1:
            raise RuntimeError(f"invalid reconstruction metric extent {(h, w)}")
        rp, rs = recon_psnr_ssim_batch(
            recon[i:i + 1, ..., :h, :w], clear[i:i + 1, ..., :h, :w], None)
        if rp is not None and rs is not None:
            psnr_sum += float(rp)
            ssim_sum += float(rs)
            count += 1
    return psnr_sum, ssim_sum, count


@torch.no_grad()
def validate(model, val_loader, n_classes: int, device) -> Dict[str, float]:
    """Per-condition native-resolution mIoU with exact DDP sample accounting."""
    was_training = model.training
    model.eval()
    try:
        per_cond: Dict[str, np.ndarray] = {}
        recon_stats: Dict[str, float] = {
            "psnr": 0.0, "ssim": 0.0, "count": 0.0,
        }
        local_images = 0
        ddp = is_ddp()
        if ddp:
            # BatchNorm running stats are rank-local between DDP forwards; score
            # exactly the rank-0 model that the checkpoint stores.
            for buf in _unwrap(model).buffers():
                dist.broadcast(buf, src=0)

        for batch in val_loader:
            img = batch["image"].to(device, non_blocking=True)
            target_t = batch["semantic"]
            if not isinstance(target_t, torch.Tensor):
                raise TypeError("validation semantic labels must be tensors")
            tgt = target_t.cpu().numpy()
            n_rows = int(img.shape[0])
            if tgt.shape[0] != n_rows:
                raise RuntimeError(
                    f"validation image/label row mismatch: {n_rows} vs "
                    f"{tgt.shape[0]}")

            padding = batch.get(EVAL_PADDING_KEY)
            if ddp:
                if padding is None:
                    raise RuntimeError(
                        "DDP validation loader is missing explicit padding "
                        "provenance; construct it with dataset.loader.make_loader")
                if not isinstance(padding, torch.Tensor):
                    padding = torch.as_tensor(padding)
                padding = padding.bool().reshape(-1)
                if padding.numel() != n_rows:
                    raise RuntimeError(
                        "DDP validation padding provenance has the wrong row "
                        f"count: {padding.numel()} vs {n_rows}")
                valid_rows = ~padding.cpu().numpy()
            else:
                valid_rows = np.ones(n_rows, dtype=bool)

            valid_hw = batch.get(VALID_HW_KEY)
            if valid_hw is None:
                valid_hw_np = np.tile(
                    np.asarray(tgt.shape[-2:], dtype=np.int64), (n_rows, 1))
            else:
                if not isinstance(valid_hw, torch.Tensor):
                    valid_hw = torch.as_tensor(valid_hw)
                valid_hw_np = valid_hw.cpu().numpy()
                if valid_hw_np.shape != (n_rows, 2):
                    raise RuntimeError(
                        "validation valid-H/W provenance must have shape "
                        f"({n_rows}, 2), got {valid_hw_np.shape}")

            has_clear = "clear_image" in batch
            unwrapped = _unwrap(model)
            if hasattr(unwrapped, "use_decoder") and unwrapped.use_decoder and has_clear:
                logits, recon = unwrapped.predict(img, return_recon=True)
                clear_img = batch["clear_image"].to(device, non_blocking=True)
                mask = batch.get("has_paired_clear")
                if mask is None:
                    raise RuntimeError(
                        "validation clear_image is missing has_paired_clear "
                        "provenance")
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask)
                metric_mask = mask.to(device=device).bool().reshape(-1)
                metric_mask &= torch.as_tensor(valid_rows, device=device)
                psnr_sum, ssim_sum, recon_count = _recon_metrics_at_valid_extents(
                    recon, clear_img, metric_mask, valid_hw_np)
                recon_stats["psnr"] += psnr_sum
                recon_stats["ssim"] += ssim_sum
                recon_stats["count"] += float(recon_count)
            else:
                logits = unwrapped.predict(img)

            if logits.shape[-2:] != tgt.shape[-2:]:
                logits = F.interpolate(logits, size=tgt.shape[-2:],
                                       mode="bilinear", align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
            conds = batch.get("condition", ["all"] * n_rows)
            if isinstance(conds, torch.Tensor):
                conds = conds.tolist()
            if len(conds) != n_rows:
                raise RuntimeError(
                    f"validation condition metadata has {len(conds)} rows, "
                    f"expected {n_rows}")
            for i, c in enumerate(conds):
                if not valid_rows[i]:
                    continue
                h, w = (int(valid_hw_np[i, 0]), int(valid_hw_np[i, 1]))
                if not 0 < h <= tgt.shape[-2] or not 0 < w <= tgt.shape[-1]:
                    raise RuntimeError(
                        f"invalid validation extent {(h, w)} for padded label "
                        f"shape {tgt.shape[-2:]}")
                per_cond.setdefault(c, np.zeros((n_classes, n_classes), np.int64))
                per_cond[c] += fast_hist(
                    tgt[i, :h, :w].flatten(), pred[i, :h, :w].flatten(),
                    n_classes)
                local_images += 1

        if ddp:
            local_keys = sorted(per_cond.keys())
            obj_list = [None] * dist.get_world_size()
            dist.all_gather_object(obj_list, local_keys)
            all_keys = sorted({k for sub in obj_list for k in sub})
            for k in all_keys:
                per_cond.setdefault(k, np.zeros((n_classes, n_classes), np.int64))
            if all_keys:
                stacked = torch.from_numpy(
                    np.stack([per_cond[k] for k in all_keys])).to(device)
                dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                for i, k in enumerate(all_keys):
                    per_cond[k] = stacked[i].cpu().numpy()
            image_count = torch.tensor([local_images], device=device,
                                       dtype=torch.int64)
            dist.all_reduce(image_count, op=dist.ReduceOp.SUM)
            total_images = int(image_count.item())
        else:
            total_images = local_images

        if total_images < 1:
            raise RuntimeError(
                "validation produced zero real images; check dataset roots, "
                "condition filters, and DDP sampler provenance")

        results: Dict[str, float] = {}
        agg = np.zeros((n_classes, n_classes), np.int64)
        for c, h in sorted(per_cond.items()):
            if h.sum() > 0:
                results[f"miou/{c}"] = float(mean_iou(h) * 100.0)
            agg += h
        if agg.sum() < 1:
            raise RuntimeError(
                "validation produced no valid labelled pixels; labels may be "
                "missing, all-ignore, or incorrectly encoded")
        results["miou/all"] = float(mean_iou(agg) * 100.0)
        results["pixel_acc"] = float(np.diag(agg).sum() / agg.sum() * 100.0)

        if ddp:
            t_stats = torch.tensor(
                [recon_stats["psnr"], recon_stats["ssim"], recon_stats["count"]],
                device=device, dtype=torch.float64)
            dist.all_reduce(t_stats, op=dist.ReduceOp.SUM)
            recon_stats["psnr"], recon_stats["ssim"], recon_stats["count"] = t_stats.tolist()

        if recon_stats["count"] > 0:
            n = recon_stats["count"]
            results["psnr"] = float(recon_stats["psnr"] / n)
            results["ssim"] = float(recon_stats["ssim"] / n)
        results["num_images"] = float(total_images)
        return results
    finally:
        if was_training:
            model.train()
