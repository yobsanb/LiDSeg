"""PSNR and SSIM for the reconstruction branch during validation."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from dataset.segdatasets.cityscapes import (
    CITYSCAPES_IMG_NORM_MEAN,
    CITYSCAPES_IMG_NORM_STD,
)
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as _tm_psnr,
    structural_similarity_index_measure as _tm_ssim,
)

NormTuple = Union[Tuple[float, float, float], Sequence[float]]


def _denorm_normalize_to_rgb01(
    t: torch.Tensor,
    mean: NormTuple,
    std: NormTuple,
) -> torch.Tensor:
    """Invert ``Normalize(mean, std)`` on [0,1] RGB tensors → ~[0, 1] RGB."""
    mean_t = t.new_tensor(mean).view(1, 3, 1, 1)
    std_t = t.new_tensor(std).view(1, 3, 1, 1)
    return (t * std_t + mean_t).clamp(0.0, 1.0)


def recon_psnr_ssim_batch(
    recon: torch.Tensor,
    clear: torch.Tensor,
    mask_b: Optional[torch.Tensor],
    *,
    data_range: float = 1.0,
    rgb01_metrics: bool = True,
    norm_mean: Optional[NormTuple] = None,
    norm_std: Optional[NormTuple] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Per-batch PSNR / SSIM on paired rows (optional ``has_paired_clear`` mask)."""
    if recon is None or clear is None:
        return None, None
    if mask_b is not None:
        m = mask_b.to(device=recon.device).bool().reshape(-1)
        if int(m.sum().item()) == 0:
            return None, None
        recon = recon[m]
        clear = clear[m]
    if recon.numel() == 0:
        return None, None
    if recon.shape[-2:] != clear.shape[-2:]:
        recon = F.interpolate(
            recon,
            size=clear.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    with torch.no_grad():
        recon = recon.detach().float()
        clear = clear.detach().float()
        if rgb01_metrics:
            m = norm_mean if norm_mean is not None else CITYSCAPES_IMG_NORM_MEAN
            s = norm_std if norm_std is not None else CITYSCAPES_IMG_NORM_STD
            recon = _denorm_normalize_to_rgb01(recon, m, s)
            clear = _denorm_normalize_to_rgb01(clear, m, s)
        p = float(_tm_psnr(recon, clear, data_range=data_range))
        s = float(_tm_ssim(recon, clear, data_range=data_range))
        if math.isinf(p):
            p = 99.0
    return p, s
