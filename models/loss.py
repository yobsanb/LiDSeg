"""LiDSeg losses: Lovasz-Softmax, OHEM cross-entropy, Charbonnier, uncertainty weighting."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LiDSegLoss", "ProbOhemCrossEntropy2d", "charbonnier", "lovasz_softmax"]

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten(probas: torch.Tensor, labels: torch.Tensor, ignore: int):
    if probas.dim() == 4:
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probas.numel() == 0:
        return probas.sum() * 0.0
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore: int = 255,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax over [B, C, H, W] probabilities."""
    flat_p, flat_l = _flatten(probas, labels, ignore)
    return _lovasz_softmax_flat(flat_p, flat_l)



_EPS = 1e-3


def charbonnier(x: Tensor, y: Tensor, eps: float = _EPS) -> Tensor:
    """Charbonnier (smooth L1) distance."""
    return torch.sqrt((x - y).pow(2) + eps * eps).mean()


def _safe_logits(logits: Tensor) -> Tensor:
    """Cast to fp32 and clamp before softmax."""
    return logits.float().clamp(-50.0, 50.0)


class ProbOhemCrossEntropy2d(nn.Module):
    """Online hard-example-mining cross-entropy."""

    def __init__(self, ignore_label: int = 255, thresh: float = 0.7,
                 min_kept: int = 100_000,
                 weight: Optional[Tensor] = None):
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label,
                                      reduction="none")

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = _safe_logits(logits)
        pixel_losses = self.ce(logits, labels).view(-1)

        with torch.no_grad():
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            max_probs = log_probs.max(dim=1).values.exp().view(-1)
            valid = (labels != self.ignore_label).view(-1).nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                return pixel_losses.sum() * 0.0
            hard = (max_probs[valid] < self.thresh)
            if int(hard.sum()) < self.min_kept:
                _, order = pixel_losses.detach()[valid].sort(descending=True)
                keep = order[: min(self.min_kept, valid.numel())]
            else:
                keep = hard.nonzero(as_tuple=True)[0]

        return pixel_losses[valid][keep].mean()


class LiDSegLoss(nn.Module):
    """Segmentation plus auxiliary reconstruction, uncertainty-weighted."""

    def __init__(
        self,
        n_classes: int = 19,
        ignore_index: int = 255,
        aux_weight: float = 0.4,
        seg_lovasz_weight: float = 0.5,
        use_ohem: bool = False,
        ohem_thresh: float = 0.7,
        ohem_min_kept: int = 100_000,
        loss_weighting: str = "uncertainty",
        lambda_segmentation: float = 1.0,
        lambda_reconstruction: float = 1.0,
        charbonnier_eps: float = _EPS,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.aux_weight = float(aux_weight)
        self.seg_lovasz_weight = float(seg_lovasz_weight)
        self.charbonnier_eps = float(charbonnier_eps)

        weighting = str(loss_weighting or "uncertainty").lower()
        if weighting not in ("uncertainty", "fixed"):
            raise ValueError("loss_weighting must be 'uncertainty' or 'fixed'")
        self.loss_weighting = weighting
        self.lambda_segmentation = float(lambda_segmentation)
        self.lambda_reconstruction = float(lambda_reconstruction)

        if use_ohem:
            self.ce = ProbOhemCrossEntropy2d(ignore_label=ignore_index,
                                             thresh=ohem_thresh,
                                             min_kept=ohem_min_kept)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.aux_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_rec = nn.Parameter(torch.zeros(1))


    def _weighted(self, loss: Tensor, log_var: nn.Parameter,
                  task_lambda: float) -> Tensor:
        """One uncertainty-weighted term: ``loss / sigma^2 + log sigma^2``."""
        if self.loss_weighting == "fixed":
            return task_lambda * loss
        return task_lambda * loss * torch.exp(-log_var).squeeze() + log_var.squeeze()

    def _segmentation(self, logits: Tensor, target: Tensor,
                      aux_logits: Optional[List[Tensor]]) -> Tensor:
        if not bool((target != self.ignore_index).any()):
            return logits.sum() * 0.0

        loss = self.ce(_safe_logits(logits), target)

        if self.seg_lovasz_weight > 0:
            probs = F.softmax(_safe_logits(logits), dim=1)
            loss = loss + self.seg_lovasz_weight * lovasz_softmax(
                probs, target, ignore=self.ignore_index)

        if aux_logits:
            n_cls = logits.shape[1]
            for aux in aux_logits:
                if aux.dim() == 4 and aux.shape[1] == n_cls:
                    loss = loss + self.aux_weight * self.aux_ce(
                        _safe_logits(aux), target)
        return loss

    def _reconstruction(self, recon: Tensor, clear: Tensor,
                        has_paired_clear: Optional[Tensor]) -> Optional[Tensor]:
        """Charbonnier over the paired-clear subset."""
        if recon.shape[-2:] != clear.shape[-2:]:
            recon = F.interpolate(recon, size=clear.shape[-2:],
                                  mode="bilinear", align_corners=False)
        recon = recon.float()
        clear = clear.float()

        if has_paired_clear is None:
            return charbonnier(recon, clear, self.charbonnier_eps)

        mask = has_paired_clear.to(torch.bool).view(-1)
        if mask.numel() != recon.shape[0] or not bool(mask.any()):
            return None
        return charbonnier(recon[mask], clear[mask], self.charbonnier_eps)


    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        aux_logits: Optional[List[Tensor]] = None,
        recon: Optional[Tensor] = None,
        clear: Optional[Tensor] = None,
        has_paired_clear: Optional[Tensor] = None,
        active_recon: bool = True,
    ) -> Dict[str, Tensor]:
        """Return {"total", "l_seg", "l_rec"}."""
        l_seg = self._segmentation(logits, target, aux_logits)
        total = self._weighted(l_seg, self.log_var_seg, self.lambda_segmentation)

        l_rec: Optional[Tensor] = None
        if active_recon and recon is not None and clear is not None:
            l_rec = self._reconstruction(recon, clear, has_paired_clear)
            if l_rec is not None:
                total = total + self._weighted(
                    l_rec, self.log_var_rec, self.lambda_reconstruction)

        return {
            "total": total,
            "l_seg": l_seg.detach(),
            "l_rec": (l_rec.detach() if l_rec is not None
                      else torch.zeros((), device=logits.device)),
        }
