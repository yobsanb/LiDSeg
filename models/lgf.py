"""Lightweight Gated Fusion: channel-gated injection of guidance into the backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LightweightGatedFusion", "build_lgf"]


def _align(g: Tensor, f: Tensor) -> Tensor:
    """Resample guidance to the segmentation feature's spatial size."""
    if g.shape[-2:] == f.shape[-2:]:
        return g
    return F.interpolate(g, size=f.shape[-2:], mode="bilinear", align_corners=False)


class _ChannelGate(nn.Module):
    """Gated fusion bottleneck."""

    def __init__(self, f_channels: int, g_channels: int, reduction: int = 8):
        super().__init__()
        in_dim = f_channels + g_channels
        hidden = max(in_dim // reduction, 8)
        self.fc1 = nn.Conv2d(in_dim, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, g_channels, 1, bias=True)

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        z = torch.cat([f, g], dim=1).mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=False)))


class LightweightGatedFusion(nn.Module):
    """Channel-gated residual injection of guidance into one backbone tap."""

    def __init__(self, f_channels: int, g_channels: int | None = None,
                 reduction: int = 8, lambda_init: float = 0.3):
        super().__init__()
        g_channels = f_channels if g_channels is None else g_channels
        self.proj_g = (nn.Identity() if g_channels == f_channels
                       else nn.Conv2d(g_channels, f_channels, 1, bias=False))
        self.gate = _ChannelGate(f_channels, f_channels, reduction=reduction)
        self.lambda_k = nn.Parameter(torch.full((1,), lambda_init))

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        g = self.proj_g(_align(g, f))
        return f + self.lambda_k * self.gate(f, g) * g


def build_lgf(f_channels: int, g_channels: int, reduction: int = 8,
              lambda_init: float = 0.3) -> LightweightGatedFusion:
    """Build the fusion module for one stage."""
    return LightweightGatedFusion(f_channels=f_channels, g_channels=g_channels,
                                  reduction=reduction, lambda_init=lambda_init)
