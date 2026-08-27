"""Guidance encoder and reconstruction decoder."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "DEHAZE_CHANNELS", "ChannelLayerNorm", "ConvStem", "Downsample",
    "EncoderBlock", "GatedDConvFFN", "GatedLinearAttention",
    "LinearDehazeFormerEncoder", "MDTA", "PixelShuffleUp",
    "ProgressiveUpsampleHead", "ReconstructionDecoder", "RestormerDecoderBlock",
]

DEHAZE_CHANNELS = (32, 64, 128)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the NCHW channel axis."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)



class GatedLinearAttention(nn.Module):
    """Linear attention with a SiLU-gated projection."""

    def __init__(self, dim: int, heads: int = 4, bias: bool = False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.qkvg_proj = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.pre_norm = ChannelLayerNorm(dim)
        self.pre_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.pre_dw(self.pre_norm(x))
        q, k, v, g = self.qkvg_proj(x).chunk(4, dim=1)

        q = q.reshape(b, self.heads, self.head_dim, -1)
        k = k.reshape(b, self.heads, self.head_dim, -1)
        v = v.reshape(b, self.heads, self.head_dim, -1)
        g = g.reshape(b, self.heads, self.head_dim, -1)

        g = F.silu(g)                       # SiLU gate
        k = F.softmax(k, dim=-1)            # softmax over the token axis

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = (out * g).reshape(b, c, h, w)  # gated aggregation
        return residual + self.project_out(out)



class GatedDConvFFN(nn.Module):
    """Depth-wise gated feed-forward network."""

    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.expand = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                padding=1, groups=hidden * 2, bias=bias)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dwconv(self.expand(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project(F.gelu(x1) * x2)



class EncoderBlock(nn.Module):
    """One guidance-encoder block: GLA then GDFN, each residual."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.66):
        super().__init__()
        self.attn = GatedLinearAttention(dim, heads=heads)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = GatedDConvFFN(dim, expansion=mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x)
        return x + self.ffn(self.norm2(x))



class ConvStem(nn.Module):
    """Convolutional stem, full resolution to 1/8, three stride-2 convolutions."""

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.stage1 = block(in_channels, 32)   # 1/1 -> 1/2
        self.stage2 = block(32, 48)            # 1/2 -> 1/4
        self.stage3 = block(48, out_channels)  # 1/4 -> 1/8

    def forward(self, x: Tensor) -> Tensor:
        return self.stage3(self.stage2(self.stage1(x)))


class Downsample(nn.Module):
    """Stride-2 projection between encoder stages."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2,
                              padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.proj(x)))



class MDTA(nn.Module):
    """Channel-axis transposed attention."""

    def __init__(self, dim: int, heads: int = 4, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.heads, self.head_dim, -1)
        k = k.reshape(b, self.heads, self.head_dim, -1)
        v = v.reshape(b, self.heads, self.head_dim, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class RestormerDecoderBlock(nn.Module):
    """Decoder refinement block: MDTA then GDFN."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.66):
        super().__init__()
        self.norm1 = ChannelLayerNorm(dim)
        self.attn = MDTA(dim, heads=heads)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = GatedDConvFFN(dim, expansion=mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class PixelShuffleUp(nn.Module):
    """2x sub-pixel upsampling."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class ProgressiveUpsampleHead(nn.Module):
    """Reconstruction head, 1/8 to full resolution with a global residual."""

    def __init__(self, in_ch: int = 32, out_ch: int = 3,
                 num_refinement_blocks: int = 4, heads: int = 4):
        super().__init__()
        self.refinement = nn.Sequential(*[
            RestormerDecoderBlock(in_ch, heads=heads)
            for _ in range(num_refinement_blocks)
        ])

        def up(last: bool = False) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_ch, in_ch * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(2),
            ]
            if not last:
                layers.append(nn.GELU())
            return nn.Sequential(*layers)

        self.up1 = up()            # 1/8 -> 1/4
        self.up2 = up()            # 1/4 -> 1/2
        self.up3 = up(last=True)   # 1/2 -> 1/1
        self.output = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor, inp_img: Tensor) -> Tensor:
        x = self.refinement(x)
        x = self.up3(self.up2(self.up1(x)))
        delta = self.output(x)
        if delta.shape[-2:] != inp_img.shape[-2:]:
            delta = F.interpolate(delta, size=inp_img.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return inp_img + delta


class _Stage(nn.Module):
    """A run of ``depth`` encoder blocks at one resolution."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads=heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class LinearDehazeFormerEncoder(nn.Module):
    """Guidance encoder returning s8, s16, and s32 features."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        depths: Optional[Sequence[int]] = (4, 6, 6),
        num_heads: Sequence[int] = (4, 8, 16),
        ffn_expansion: float = 2.66,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        depths = tuple(depths if depths is not None else (4, 6, 6))
        channels = tuple(channels)
        num_heads = tuple(num_heads)
        if not (len(channels) == len(depths) == len(num_heads) == 3):
            raise ValueError("channels, depths and num_heads must each have length 3")

        self.channels = channels
        self.depths = depths
        self.use_checkpoint = use_checkpoint

        self.stem = ConvStem(in_channels, channels[0])          # -> 1/8
        self.stage1 = _Stage(channels[0], depths[0], num_heads[0], ffn_expansion)
        self.down1 = Downsample(channels[0], channels[1])       # -> 1/16
        self.stage2 = _Stage(channels[1], depths[1], num_heads[1], ffn_expansion)
        self.down2 = Downsample(channels[1], channels[2])       # -> 1/32
        self.stage3 = _Stage(channels[2], depths[2], num_heads[2], ffn_expansion)

    def _run(self, stage: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        s8 = self._run(self.stage1, self.stem(x))
        s16 = self._run(self.stage2, self.down1(s8))
        s32 = self._run(self.stage3, self.down2(s16))
        return {"s8": s8, "s16": s16, "s32": s32}


class ReconstructionDecoder(nn.Module):
    """Training-only decoder over the guidance pyramid."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = (32, 64, 128),
        depths: Sequence[int] = (2, 2),
        heads: Sequence[int] = (8, 4),
        out_channels: int = 3,
        ffn_expansion: float = 2.66,
    ):
        super().__init__()
        c8, c16, c32 = tuple(encoder_channels)

        self.up_s32 = PixelShuffleUp(c32)                 # c32 -> c32/2 at 1/16
        self.reduce_s16 = nn.Conv2d(c32 // 2 + c16, c16, kernel_size=1, bias=False)
        self.refine_s16 = nn.Sequential(*[
            RestormerDecoderBlock(c16, heads=heads[0], mlp_ratio=ffn_expansion)
            for _ in range(depths[0])
        ])

        self.up_s16 = PixelShuffleUp(c16)                 # c16 -> c16/2 at 1/8
        self.reduce_s8 = nn.Conv2d(c16 // 2 + c8, c8, kernel_size=1, bias=False)
        self.refine_s8 = nn.Sequential(*[
            RestormerDecoderBlock(c8, heads=heads[1], mlp_ratio=ffn_expansion)
            for _ in range(depths[1])
        ])

        self.head = ProgressiveUpsampleHead(
            in_ch=c8, out_ch=out_channels,
            num_refinement_blocks=4, heads=heads[1],
        )

    def forward(self, features: Dict[str, Tensor], x_in: Tensor) -> Dict[str, Tensor]:
        """Reconstruct the clear image from the guidance pyramid and degraded input."""
        d = self.up_s32(features["s32"])
        d = self.reduce_s16(_cat(d, features["s16"]))
        d = self.refine_s16(d)

        d = self.up_s16(d)
        d = self.reduce_s8(_cat(d, features["s8"]))
        d = self.refine_s8(d)

        return {"recon": self.head(d, x_in)}


def _cat(decoder_feature: Tensor, skip: Tensor) -> Tensor:
    """Concatenate a decoder feature with its skip, aligning odd sizes first."""
    if decoder_feature.shape[-2:] != skip.shape[-2:]:
        decoder_feature = F.interpolate(
            decoder_feature, size=skip.shape[-2:],
            mode="bilinear", align_corners=False)
    return torch.cat([decoder_feature, skip], dim=1)
