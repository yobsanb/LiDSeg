# Ported from https://github.com/XuJiacong/PIDNet
# Copyright (c) 2022 Jiacong Xu. MIT License.
# Checkpoint layout follows https://github.com/open-mmlab/mmsegmentation
# Copyright 2020 The MMSegmentation Authors. Apache License 2.0.
"""PIDNet (Xu et al., CVPR 2023).
Dependency-free port of the mmsegmentation implementation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model_registry import aux_models, register_model

__all__ = ["PIDNetS", "PIDNetM", "PIDNetL"]



class ConvBNReLU(nn.Module):
    """``Conv2d -> BatchNorm2d -> ReLU`` (mmseg ``ConvModule`` default order)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, *,
                 act: bool = True, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self._act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        if self._act:
            x = F.relu(x, inplace=False)
        return x


class PreNormConvBNReLU(nn.Module):
    """BatchNorm2d -> ReLU -> Conv2d, the mmseg pre-norm ConvModule order."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1,
                 stride: int = 1, padding: int = 0, *,
                 groups: int = 1, act: bool = True, bias: bool = False):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=bias)
        self._act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        if self._act:
            x = F.relu(x, inplace=False)
        return self.conv(x)



class BasicBlock(nn.Module):
    """ResNet basic block. ``act_out`` controls the final ReLU."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 act_out: bool = True):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, 3, stride, 1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3, 1, 1, act=False)
        self.downsample = downsample
        self._act_out = act_out

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x)) + identity
        if self._act_out:
            out = F.relu(out, inplace=False)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck with expansion=2 (PIDNet/mmseg convention)."""

    expansion = 2

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 act_out: bool = False):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3, stride, 1)
        self.conv3 = ConvBNReLU(out_ch, out_ch * self.expansion, 1, 1, 0,
                                act=False)
        self.downsample = downsample
        self._act_out = act_out

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x))) + identity
        if self._act_out:
            out = F.relu(out, inplace=False)
        return out



class PagFM(nn.Module):
    """f_p, f_i are 1x1 ConvBN (no act). Gate via dot-product sigmoid."""

    def __init__(self, in_ch: int, mid_ch: int, after_relu: bool = False):
        super().__init__()
        self.f_i = ConvBNReLU(in_ch, mid_ch, 1, 1, 0, act=False)
        self.f_p = ConvBNReLU(in_ch, mid_ch, 1, 1, 0, act=False)
        self.after_relu = after_relu

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        if self.after_relu:
            x_p = F.relu(x_p, inplace=False)
            x_i = F.relu(x_i, inplace=False)
        f_i = self.f_i(x_i)
        f_i = F.interpolate(f_i, size=x_p.shape[-2:],
                            mode="bilinear", align_corners=False)
        f_p = self.f_p(x_p)
        sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1, keepdim=True))
        x_i_up = F.interpolate(x_i, size=x_p.shape[-2:],
                               mode="bilinear", align_corners=False)
        return sigma * x_i_up + (1.0 - sigma) * x_p



class LightBag(nn.Module):
    """f_p / f_i are 1x1 ConvBN (no act). Used by PIDNet-S/M."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.f_p = ConvBNReLU(in_ch, out_ch, 1, 1, 0, act=False)
        self.f_i = ConvBNReLU(in_ch, out_ch, 1, 1, 0, act=False)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        sigma = torch.sigmoid(x_d)
        f_p = self.f_p((1.0 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)
        return f_p + f_i


class Bag(nn.Module):
    """3x3 ConvBN+ReLU with mmseg ``order=('norm','act','conv')``. PIDNet-L."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = PreNormConvBNReLU(in_ch, out_ch, 3, 1, 1, act=True)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        sigma = torch.sigmoid(x_d)
        return self.conv(sigma * x_p + (1.0 - sigma) * x_i)



class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (DDRNet/PIDNet-L)."""

    def __init__(self, in_ch: int, branch_ch: int, out_ch: int,
                 num_scales: int = 5,
                 kernel_sizes: Tuple[int, ...] = (5, 9, 17),
                 strides: Tuple[int, ...] = (2, 4, 8),
                 paddings: Tuple[int, ...] = (2, 4, 8)):
        super().__init__()
        self.num_scales = num_scales
        self.scales = nn.ModuleList()
        self.scales.append(PreNormConvBNReLU(in_ch, branch_ch, 1, 1, 0))
        for i in range(1, num_scales - 1):
            self.scales.append(nn.Sequential(
                nn.AvgPool2d(kernel_sizes[i - 1], strides[i - 1],
                             paddings[i - 1]),
                PreNormConvBNReLU(in_ch, branch_ch, 1, 1, 0),
            ))
        self.scales.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            PreNormConvBNReLU(in_ch, branch_ch, 1, 1, 0),
        ))
        self.processes = nn.ModuleList([
            PreNormConvBNReLU(branch_ch, branch_ch, 3, 1, 1)
            for _ in range(num_scales - 1)
        ])
        self.compression = PreNormConvBNReLU(
            branch_ch * num_scales, out_ch, 1, 1, 0)
        self.shortcut = PreNormConvBNReLU(in_ch, out_ch, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        feats = [self.scales[0](x)]
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(self.scales[i](x), size=x.shape[-2:],
                                    mode="bilinear", align_corners=False)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))
        return (self.compression(torch.cat(feats, dim=1))
                + self.shortcut(x))


class PAPPM(DAPPM):
    """Patch Aggregation Pyramid Pooling Module (PIDNet-S/M)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_minus_1 = self.num_scales - 1
        branch_ch = self.scales[0].conv.out_channels
        self.processes = PreNormConvBNReLU(
            branch_ch * n_minus_1, branch_ch * n_minus_1,
            3, 1, 1, groups=n_minus_1)

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.scales[0](x)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(self.scales[i](x), size=x.shape[-2:],
                                    mode="bilinear", align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return (self.compression(torch.cat([x_, scale_out], dim=1))
                + self.shortcut(x))



class _PIDHeadBlock(nn.Module):
    """PreNormConvBNReLU 3x3, then BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = PreNormConvBNReLU(in_ch, out_ch, 3, 1, 1, act=True)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x, inplace=False)



class _PIDNet(nn.Module):
    """Backbone and head in one module, mirroring mmseg's PIDNet + PIDHead."""

    def __init__(self, *,
                 num_classes: int = 19,
                 channels: int = 32,
                 ppm_channels: int = 96,
                 head_channels: Optional[int] = None,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

        c = channels
        if head_channels is None:
            head_channels = c * 2 if c >= 64 else 128
        self._head_channels = head_channels

        self.stem = self._make_stem_layer(3, c, num_stem_blocks)
        self.relu = nn.ReLU(inplace=False)

        self.i_branch_layers = nn.ModuleList([
            self._make_layer(BasicBlock, c * 2, c * 4,
                             num_branch_blocks, stride=2),
            self._make_layer(BasicBlock, c * 4, c * 8,
                             num_branch_blocks, stride=2),
            self._make_layer(Bottleneck, c * 8, c * 8, 2, stride=2),
        ])

        self.p_branch_layers = nn.ModuleList([
            self._make_layer(BasicBlock, c * 2, c * 2, num_stem_blocks),
            self._make_layer(BasicBlock, c * 2, c * 2, num_stem_blocks),
            self._make_layer(Bottleneck, c * 2, c * 2, 1),
        ])

        self.compression_1 = ConvBNReLU(c * 4, c * 2, 1, 1, 0, act=False)
        self.compression_2 = ConvBNReLU(c * 8, c * 2, 1, 1, 0, act=False)
        self.pag_1 = PagFM(c * 2, c)
        self.pag_2 = PagFM(c * 2, c)

        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, c * 2, c),
                self._make_layer(Bottleneck, c, c, 1),
            ])
            channel_expand = 1
            spp_module: type = PAPPM
            dfm_module: type = LightBag
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, c * 2, c * 2),
                self._make_single_layer(BasicBlock, c * 2, c * 2),
            ])
            channel_expand = 2
            spp_module = DAPPM
            dfm_module = Bag

        self.diff_1 = ConvBNReLU(c * 4, c * channel_expand, 3, 1, 1,
                                 act=False)
        self.diff_2 = ConvBNReLU(c * 8, c * 2, 3, 1, 1, act=False)

        self.spp = spp_module(c * 16, ppm_channels, c * 4, num_scales=5)
        self.dfm = dfm_module(c * 4, c * 4)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, c * 2, c * 2, 1))

        in_main = c * 4   # = output of dfm
        in_p = c * 2      # = output of P-branch stage 1 (pre-stage2)
        in_d = c * 2      # = output of D-branch stage 1 (pre-stage2)
        self.i_head = _PIDHeadBlock(in_main, head_channels)
        self.p_head = _PIDHeadBlock(in_p, head_channels)
        self.d_head = _PIDHeadBlock(in_d, in_main // 4)
        self.conv_seg = nn.Conv2d(head_channels, num_classes, 1)
        self.p_cls_seg = nn.Conv2d(head_channels, num_classes, 1)
        self.d_cls_seg = nn.Conv2d(in_main // 4, 1, 1)

        self._init_weights()


    def _make_stem_layer(self, in_ch: int, c: int,
                         num_blocks: int) -> nn.Sequential:
        layers = [
            ConvBNReLU(in_ch, c, 3, 2, 1),
            ConvBNReLU(c, c, 3, 2, 1),
            self._make_layer(BasicBlock, c, c, num_blocks),
            nn.ReLU(inplace=False),
            self._make_layer(BasicBlock, c, c * 2, num_blocks, stride=2),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def _make_layer(self, block: type, in_ch: int, ch: int,
                    num_blocks: int, stride: int = 1) -> nn.Sequential:
        out_ch = ch * block.expansion
        downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            downsample = ConvBNReLU(in_ch, out_ch, 1, stride, 0, act=False)
        layers = [block(in_ch, ch, stride, downsample)]
        cur = out_ch
        for i in range(1, num_blocks):
            last = (i == num_blocks - 1)
            if block is BasicBlock:
                layers.append(block(cur, ch, 1, None,
                                    act_out=not last))
            else:
                layers.append(block(cur, ch, 1, None,
                                    act_out=False))
            cur = ch * block.expansion
        return nn.Sequential(*layers)

    def _make_single_layer(self, block: type, in_ch: int,
                           ch: int, stride: int = 1) -> nn.Module:
        out_ch = ch * block.expansion
        downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            downsample = ConvBNReLU(in_ch, out_ch, 1, stride, 0, act=False)
        if block is BasicBlock:
            return block(in_ch, ch, stride, downsample, act_out=False)
        return block(in_ch, ch, stride, downsample, act_out=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def load_official_state_dict(self, state_dict: dict, *,
                                 strict: bool = True
                                 ) -> Tuple[list, list]:
        """Load an mmseg-style checkpoint, stripping backbone./decode_head. prefixes."""
        log = logging.getLogger("pidnet")

        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")
        for sub_key in ("state_dict", "model"):
            if sub_key in state_dict and isinstance(state_dict[sub_key], dict):
                state_dict = state_dict[sub_key]
                break

        renamed: dict = {}
        for k, v in state_dict.items():
            nk = k
            while nk.startswith("module."):
                nk = nk[len("module."):]
            for prefix in ("backbone.", "decode_head."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            renamed[nk] = v

        result = self.load_state_dict(renamed, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing or unexpected:
            log.info(
                "PIDNet official load: %d/%d keys matched "
                "(missing=%d, unexpected=%d)",
                len(renamed) - len(unexpected),
                len(self.state_dict()),
                len(missing), len(unexpected),
            )
            if missing[:5]:
                log.info("  e.g. missing: %s", ", ".join(missing[:5]))
            if unexpected[:5]:
                log.info("  e.g. unexpected: %s",
                         ", ".join(unexpected[:5]))
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Strict load failed: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys.")
        return missing, unexpected


    def forward(self, x: Tensor
                ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        H, W = x.shape[-2:]
        h_out = H // 8
        w_out = W // 8

        x = self.stem(x)

        x_i = self.relu(self.i_branch_layers[0](x))   # 4c, s=16
        x_p = self.p_branch_layers[0](x)              # 2c, s=8
        x_d = self.d_branch_layers[0](x)              # c, s=8

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)

        diff_i = self.diff_1(x_i)
        diff_i = F.interpolate(diff_i, size=(h_out, w_out),
                               mode="bilinear",
                               align_corners=self.align_corners)
        x_d = x_d + diff_i

        if self.training:
            temp_p = x_p

        x_i = self.relu(self.i_branch_layers[1](x_i))                # 8c, s=32
        x_p = self.p_branch_layers[1](self.relu(x_p))                # 2c, s=8
        x_d = self.d_branch_layers[1](self.relu(x_d))                # 2c, s=8

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)

        diff_i = self.diff_2(x_i)
        diff_i = F.interpolate(diff_i, size=(h_out, w_out),
                               mode="bilinear",
                               align_corners=self.align_corners)
        x_d = x_d + diff_i

        if self.training:
            temp_d = x_d

        x_i = self.i_branch_layers[2](x_i)                           # 16c, s=64
        x_p = self.p_branch_layers[2](self.relu(x_p))                # 4c, s=8
        x_d = self.d_branch_layers[2](self.relu(x_d))                # 4c, s=8

        x_i = self.spp(x_i)
        x_i = F.interpolate(x_i, size=(h_out, w_out),
                            mode="bilinear",
                            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)                                # 4c, s=8

        main = self.conv_seg(self.i_head(out))
        main = F.interpolate(main, size=(H, W),
                             mode="bilinear",
                             align_corners=self.align_corners)

        if self.training:
            p_aux = self.p_cls_seg(self.p_head(temp_p))
            p_aux = F.interpolate(p_aux, size=(H, W),
                                  mode="bilinear",
                                  align_corners=self.align_corners)
            d_bnd = self.d_cls_seg(self.d_head(temp_d))
            d_bnd = F.interpolate(d_bnd, size=(H, W),
                                  mode="bilinear",
                                  align_corners=self.align_corners)
            return main, p_aux, d_bnd

        return main



@register_model(aux_models)
class PIDNetS(_PIDNet):
    """PIDNet-Small: 7.6 M params, 78.4 mIoU @ 1024x2048."""

    def __init__(self, num_classes: int = 19, **kwargs):
        super().__init__(
            num_classes=num_classes,
            channels=32,
            ppm_channels=96,
            head_channels=128,
            num_stem_blocks=2,
            num_branch_blocks=3,
        )


@register_model(aux_models)
class PIDNetM(_PIDNet):
    """PIDNet-Medium: 28.4 M params, 79.8 mIoU."""

    def __init__(self, num_classes: int = 19, **kwargs):
        super().__init__(
            num_classes=num_classes,
            channels=64,
            ppm_channels=96,
            head_channels=128,
            num_stem_blocks=2,
            num_branch_blocks=3,
        )


@register_model(aux_models)
class PIDNetL(_PIDNet):
    """PIDNet-Large: 37.4 M params, 80.9 mIoU."""

    def __init__(self, num_classes: int = 19, **kwargs):
        super().__init__(
            num_classes=num_classes,
            channels=64,
            ppm_channels=112,
            head_channels=256,
            num_stem_blocks=3,
            num_branch_blocks=4,
        )
