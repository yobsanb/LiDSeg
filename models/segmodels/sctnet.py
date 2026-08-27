# Ported from https://github.com/xzz777/SCTNet
# Copyright (c) 2024 Zhengze Xu. MIT License.
"""SCTNet (Xu et al., AAAI 2024).
Standalone port of the inference architecture from xzz777/SCTNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module = None,
                 no_relu: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class _MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = None,
                 out_channels: int = None, drop_rate: float = 0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm  = nn.BatchNorm2d(in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop  = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.conv1(x))
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class _ConvAttention(nn.Module):
    """Strip-shaped convolutional attention (SCTNet's CFBlock attention)."""
    def __init__(self, in_channels: int, out_channels: int,
                 inter_channels: int, num_heads: int = 8):
        super().__init__()
        assert out_channels % num_heads == 0
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.inter_channels = inter_channels
        self.num_heads      = num_heads
        self.norm = nn.BatchNorm2d(in_channels)
        self.kv  = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        nn.init.trunc_normal_(self.kv, std=1e-3)
        nn.init.trunc_normal_(self.kv3, std=1e-3)

    def _act_dn(self, x):
        b, _, h, w = x.shape
        x = x.reshape(b, self.num_heads,
                      self.inter_channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
        return x.reshape(b, self.inter_channels, h, w)

    def forward(self, x):
        x = self.norm(x)
        x1 = F.conv2d(x, self.kv, bias=None, stride=1, padding=(3, 0))
        x1 = self._act_dn(x1)
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), stride=1, padding=(3, 0))
        x3 = F.conv2d(x, self.kv3, bias=None, stride=1, padding=(0, 3))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), stride=1, padding=(0, 3))
        return x1 + x3


class _CFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 8, drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.attn_l = _ConvAttention(in_channels, out_channels,
                                     inter_channels=64, num_heads=num_heads)
        self.mlp_l  = _MLP(out_channels, drop_rate=drop_rate)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x))
        return x


class _DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (PIDNet/SCTNet variant)."""
    def __init__(self, in_channels: int, inter_channels: int, out_channels: int):
        super().__init__()
        def _branch(pool):
            return nn.Sequential(
                pool,
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, inter_channels, 1))
        self.scale1 = _branch(nn.AvgPool2d(5, 2, 2, count_include_pad=True))
        self.scale2 = _branch(nn.AvgPool2d(9, 4, 4, count_include_pad=True))
        self.scale3 = _branch(nn.AvgPool2d(17, 8, 8, count_include_pad=True))
        self.scale4 = _branch(nn.AdaptiveAvgPool2d((1, 1)))
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, inter_channels, 1))

        def _proc():
            return nn.Sequential(
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=False),
                nn.Conv2d(inter_channels, inter_channels, 3, padding=1))
        self.process1 = _proc(); self.process2 = _proc()
        self.process3 = _proc(); self.process4 = _proc()
        self.compression = nn.Sequential(
            nn.BatchNorm2d(inter_channels * 5), nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels * 5, out_channels, 1))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        size = x.shape[2:]
        xs = [self.scale0(x)]
        xs.append(self.process1(F.interpolate(self.scale1(x), size=size,
                                              mode='bilinear') + xs[0]))
        xs.append(self.process2(F.interpolate(self.scale2(x), size=size,
                                              mode='bilinear') + xs[1]))
        xs.append(self.process3(F.interpolate(self.scale3(x), size=size,
                                              mode='bilinear') + xs[2]))
        xs.append(self.process4(F.interpolate(self.scale4(x), size=size,
                                              mode='bilinear') + xs[3]))
        return self.compression(torch.cat(xs, dim=1)) + self.shortcut(x)


class SCTNet(nn.Module):
    """SCTNet inference network (CNN-only branch + SCT head)."""

    OFFICIAL_PREFIXES = ("backbone.", "decode_head.")

    def __init__(self,
                 n_classes: int = 19,
                 base_channels: int = 32,
                 spp_channels: int = 64,
                 layer_nums=(2, 2, 2, 2),
                 num_heads: int = 8,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.n_classes      = n_classes
        self.base_channels  = base_channels
        self.spp_channels   = spp_channels

        c = base_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=False),
            nn.Conv2d(c, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._stage(_BasicBlock, c,     c,     layer_nums[0])
        self.layer2 = self._stage(_BasicBlock, c,     c * 2, layer_nums[1], stride=2)
        self.layer3 = self._stage(_BasicBlock, c * 2, c * 4, layer_nums[2], stride=2)

        self.layer3_2 = _CFBlock(c * 4, c * 4, num_heads=num_heads,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate)
        self.convdown4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, 3, 2, 1),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=False))
        self.layer4 = _CFBlock(c * 8, c * 8, num_heads=num_heads,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate)
        self.layer5 = _CFBlock(c * 8, c * 8, num_heads=num_heads,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate)

        self.spp = _DAPPM(c * 8, spp_channels, c * 2)

        head_in_ch = c * 4
        head_ch    = c * 4
        self.bn1   = nn.BatchNorm2d(head_in_ch)
        self.conv1_head = nn.Conv2d(head_in_ch, head_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(head_ch)
        self.conv_seg = nn.Conv2d(head_ch, n_classes, 1)

        self.encoder_s8  = nn.Identity()
        self.encoder_s16 = nn.Identity()
        self.encoder_s32 = nn.Identity()

    @staticmethod
    def _stage(block, in_ch, out_ch, n, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                nn.BatchNorm2d(out_ch))
        layers = [block(in_ch, out_ch, stride, downsample)]
        for i in range(1, n):
            layers.append(block(out_ch, out_ch, 1,
                                no_relu=(i == n - 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(self.conv1(x))                  # c, 1/4
        x2 = self.layer2(self.relu(x1))                  # 2c, 1/8
        x3_1 = self.layer3(self.relu(x2))                # 4c, 1/16
        x3 = self.layer3_2(self.relu(x3_1))              # 4c, 1/16
        x3 = self.encoder_s16(x3)
        x4 = self.layer4(self.relu(self.convdown4(x3)))  # 8c, 1/32
        x5 = self.layer5(self.relu(x4))                  # 8c, 1/32
        x5 = self.encoder_s32(x5)
        x6 = self.spp(x5)                                # 2c, 1/32
        x7 = F.interpolate(x6, size=x2.shape[2:], mode='bilinear',
                           align_corners=False)          # 2c, 1/8
        x_out = torch.cat([x2, x7], dim=1)               # 4c, 1/8
        x_out = self.encoder_s8(x_out)

        h = self.conv1_head(self.relu(self.bn1(x_out)))
        out = self.conv_seg(self.relu(self.bn2(h)))
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear',
                            align_corners=False)
        return out

    @torch.no_grad()
    def load_official_state_dict(self, sd: dict, strict: bool = True):
        """Map the official mmseg checkpoint onto our layer names."""
        out = {}
        rename = {
            "conv1.0":          "conv1.0",   # Conv1
            "conv1.1":          "conv1.1",   # BN1
            "conv1.3":          "conv1.3",
            "conv1.4":          "conv1.4",
            "conv_seg":         "conv_seg",
            "bn1":              "bn1",
            "conv1":            "conv1_head",
            "bn2":              "bn2",
        }
        for k, v in sd.items():
            new_k = k
            for pref in self.OFFICIAL_PREFIXES:
                if new_k.startswith(pref):
                    new_k = new_k[len(pref):]
                    break
            if new_k == "conv1.weight" or new_k == "conv1.bias":
                new_k = "conv1_head" + new_k[len("conv1"):]
            out[new_k] = v
        return self.load_state_dict(out, strict=strict)


from .model_registry import register_model


@register_model()
class SCTNetS(SCTNet):
    """SCTNet-Slim: ~4.7 M params, 76.33 % mIoU @ 149 FPS on Cityscapes (Seg75)."""

    def __init__(self, n_classes: int = 19, **kwargs):
        super().__init__(n_classes=n_classes,
                         base_channels=32, spp_channels=64)


@register_model()
class SCTNetB(SCTNet):
    """SCTNet-Base: ~17.4 M params, 79.83 % mIoU @ 105 FPS on Cityscapes (Seg75)."""

    def __init__(self, n_classes: int = 19, **kwargs):
        super().__init__(n_classes=n_classes,
                         base_channels=64, spp_channels=128)
