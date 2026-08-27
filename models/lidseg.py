"""LiDSeg model."""

from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from models.dehazeformer import (DEHAZE_CHANNELS, LinearDehazeFormerEncoder,
                                 ReconstructionDecoder)
from models.lgf import build_lgf
from models.segmodels import aux_models, model_hub
from models.segmodels.feature_registry import get_features

__all__ = ["LiDSegModel", "build_lidseg"]

_SCALES: Tuple[str, str, str] = ("s8", "s16", "s32")


def forward_accepts_kw(model: nn.Module, name: str) -> bool:
    """True if ``forward`` accepts ``name=`` or ``**kwargs``."""
    core = model
    for _ in range(16):
        inner = getattr(core, "module", None)
        if isinstance(inner, nn.Module):
            core = inner
        else:
            break
    try:
        sig = inspect.signature(core.forward)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    """Walk a dotted attribute path to a submodule."""
    module = root
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


class _LGFInjector:
    """Forward hook replacing a backbone tap output with LGF(f, g)."""

    def __init__(self, lgf: nn.Module, scale: str, store: dict):
        self.lgf = lgf
        self.scale = scale
        self.store = store

    def __call__(self, module, inputs, output):
        g = self.store.get(self.scale)
        if g is None:
            raise RuntimeError(
                f"guidance feature '{self.scale}' missing; the guidance encoder "
                "must run before the backbone forward pass")
        return self.lgf(output.clone(), g).clone()


class LiDSegModel(nn.Module):
    """Segmentation backbone, optionally wrapped by the guidance encoder and LGF."""

    def __init__(
        self,
        backbone_name: str,
        n_classes: int = 19,
        guidance: bool = True,
        use_decoder: bool = True,
        guidance_channels: Sequence[int] = DEHAZE_CHANNELS,
        guidance_depths: Sequence[int] = (4, 6, 6),
        backbone_kwargs: Optional[dict] = None,
        guidance_encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if use_decoder and not guidance:
            raise ValueError("use_decoder requires guidance=True")

        self.backbone = self._build_backbone(backbone_name, n_classes,
                                             dict(backbone_kwargs or {}))
        self.backbone_name = backbone_name
        self.guidance_enabled = guidance
        self.use_decoder = use_decoder

        spec = get_features(backbone_name)
        key = self._normalise(backbone_name)
        self.has_aux = spec.get("has_aux", key in aux_models)
        self._backbone_accepts_training = forward_accepts_kw(
            self.backbone, "is_training")

        if not guidance:
            return

        for module in self.backbone.modules():
            if isinstance(module, nn.ReLU) and getattr(module, "inplace", False):
                module.inplace = False

        guidance_channels = tuple(guidance_channels)
        self.guidance = LinearDehazeFormerEncoder(
            in_channels=3, channels=guidance_channels,
            depths=tuple(guidance_depths),
            **dict(guidance_encoder_kwargs or {}))
        if use_decoder:
            self.decoder = ReconstructionDecoder(encoder_channels=guidance_channels)

        self._guidance_buffer: dict = {}
        self.lgf = nn.ModuleDict()
        self._hook_handles: list = []
        for index, tag in enumerate(_SCALES):
            path, channels = spec["features"][tag]
            self.lgf[tag] = build_lgf(f_channels=channels,
                                      g_channels=guidance_channels[index])
            tap = _resolve_module(self.backbone, path)
            self._hook_handles.append(tap.register_forward_hook(
                _LGFInjector(self.lgf[tag], tag, self._guidance_buffer)))

    @staticmethod
    def _normalise(name: str) -> str:
        """Registry key form of a backbone name."""
        return name.lower().replace("_", "").replace("-", "")

    def _build_backbone(self, name: str, n_classes: int, kwargs: dict) -> nn.Module:
        """Instantiate a registered backbone, passing its class-count keyword."""
        key = self._normalise(name)
        if key not in model_hub:
            raise KeyError(f"backbone {name!r} is not registered; "
                           f"available: {sorted(model_hub)}")
        cls = model_hub[key]
        params = inspect.signature(cls).parameters
        for kw in ("n_classes", "num_class", "num_classes"):
            if kw in params and kw not in kwargs:
                kwargs[kw] = n_classes
                break
        return cls(**kwargs)

    def _backbone_forward(self, x: Tensor):
        """Run the backbone, requesting aux heads when it takes that flag."""
        if self.has_aux and self.training and self._backbone_accepts_training:
            return self.backbone(x, is_training=True)
        return self.backbone(x)

    @staticmethod
    def _split_outputs(raw, n_classes: int) -> Tuple[Tensor, List[Tensor]]:
        """Flatten a backbone return value into main logits and aux logits."""
        bag: List[Tensor] = []

        def walk(node):
            if torch.is_tensor(node):
                bag.append(node)
            elif isinstance(node, (tuple, list)):
                for item in node:
                    walk(item)

        walk(raw)
        if not bag:
            raise RuntimeError("backbone returned no tensors")
        aux = [t for t in bag[1:] if t.dim() == 4 and t.shape[1] == n_classes]
        return bag[0], aux

    def _run_backbone(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Backbone forward with guidance already staged in the hook buffer."""
        raw = self._backbone_forward(x)
        first = raw[0] if isinstance(raw, (tuple, list)) else raw
        while isinstance(first, (tuple, list)):
            first = first[0]
        return self._split_outputs(raw, first.shape[1])

    def forward(self, x_adverse: Tensor, x_clear: Optional[Tensor] = None,
                return_recon: bool = False) -> dict:
        """Logits and aux logits, plus recon while training with a decoder."""
        del x_clear

        if not self.guidance_enabled:
            logits, aux_logits = self._run_backbone(x_adverse)
            return {"logits": logits, "aux_logits": aux_logits}

        guidance = self.guidance(x_adverse)
        self._guidance_buffer.clear()
        self._guidance_buffer.update(guidance)
        try:
            logits, aux_logits = self._run_backbone(x_adverse)
        finally:
            self._guidance_buffer.clear()

        out = {"logits": logits, "aux_logits": aux_logits, "g_adverse": guidance}
        if (self.training or return_recon) and self.use_decoder:
            out.update(self.decoder(guidance, x_adverse))
        return out

    @torch.no_grad()
    def predict(self, x: Tensor, return_recon: bool = False):
        """Deployed inference path: backbone, guidance encoder and LGF.

        ``return_recon=True`` also runs the decoder and returns ``(logits, recon)``.
        """
        if return_recon and not self.use_decoder:
            raise ValueError(
                "return_recon=True requires a model built with use_decoder=True")
        was_training = self.training
        self.eval()
        try:
            out = self.forward(x, return_recon=return_recon)
            if return_recon:
                return out["logits"], out["recon"]
            return out["logits"]
        finally:
            if was_training:
                self.train()


def build_lidseg(backbone: str, num_classes: int = 19, guidance: bool = True,
                 use_decoder: bool = True, **kwargs) -> LiDSegModel:
    """Convenience factory for :class:`LiDSegModel`."""
    return LiDSegModel(backbone_name=backbone, n_classes=num_classes,
                       guidance=guidance, use_decoder=use_decoder, **kwargs)
