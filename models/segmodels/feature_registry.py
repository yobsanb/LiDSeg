"""s8/s16/s32 tap paths and channel widths per backbone, used by the LGF hooks."""

from copy import deepcopy
from typing import Dict, List, Tuple

__all__ = [
    "BACKBONE_FEATURES",
    "get_features",
    "get_channels",
    "get_hook_paths",
    "list_backbones",
    "register_backbone",
]

BACKBONE_FEATURES: Dict[str, Dict] = {
    # PIDNet-L (c=64): s8 = P-branch before the DFM fusion, s16/s32 = I-branch stages.
    "pidnetl": {
        "has_aux": True,
        "features": {
            "s8":  ("p_branch_layers.2", 256),  # 4*c after Bottleneck
            "s16": ("i_branch_layers.0", 256),  # 4*c, stride 16
            "s32": ("i_branch_layers.1", 512),  # 8*c, stride 32
        },
    },
}


def _norm(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")

def list_backbones() -> List[str]:
    """Return the sorted list of registered backbone names."""
    return sorted(BACKBONE_FEATURES.keys())

def get_features(backbone: str) -> Dict:
    """Return the (deep-copied) feature spec for a backbone."""
    key = _norm(backbone)
    if key not in BACKBONE_FEATURES:
        raise KeyError(
            f"Backbone '{backbone}' not registered. "
            f"Known backbones: {list_backbones()}"
        )
    return deepcopy(BACKBONE_FEATURES[key])

def get_hook_paths(backbone: str) -> Dict[str, str]:
    """Return {scale_tag: module_path} for the given backbone."""
    return {tag: spec[0] for tag, spec in get_features(backbone)["features"].items()}

def get_channels(backbone: str) -> Tuple[int, int, int]:
    """Return (s8_ch, s16_ch, s32_ch) tuple."""
    feats = get_features(backbone)["features"]
    return tuple(spec[1] for spec in feats.values())

def register_backbone(name: str, has_aux: bool,
                      features: Dict[str, Tuple[str, int]]) -> None:
    """Register a backbone's taps at runtime."""
    key = _norm(name)
    if key in BACKBONE_FEATURES:
        raise ValueError(f"Backbone '{name}' already registered.")
    if set(features.keys()) != {"s8", "s16", "s32"}:
        raise ValueError(
            f"Backbone '{name}' must define exactly the scales "
            f"{{'s8','s16','s32'}}, got {set(features.keys())}."
        )
    BACKBONE_FEATURES[key] = {"has_aux": bool(has_aux), "features": dict(features)}
