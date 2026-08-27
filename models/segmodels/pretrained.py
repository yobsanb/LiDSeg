"""Registry of pretrained backbone checkpoints resolved under ``pretrained/legacy/``."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRETRAIN_DIR = _REPO_ROOT / "pretrained" / "legacy"


BACKBONE_PRETRAINED: dict[str, Tuple[str, str, Optional[str]]] = {
    "bisenetv2":      ("bisenet_v2_cityscapes.pth",        "cityscapes",         None),
    "sctnets":        ("sctnet_s_seg75_cityscapes.pth",    "cityscapes",         None),
    "ddrnet":         ("DDRNet_23_slim_cityscapes.pth",    "cityscapes",         None),
    "stdc":           ("STDC1-Seg_model_maxmIOU75.pth",    "cityscapes",         None),
    "pidnets":        ("PIDNet_S_Cityscapes_mmseg.pth",    "cityscapes",         None),
}

# Digests for automatically downloaded artifacts; keep in sync with tools/fetch_pretrained.py.
BACKBONE_PRETRAINED_SHA256: dict[str, str] = {
    "pidnets": "bb8e3bccdda7aa8e35362422363ce2190c68a09b46d0df261d438e7516adaf5c",
}


def _normalise(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_pinned_artifact(backbone: str, path: Path) -> None:
    expected = BACKBONE_PRETRAINED_SHA256.get(backbone)
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"Refusing {path}: SHA-256 mismatch for {backbone}; expected "
            f"{expected}, got {actual}. Re-fetch it with "
            "'python tools/fetch_pretrained.py --only pidnets --force'."
        )


def resolve_pretrained(backbone_name: str) -> Optional[str]:
    """Absolute path of a backbone's checkpoint, or None."""
    info = BACKBONE_PRETRAINED.get(_normalise(backbone_name))
    if info is None:
        return None
    rel = info[0]
    p = _PRETRAIN_DIR / rel
    if not p.is_file():
        return None
    _verify_pinned_artifact(_normalise(backbone_name), p)
    return str(p)


def resolve_pretrained_full(backbone_name: str
                            ) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return (path, kind, prefix) for a backbone, or None."""
    info = BACKBONE_PRETRAINED.get(_normalise(backbone_name))
    if info is None:
        return None
    rel, kind, prefix = info
    p = _PRETRAIN_DIR / rel
    if not p.is_file():
        return None
    _verify_pinned_artifact(_normalise(backbone_name), p)
    return str(p), kind, prefix
