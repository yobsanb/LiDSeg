"""Registry of pretrained backbone checkpoints resolved under ``pretrained/``."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRETRAIN_DIR = _REPO_ROOT / "pretrained"


BACKBONE_PRETRAINED: dict[str, Tuple[str, str, Optional[str]]] = {
    "pidnetl": ("PIDNet_L_Cityscapes_mmseg.pth", "cityscapes", None),
}

# Weights-only file written by tools/prepare_pidnetl_weights.py; keep in sync with it.
BACKBONE_PRETRAINED_SHA256: dict[str, str] = {
    "pidnetl": "56f23bc4c2ba3bf87f647bbadd8e8e6f924ef1f68d6b6f4cbbd0085b38bf0836",
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
            f"{expected}, got {actual}. Recreate it with "
            "'python tools/prepare_pidnetl_weights.py'."
        )


def resolve_pretrained(backbone_name: str) -> Optional[str]:
    """Absolute path of a backbone's checkpoint, or None."""
    info = resolve_pretrained_full(backbone_name)
    return None if info is None else info[0]


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
