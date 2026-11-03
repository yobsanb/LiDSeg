"""Download the official OpenMMLab PIDNet-L Cityscapes checkpoint and write its weights.

The mmseg file also pickles mmengine training-log objects, which the trainer's
``weights_only`` loader rejects. This extracts the tensor ``state_dict`` with a
restricted unpickler (only torch/collections globals resolve; everything else is
an inert stub, so nothing from the download executes) and writes a weights-only
file that the trainer loads safely.

    python tools/prepare_pidnetl_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import tempfile
import types
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
URL = ("https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/"
       "pidnet-l_2xb6-120k_1024x1024-cityscapes/"
       "pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth")
SOURCE_SHA256 = "0783ca6bb5a68a6992daf07a6ece908a236ef9dd17313d53e95467f13d810d2b"
OUTPUT_SHA256 = "56f23bc4c2ba3bf87f647bbadd8e8e6f924ef1f68d6b6f4cbbd0085b38bf0836"
DEFAULT_OUT = REPO / "pretrained" / "PIDNet_L_Cityscapes_mmseg.pth"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _stub_factory(*args, **kwargs):
    return _Stub()


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("torch", "collections"):
            return super().find_class(module, name)
        return _stub_factory if name in ("scalar", "_reconstruct", "dtype") else _Stub


_restricted = types.ModuleType("restricted_pickle")
_restricted.Unpickler = _RestrictedUnpickler
_restricted.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=None,
                   help="already-downloaded mmseg checkpoint (default: download)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        src = args.source
        if src is None:
            src = Path(tmp) / "pidnet-l_mmseg.pth"
            print(f"GET {URL}")
            urllib.request.urlretrieve(URL, src)
        if _sha256(src) != SOURCE_SHA256:
            raise SystemExit(f"{src}: SHA-256 differs from the published checkpoint")

        blob = torch.load(src, map_location="cpu", weights_only=False,
                          pickle_module=_restricted)
        state = dict(blob["state_dict"])
        if not all(torch.is_tensor(v) for v in state.values()):
            raise SystemExit("state_dict contains non-tensor entries")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state}, args.out)

    torch.load(args.out, map_location="cpu", weights_only=True)
    digest = _sha256(args.out)
    if digest != OUTPUT_SHA256:
        raise SystemExit(f"{args.out}: unexpected SHA-256 {digest}")
    print(f"wrote {args.out} ({len(state)} tensors, sha256 {digest})")


if __name__ == "__main__":
    main()
