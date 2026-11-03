"""Export a training checkpoint as a weights-only file (no optimiser state).

The result is what the ACDC submission form's "Model File" field takes (limit 600 MB):
the deployed model's state_dict plus the architecture record needed to rebuild it.

    python tools/export_weights.py --checkpoint output/acdc/<run>/best.pth --out model.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from evaluation._common import NUM_CLASSES, build_model_from_checkpoint  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    model, architecture, meta = build_model_from_checkpoint(
        str(args.checkpoint), expected_n_classes=NUM_CLASSES)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "architecture": architecture,
                "source_checkpoint": args.checkpoint.name,
                "meta": {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in meta.items() if k in ("stage", "epoch", "miou", "best")}},
               args.out)
    n = sum(v.numel() for v in state.values() if v.is_floating_point())
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"-> {args.out}  {args.out.stat().st_size / 1e6:.1f} MB  {len(state)} tensors  "
          f"{n / 1e6:.2f}M values  sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
