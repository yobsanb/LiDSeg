"""Load the LiDSeg PIDNet-L model from its weights file and prepare its input (inference only).

    from lidseg_infer import load_model, load_image
    model = load_model("lidseg_pidnetl_acdc.pth", device="cuda")
    x = load_image("data/ACDC/rgb_anon/fog/val/GOPR0476/GOPR0476_frame_000761_rgb_anon.png")
    logits = model.predict(x.to("cuda"))  # [1, 19, 1080, 1920]
    pred = logits.argmax(1)               # Cityscapes trainIds 0..18

Run as a script it prints the parameter breakdown and writes the per-module layer table:

    python lidseg_infer.py --layers layers.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
# In the hand-off bundle models/ sits next to this file; in LiDSeg-Release it is one level up.
sys.path.insert(0, str(HERE if (HERE / "models").is_dir() else HERE.parent))

from models.lidseg import LiDSegModel  # noqa: E402

CLASSES: Tuple[str, ...] = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
IGNORE_INDEX = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet, RGB order
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(weights: Union[str, Path], device: Union[str, torch.device] = "cpu") -> LiDSegModel:
    """Rebuild the network recorded in the weights file and load it strictly, in eval mode."""
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    arch = blob["architecture"]
    model = LiDSegModel(
        backbone_name=arch["backbone"], n_classes=arch["n_classes"],
        guidance=arch["guidance"], use_decoder=arch["use_decoder"],
        guidance_channels=arch["guidance_channels"],
        guidance_depths=arch["guidance_depths"],
        guidance_encoder_kwargs=arch["guidance_encoder_kwargs"],
        backbone_kwargs=arch["backbone_kwargs"])
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.to(device).eval()


def to_input(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB array -> [3, H, W] float32, ImageNet-normalized."""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """Read an RGB image file as a [1, 3, H, W] model input."""
    return to_input(np.asarray(Image.open(path).convert("RGB")))[None]


def layer_table(model: LiDSegModel, height: int = 1080, width: int = 1920) -> List[Dict]:
    """One row per module that owns parameters or has no children, from one inference.

    ``calls`` counts how often the module runs in one forward pass: 0 marks parameters
    that inference never uses, and modules reused at several places run more than once.
    ``output_shape`` is the first call's output.
    """
    device = next(model.parameters()).device
    rows: Dict[str, Dict] = {}
    handles = []
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if not name or (own == 0 and any(True for _ in module.children())):
            continue
        row = rows[name] = {"module": name, "type": type(module).__name__,
                            "params": own, "calls": 0, "output_shape": ""}

        def hook(_module, _inputs, output, row=row):
            row["calls"] += 1
            if not row["output_shape"]:
                row["output_shape"] = ("x".join(map(str, output.shape))
                                       if torch.is_tensor(output) else type(output).__name__)
        handles.append(module.register_forward_hook(hook))
    try:
        model.predict(torch.zeros(1, 3, height, width, device=device))
    finally:
        for handle in handles:
            handle.remove()
    total = sum(p.numel() for p in model.parameters())
    if sum(r["params"] for r in rows.values()) != total:
        raise RuntimeError("layer table does not account for every parameter")
    return list(rows.values())


def main() -> int:
    p = argparse.ArgumentParser(description="Parameter breakdown and per-module layer table.")
    p.add_argument("--weights", type=Path, default=HERE / "lidseg_pidnetl_acdc.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", type=Path, default=None, help="write the layer table as CSV")
    args = p.parse_args()

    model = load_model(args.weights, args.device)
    rows = layer_table(model)
    groups: Dict[str, List[int]] = {}
    for r in rows:
        g = groups.setdefault(r["module"].split(".")[0], [0, 0])
        g[0] += r["params"]
        g[1] += r["params"] if r["calls"] else 0
    print(f"{args.weights}  sha256 {sha256_file(args.weights)}")
    print(f"{'group':<10}{'params':>14}{'used at inference':>20}")
    for name, (n, used) in groups.items():
        print(f"{name:<10}{n:>14,}{used:>20,}")
    total = sum(n for n, _ in groups.values())
    used = sum(u for _, u in groups.values())
    print(f"{'total':<10}{total:>14,}{used:>20,}")
    unused = sorted({".".join(r["module"].split(".")[:2])
                     for r in rows if r["params"] and not r["calls"]})
    print("parameters never used at inference (training-only heads):", ", ".join(unused) or "none")
    if args.layers:
        args.layers.parent.mkdir(parents=True, exist_ok=True)
        with open(args.layers, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {args.layers}  ({len(rows)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
