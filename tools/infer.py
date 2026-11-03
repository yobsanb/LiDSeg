"""LiDSeg inference on images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.lidseg import LiDSegModel  # noqa: E402

DEFAULT_WEIGHTS = REPO / "pretrained" / "lidseg_pidnetl_acdc.pth"
CLASSES: Tuple[str, ...] = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)  # Cityscapes colors
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def load_model(weights: Union[str, Path], device: Union[str, torch.device] = "cpu") -> LiDSegModel:
    """Load released weights, eval mode."""
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
    """RGB uint8 to normalized tensor."""
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """Image file to [1, 3, H, W] input."""
    return to_input(np.asarray(Image.open(path).convert("RGB")))[None]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="image or folder")
    p.add_argument("--output", type=Path, required=True, help="output folder")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--color", action="store_true", help="also save color maps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.input.is_dir():
        images = sorted(f for f in args.input.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    else:
        images = [args.input]
    if not images:
        raise SystemExit(f"no images found under {args.input}")

    model = load_model(args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for f in images:
            pred = model.predict(load_image(f).to(args.device)).argmax(1)[0]
            pred = pred.to(torch.uint8).cpu().numpy()
            Image.fromarray(pred).save(args.output / f"{f.stem}_labelTrainIds.png")
            if args.color:
                Image.fromarray(PALETTE[pred]).save(args.output / f"{f.stem}_color.png")
    print(f"{len(images)} prediction(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
