"""Validate and zip ACDC test predictions."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs.paths import load as load_paths  # noqa: E402

SIZE_LIMIT = 200_000_000  # benchmark zip limit, bytes
CONDITIONS = ("fog", "night", "rain", "snow")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_dir", type=Path, required=True)
    p.add_argument("--zip", type=Path, default=None)
    args = p.parse_args()

    root = load_paths()["ACDC_ROOT"]
    frames = {}
    for cond in CONDITIONS:
        for f in glob.glob(os.path.join(root, "rgb_anon", cond, "test", "*", "*_rgb_anon.png")):
            frames[os.path.basename(f).replace("_rgb_anon.png", "")] = cond
    if len(frames) != 2000:
        raise SystemExit(f"expected 2000 ACDC test frames, found {len(frames)}")

    preds = sorted(glob.glob(str(args.pred_dir / "labelTrainIds" / "**" / "*.png"), recursive=True))
    matched = {}
    for f in preds:
        name = os.path.basename(f)
        if "_ref_" in name:
            raise SystemExit(f"{f}: '_ref_' names are reserved for normal-condition images")
        frame = "_".join(name.split("_")[:3])  # GOPRxxxx_frame_xxxxxx
        if frame not in frames:
            raise SystemExit(f"{f}: does not start with an ACDC test frame name")
        if frame in matched:
            raise SystemExit(f"{f}: second prediction for {frame} ({matched[frame]})")
        matched[frame] = f
    missing = sorted(set(frames) - set(matched))
    if missing:
        raise SystemExit(f"{len(missing)} test frames have no prediction, e.g. {missing[:3]}")

    counts = np.zeros(256, np.int64)
    for f in preds:
        im = Image.open(f)
        if im.size != (1920, 1080) or im.mode not in ("L", "P"):
            raise SystemExit(f"{f}: size {im.size} mode {im.mode}; need 1920x1080 8-bit")
        counts += np.bincount(np.asarray(im, np.uint8).ravel(), minlength=256)
    if counts[19:].any():
        raise SystemExit(f"labels outside trainIds 0..18: {np.nonzero(counts[19:])[0] + 19}")
    per_cond = {c: sum(1 for fr in matched if frames[fr] == c) for c in CONDITIONS}
    print(f"OK: {len(preds)} predictions {per_cond}, trainIds 0..18, 1920x1080 8-bit")

    zpath = args.zip or args.pred_dir.parent / f"{args.pred_dir.name}.zip"
    rel = [Path(f).relative_to(args.pred_dir) for f in preds]
    dirs = sorted({d for r in rel for d in r.parents if d != Path(".")})
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # explicit folder entries, like zip -r
        for d in dirs:
            entry = zipfile.ZipInfo(d.as_posix().rstrip("/") + "/")
            entry.external_attr = (0o40777 << 16) | 0x10
            z.writestr(entry, b"")
        for f, r in zip(preds, rel):
            z.write(f, arcname=r.as_posix())
    size = zpath.stat().st_size
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        bad = z.testzip()
    files = [i for i in infos if not i.is_dir()]
    tops = {i.filename.split("/")[0] for i in infos}
    if bad is not None or len(files) != 2000 or tops != {"labelTrainIds"}:
        raise SystemExit(f"zip check failed: bad={bad} files={len(files)} top level={tops}")
    if size >= SIZE_LIMIT:
        raise SystemExit(f"{zpath} is {size / 1e6:.1f} MB, over the 200 MB limit")
    print(f"-> {zpath}  ({size / 1e6:.1f} MB, {len(files)} files in {len(dirs)} folders, "
          f"top level labelTrainIds/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
