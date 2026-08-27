"""Fetch public backbone checkpoints pinned by SHA-256 in ``MANIFEST``."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_PRETRAIN_DIR = _REPO / "pretrained" / "legacy"


@dataclass(frozen=True)
class CkptSpec:
    """One supported pretrained checkpoint and its acquisition policy."""
    backbone: str
    filename: str
    kind: str
    source: str
    locator: str = ""
    notes: str = ""
    sha256: str | None = None
    size_bytes: int | None = None


MANIFEST: list[CkptSpec] = [
    CkptSpec(
        backbone="bisenetv2",
        filename="bisenet_v2_cityscapes.pth",
        kind="cityscapes",
        source="manual",
        locator="https://github.com/CoinCheung/BiSeNet  (Cityscapes BiSeNetV2)",
        notes="Manual upstream download; no publisher SHA-256 is available.",
    ),
    CkptSpec(
        backbone="ddrnet",
        filename="DDRNet_23_slim_cityscapes.pth",
        kind="cityscapes",
        source="manual",
        locator="https://github.com/ydhongHIT/DDRNet  (DDRNet-23-slim)",
        notes="Manual upstream download; no publisher SHA-256 is available.",
    ),
    CkptSpec(
        backbone="stdc",
        filename="STDC1-Seg_model_maxmIOU75.pth",
        kind="cityscapes",
        source="manual",
        locator="https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1",
        notes=("Manual upstream download; no publisher SHA-256 is available. "
               "Official STDC1-Seg checkpoint, 768x1536 input."),
    ),
    CkptSpec(
        backbone="pidnets",
        filename="PIDNet_S_Cityscapes_mmseg.pth",
        kind="cityscapes",
        source="url",
        locator=(
            "https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/"
            "pidnet-s_2xb6-120k_1024x1024-cityscapes/"
            "pidnet-s_2xb6-120k_1024x1024-cityscapes_20230302_191700-bb8e3bcc.pth"
        ),
        sha256="bb8e3bccdda7aa8e35362422363ce2190c68a09b46d0df261d438e7516adaf5c",
        size_bytes=51_073_991,
        notes="OpenMMLab PIDNet-S; loads strict into models/segmodels/pidnet.py.",
    ),
    CkptSpec(
        backbone="sctnets",
        filename="sctnet_s_seg75_cityscapes.pth",
        kind="cityscapes",
        source="manual",
        locator="https://github.com/xzz777/SCTNet  (SCTNet-S Seg75)",
        notes="Manual upstream download; no publisher SHA-256 is available.",
    ),
]


def _url_file(spec: CkptSpec, dst: Path) -> None:
    """Download to a .part file, then rename."""
    if not spec.locator.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download URL for {spec.backbone}")
    print(f"  GET {spec.locator}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(spec.locator, timeout=60) as r, open(tmp, "wb") as f:
        if not r.geturl().startswith("https://"):
            raise RuntimeError(
                f"refusing redirected non-HTTPS download URL for {spec.backbone}: {r.geturl()}"
            )
        content_length = r.headers.get("Content-Length")
        if content_length and int(content_length) != spec.size_bytes:
            raise RuntimeError(
                f"unexpected Content-Length for {spec.backbone}: {content_length} "
                f"(expected {spec.size_bytes})"
            )
        total = 0
        while chunk := r.read(1 << 20):
            total += len(chunk)
            if total > spec.size_bytes:
                raise RuntimeError(
                    f"download for {spec.backbone} exceeded expected size "
                    f"{spec.size_bytes}"
                )
            f.write(chunk)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_spec(spec: CkptSpec) -> None:
    if spec.source not in {"manual", "url"}:
        raise ValueError(f"unknown manifest source {spec.source!r} for {spec.backbone!r}")
    if spec.source == "url":
        if not spec.sha256 or len(spec.sha256) != 64:
            raise ValueError(
                f"automated manifest entry {spec.backbone!r} needs a full SHA-256"
            )
        if any(c not in "0123456789abcdef" for c in spec.sha256.lower()):
            raise ValueError(f"invalid SHA-256 for {spec.backbone!r}")
        if not spec.size_bytes or spec.size_bytes <= 1024:
            raise ValueError(
                f"automated manifest entry {spec.backbone!r} needs an expected byte size"
            )


def _check_file(spec: CkptSpec, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size <= 1024:
        return False, "file is unexpectedly small"
    if spec.size_bytes is not None and path.stat().st_size != spec.size_bytes:
        return False, (
            f"size mismatch: expected {spec.size_bytes} bytes, "
            f"got {path.stat().st_size}"
        )
    if not spec.sha256:
        return True, "unverified (no publisher SHA-256)"
    actual = _sha256(path)
    if actual != spec.sha256:
        return False, f"SHA-256 mismatch: expected {spec.sha256}, got {actual}"
    return True, f"SHA-256 {actual}"


def _exists(spec: CkptSpec) -> bool:
    ok, _ = _check_file(spec, _PRETRAIN_DIR / spec.filename)
    return ok


def _fetch(spec: CkptSpec, force: bool) -> bool:
    _validate_spec(spec)
    dst = _PRETRAIN_DIR / spec.filename
    if spec.source == "manual":
        ok, status = _check_file(spec, dst)
        if ok:
            print(f"[{spec.backbone}] {spec.filename} already present ({status}).")
            return True
        print(f"[{spec.backbone}] manual download required from "
              f"{spec.locator}\n        -> save it as "
              f"{_PRETRAIN_DIR/spec.filename}")
        return False

    if dst.is_file() and not force:
        ok, status = _check_file(spec, dst)
        if ok:
            print(f"[{spec.backbone}] {spec.filename} already present ({status}) — skipping.")
            return True
        print(f"[{spec.backbone}] refusing existing {spec.filename}: {status}. "
              "Use --force to replace it.")
        return False

    print(f"[{spec.backbone}] downloading {spec.filename} ({spec.source}) ...")
    if spec.source == "url":
        _url_file(spec, dst)
    else:
        raise ValueError(f"unknown source {spec.source!r}")

    tmp = dst.with_suffix(dst.suffix + ".part")
    ok, status = _check_file(spec, tmp)
    if not ok:
        raise RuntimeError(f"download verification failed for {spec.backbone}: {status}")
    tmp.replace(dst)
    print(f"[{spec.backbone}] OK ({dst.stat().st_size//1024} KiB, {status}).")
    return True


def _verify(todo: list[CkptSpec]) -> bool:
    """Verify selected automatically fetched artifacts against their digests."""
    automated = [s for s in todo if s.source == "url"]
    if not automated:
        print("No automatically fetched, digest-pinned artifacts selected.")
        return True
    ok = True
    for spec in automated:
        valid, status = _check_file(spec, _PRETRAIN_DIR / spec.filename)
        print(f"[{spec.backbone}] {'OK' if valid else 'FAIL'}: {status}")
        ok = ok and valid
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None,
                   help="Limit to exact manifest backbone name(s).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file exists.")
    p.add_argument("--list", action="store_true",
                   help="Just print the manifest and exit.")
    p.add_argument("--verify", action="store_true",
                   help="Verify selected automated downloads against their SHA-256.")
    args = p.parse_args()

    _PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    for spec in MANIFEST:
        _validate_spec(spec)

    print(f"\nManifest ({len(MANIFEST)} entries) — pretrain dir = {_PRETRAIN_DIR}\n")
    for s in MANIFEST:
        flag = "X" if _exists(s) else " "
        digest = s.sha256 if s.sha256 else "manual/unverified"
        print(f"  [{flag}] {s.backbone:<12s} {s.kind:<12s} {s.filename}\n"
              f"      source: {s.source}\n"
              f"      sha256: {digest}")
        if s.size_bytes is not None:
            print(f"      size_bytes: {s.size_bytes}")
        if s.locator:
            print(f"      locator: {s.locator}")
        if s.notes:
            print(f"      notes: {s.notes}")
    print()

    todo = MANIFEST
    if args.only:
        wanted = set(args.only)
        known = {s.backbone for s in MANIFEST}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"Unknown backbone(s) for --only: {', '.join(unknown)}.")
            return 1
        todo = [s for s in MANIFEST if s.backbone in wanted]

    if args.list:
        return 0
    if args.verify:
        return 0 if _verify(todo) else 1

    results = {s.backbone: _fetch(s, args.force) for s in todo}
    pending = [s for s in todo if not results[s.backbone]]
    automated = [s for s in todo if s.source == "url"]
    failed = [s for s in automated if not results[s.backbone]]

    print(f"\n{sum(results.values())}/{len(todo)} checkpoints present.")
    if pending:
        print("\nStill needed (upstream does not allow an automated download; "
              f"save each file into {_PRETRAIN_DIR}/):")
        for s in pending:
            print(f"  - {s.filename}\n      from {s.locator}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
