"""Evaluate one experiment on Cityscapes-Foggy val at every fog density."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from configs.paths import assert_exists  # noqa: E402
from evaluation._common import (  # noqa: E402
    CS_FOGGY_BETAS,
    NUM_CLASSES,
    architecture_for_report,
    build_cs_loader,
    build_model_from_checkpoint,
    collect_checkpoints,
    evaluate,
    final_stage_checkpoint,
    load_checkpoint,
    parse_exp_name,
    repo_relative,
)


def _eval_at_all_betas(
    model: torch.nn.Module, *, crop, batch_size, num_workers, device,
    label_prefix: str, full_res: bool = True,
) -> dict:
    out = {}
    for beta in CS_FOGGY_BETAS:
        loader = build_cs_loader(beta, crop, batch_size, num_workers,
                                 full_res=full_res)
        rep = evaluate(
            model, loader, NUM_CLASSES, device,
            desc=f"{label_prefix} | beta={beta}",
        )
        print(
            f"  {label_prefix} | beta={beta:>5}: "
            f"mIoU={rep['miou_percent']:6.2f}  "
            f"pAcc={rep['pixel_acc_percent']:6.2f}  "
            f"n={rep['num_images']}"
        )
        out[f"beta_{beta}"] = rep
    return out


def evaluate_one_exp(
    exp_dir: Path,
    *,
    crop,
    batch_size: int,
    num_workers: int,
    stage_supplementary: bool,
    device: torch.device,
    full_res: bool = True,
) -> dict:
    name = exp_dir.name
    parsed = parse_exp_name(name, dataset="cityscapes")
    if parsed is None:
        raise ValueError(f"can't parse backbone/variant from {name!r}")
    backbone, variant, _ = parsed
    print(f"[exp={name}]  backbone={backbone}  variant={variant}")

    out: dict = {
        "experiment": name,
        "backbone": backbone,
        "variant": variant,
        "final": {},
        "per_stage": {},
    }

    final = final_stage_checkpoint(exp_dir)
    if final is None:
        raise FileNotFoundError(f"no checkpoint found under {exp_dir}")

    final_tag, final_path = final
    print(f"[final={final_tag}]  {final_path}")
    model, architecture, meta = build_model_from_checkpoint(
        str(final_path), expected_backbone=backbone,
        expected_variant=variant, expected_n_classes=NUM_CLASSES)
    model = model.to(device)
    out["final"]["checkpoint"] = repo_relative(final_path)
    out["final"]["checkpoint_tag"] = final_tag
    out["architecture"] = architecture_for_report(architecture)
    out["final"]["checkpoint_meta"] = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in meta.items()
        if k in ("stage", "epoch", "miou", "best")
    }
    out["final"].update(_eval_at_all_betas(
        model, crop=crop, batch_size=batch_size,
        num_workers=num_workers, device=device,
        label_prefix=f"final[{final_tag}]", full_res=full_res,
    ))

    if stage_supplementary:
        cks = collect_checkpoints(exp_dir)
        per_stage = sorted(
            k for k in cks
            if k.startswith("stage") and k.endswith("_best")
        )
        for tag in per_stage:
            print(f"[stage={tag}]  {cks[tag]}")
            load_checkpoint(
                model, str(cks[tag]), expected_backbone=backbone,
                expected_variant=variant, expected_n_classes=NUM_CLASSES,
                expected_architecture=architecture)
            model.to(device)
            out["per_stage"][tag] = {
                "checkpoint": repo_relative(cks[tag]),
            }
            out["per_stage"][tag].update(_eval_at_all_betas(
                model, crop=crop, batch_size=batch_size,
                num_workers=num_workers, device=device,
                label_prefix=tag, full_res=full_res,
            ))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate one Cityscapes-Foggy experiment "
                    "(final + per-stage) on Cityscapes val.",
    )
    p.add_argument(
        "--exp_dir", required=True,
        help="Experiment dir, e.g. output/cityscapes/<bb>_<variant>/",
    )
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--crop_eval", action="store_true",
        help="Score a centre crop of --crop_h x --crop_w instead of the whole "
             "image. Default: full-resolution validation.",
    )
    p.add_argument(
        "--stage_supplementary", action="store_true", default=True,
        help="Evaluate every per-stage checkpoint (default: ON).",
    )
    p.add_argument(
        "--no_stage_supplementary",
        dest="stage_supplementary", action="store_false",
        help="Skip per-stage checkpoints (final only).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output JSON (default: <exp_dir>/eval_cs_foggy.json).",
    )
    args = p.parse_args()
    assert_exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir).resolve()
    rep = evaluate_one_exp(
        exp_dir,
        crop=(args.crop_h, args.crop_w),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stage_supplementary=args.stage_supplementary,
        device=device,
        full_res=not args.crop_eval,
    )
    out_path = Path(args.out) if args.out else exp_dir / "eval_cs_foggy.json"
    out_path.write_text(json.dumps(rep, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
