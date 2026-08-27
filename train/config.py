"""Configuration loader: parses YAML + CLI args, resolves pretrain path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from configs.paths import load as load_paths
from models.segmodels.pretrained import resolve_pretrained

VARIANT_SPEC: Dict[str, tuple] = {
    "baseline": (False, False),
    "guided": (True, True),
    "norecon": (True, False),
}


def parse_args() -> argparse.Namespace:
    """CLI for the guided LiDSeg curriculum trainer."""
    p = argparse.ArgumentParser(description="LiDSegModel curriculum trainer")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output_root", required=True, type=str)
    p.add_argument("--only_stage", type=int, default=0,
                   help="0 = all stages; 1..N = run only that stage")
    p.add_argument("--backbone", default=None,
                   help="override cfg.backbone (used by sweep launcher)")
    p.add_argument("--exp_name", default=None,
                   help="override cfg.exp_name (used by sweep launcher)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override batch_size from YAML (per-GPU tuning)")
    p.add_argument("--variant", choices=sorted(VARIANT_SPEC), default=None,
                   help="override cfg.guidance/use_decoder/lambda_reconstruction")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="override the ACDC conditions of every stage that sets them")
    return p.parse_args()


class Configuration:
    """Holds the YAML config dict + resolved pretrain path + stage list."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        with self.config_path.open(encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)
        if not isinstance(self.cfg, dict):
            raise ValueError("config must be a YAML mapping")

        for key in ("backbone", "exp_name"):
            v = getattr(args, key, None)
            if v:
                self.cfg[key] = v

        if getattr(args, "batch_size", None):
            self.cfg["batch_size"] = args.batch_size

        variant = getattr(args, "variant", None)
        if variant:
            guidance, use_decoder = VARIANT_SPEC[variant]
            self.cfg["guidance"] = guidance
            self.cfg["use_decoder"] = use_decoder
            self.cfg["lambda_reconstruction"] = 1.0 if use_decoder else 0.0

        conditions = getattr(args, "conditions", None)
        if conditions:
            for stage in self.cfg.get("stages") or []:
                if isinstance(stage, dict) and "conditions" in stage:
                    stage["conditions"] = list(conditions)

        self.cfg.setdefault("deterministic", True)
        self.cfg.setdefault("validation_full_res", True)
        self.cfg.setdefault("val_batch_size", 1)

        if "stages" not in self.cfg or not isinstance(self.cfg["stages"], list):
            raise ValueError("config must contain a top-level 'stages:' list")
        self.stages: List[Dict[str, Any]] = self.cfg["stages"]
        self._validate_lifecycle_config()

        self._resolve_dataset_roots()

        pretrain_from = os.environ.get("PRETRAIN_FROM", "").strip()
        if pretrain_from:
            self.cfg["pretrain_path"] = pretrain_from
        elif not self.cfg.get("pretrain_path"):
            self.cfg["pretrain_path"] = resolve_pretrained(self.cfg["backbone"])

        self.output_root = Path(args.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    _ROOT_KEYS = {
        "cityscapes_root": "CITYSCAPES_ROOT",
        "cityscapes_clear_root": "CITYSCAPES_CLEAR_ROOT",
        "acdc_root": "ACDC_ROOT",
    }

    def _resolve_dataset_roots(self) -> None:
        """Fill dataset roots from configs/paths.env unless the YAML overrides them."""
        env = None
        for cfg_key, env_key in self._ROOT_KEYS.items():
            value = self.cfg.get(cfg_key)
            if value and "__PATH__" not in str(value):
                continue
            if env is None:
                env = load_paths()
            self.cfg[cfg_key] = env[env_key]

    def _validate_lifecycle_config(self) -> None:
        if not self.stages:
            raise ValueError("config stages list must not be empty")
        if bool(self.cfg.get("use_decoder", True)) and not bool(
                self.cfg.get("guidance", True)):
            raise ValueError("use_decoder=true requires guidance=true")
        try:
            val_batch_size = int(self.cfg["val_batch_size"])
        except (TypeError, ValueError) as exc:
            raise ValueError("val_batch_size must be a positive integer") from exc
        if val_batch_size < 1:
            raise ValueError("val_batch_size must be a positive integer")

        known_init = {"pretrain", "full_ckpt", "prev_stage", "scratch"}
        for index, stage in enumerate(self.stages, start=1):
            if not isinstance(stage, dict):
                raise ValueError(f"stage {index} must be a mapping")
            init_kind = str(stage.get("init", "prev_stage"))
            if init_kind not in known_init:
                raise ValueError(
                    f"stage {index} has unknown init={init_kind!r}; expected one "
                    f"of {sorted(known_init)}")
            if index == 1 and init_kind == "prev_stage":
                raise ValueError("stage 1 cannot use init=prev_stage")
            if "epochs" not in stage:
                raise ValueError(f"stage {index} is missing required 'epochs'")
            try:
                epochs = int(stage["epochs"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stage {index} epochs must be an integer") from exc
            if epochs < 0:
                raise ValueError(f"stage {index} epochs must be non-negative")

        only_stage = int(getattr(self.args, "only_stage", 0))
        if only_stage < 0 or only_stage > len(self.stages):
            raise ValueError(
                f"--only_stage must be 0 (all stages) or 1..{len(self.stages)}; "
                f"got {only_stage}")

    @property
    def crop(self):
        return (self.cfg["crop_h"], self.cfg["crop_w"])

    def __getitem__(self, k):
        return self.cfg[k]

    def get(self, k, default=None):
        return self.cfg.get(k, default)
