"""Training entry point: fog curriculum, ACDC Uber or Expert, and the baseline."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dataset.loader import (MixedFogDataset, build_acdc,
                            build_cityscapes_foggy, collate,
                            build_replay_loader, make_loader,
                            make_replay_subset)
from models.lidseg import LiDSegModel
from models.loss import LiDSegLoss
from train.config import Configuration, parse_args
from train.core.checkpoint import CheckpointManager
from train.core.utils import (CheckpointFormatError, build_run_contract,
                              barrier_after_rank0_checkpoint, is_done,
                              load_resume_state, mark_done, restore_rng_state,
                              sha256_file, strict_load_state_dict,
                              build_logger, init_distributed, is_main,
                              load_full_ckpt, load_pretrained, set_seed)
from train.engine import TrainingEngine
from train.val import validate


def _format_val_metrics(m: Dict[str, float]) -> str:
    """Stable val log line (SSIM extra precision)."""
    return "  ".join(
        f"{k}={v:.4f}" if k == "ssim" else f"{k}={v:.2f}"
        for k, v in sorted(m.items()))


def _epoch_line(stage_idx: int, name: str, epoch: int, epochs: int,
                       train_stats: Dict[str, float],
                       val_m: Dict[str, float]) -> str:
    """Format an epoch summary."""
    ts = train_stats or {}
    parts = [f"Stage {stage_idx}", f"Epoch {epoch + 1}/{epochs}"]
    if ts.get("total") is not None:
        parts.append(f"Train Loss {ts['total']:.4f}")
    if ts.get("l_seg") is not None:
        parts.append(f"seg {ts['l_seg']:.4f}")
    if ts.get("l_rec") is not None:
        parts.append(f"recon {ts['l_rec']:.4f}")
    psnr = val_m.get("psnr")
    if psnr is not None:
        parts.append(f"PSNR {psnr:.2f}dB")
    parts.append(f"mIoU {val_m.get('miou/all', float('nan')):.2f}%")
    return " | ".join(parts)

class Trainer:
    def __init__(self, cfg: Configuration, rank: int, world: int, local: int):
        self.cfg = cfg
        self.rank, self.world, self.local = rank, world, local
        self.device = torch.device("cuda", local) if torch.cuda.is_available() \
            else torch.device("cpu")
        self.log = build_logger("lidseg",
                                str(cfg.output_root / "trainf.log"))
        self.log.info("DDP world=%d rank=%d local=%d device=%s",
                      world, rank, local, self.device)
        self.log.info("config:\n%s", json.dumps(cfg.cfg, indent=2))
        set_seed(cfg["seed"], rank,
                 deterministic=bool(cfg.get("deterministic", True)))
        self.run_contract = build_run_contract(
            cfg.cfg, repo_root=ROOT, world_size=world,
            config_path=cfg.config_path)
        self.architecture = self.run_contract["architecture"]
        source_audit = self.run_contract.get("audit", {}).get(
            "source_worktree", {})
        self.log.info("[provenance] contract=%s git=%s tree=%s dirty=%s",
                      self.run_contract["contract_hash"],
                      self.run_contract["source"]["git_revision"],
                      self.run_contract["source"]["tracked_tree_sha256"][:12],
                      source_audit.get("dirty", "unknown"))
        self.ckpt = CheckpointManager(
            str(cfg.output_root),
            self.log, run_contract=self.run_contract,
        )

        self.tb = None
        if cfg.get("tensorboard", False) and is_main():
            self.tb = SummaryWriter(str(cfg.output_root / "tb"))

        self._cs_train_cache: Dict[str, object] = {}

    def _build_model(self):
        """Build the guided model, or the bare backbone when guidance is off."""
        cfg = self.cfg
        architecture = self.architecture
        guidance = bool(architecture["guidance"])
        model = LiDSegModel(
            backbone_name=architecture["backbone"],
            n_classes=architecture["n_classes"],
            guidance=guidance,
            use_decoder=bool(architecture["use_decoder"]),
            guidance_channels=tuple(architecture["guidance_channels"]),
            guidance_depths=tuple(architecture["guidance_depths"]),
            backbone_kwargs=dict(architecture["backbone_kwargs"]),
            guidance_encoder_kwargs=dict(architecture["guidance_encoder_kwargs"]),
        ).to(self.device)
        return model

    def _build_criterion(self):
        cfg = self.cfg
        return LiDSegLoss(
            n_classes=cfg["n_classes"],
            ignore_index=cfg["ignore_index"],
            aux_weight=float(cfg.get("lambda_aux", 0.4)),
            seg_lovasz_weight=float(cfg.get("lambda_lovasz", 0.5)),
            use_ohem=bool(cfg.get("use_ohem", False)),
            ohem_thresh=float(cfg.get("ohem_thresh", 0.7)),
            ohem_min_kept=int(cfg.get("ohem_min_kept", 100000)),
            loss_weighting=str(cfg.get("loss_weighting", "uncertainty")),
            lambda_segmentation=float(cfg.get("lambda_segmentation", 1.0)),
            lambda_reconstruction=float(cfg.get("lambda_reconstruction", 1.0)),
        ).to(self.device)

    def _wrap_ddp(self, model):
        if self.world > 1:
            find_unused = bool(self.cfg.get("ddp_find_unused", True))
            return DDP(model, device_ids=[self.local], output_device=self.local,
                       find_unused_parameters=find_unused)
        return model

    def _build_optimizer(self, model, criterion, stage):
        cfg = self.cfg
        bb_mult = stage.get("backbone_lr_multiplier",
                            cfg.get("backbone_lr_mult", 0.1))
        guid_mult = cfg.get("guidance_lr_mult", 1.0)
        fus_mult = cfg.get("fusion_lr_mult", 1.0)
        bb, guid, fus = [], [], []
        from train.core.utils import _unwrap
        for n, p in _unwrap(model).named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("backbone."):
                bb.append(p)
            elif n.startswith("guidance."):
                guid.append(p)
            else:
                fus.append(p)
        crit_names = []
        for n, p in criterion.named_parameters():
            if not p.requires_grad:
                continue
            if (str(cfg.get("loss_weighting", "")).lower() == "fixed"
                    and n.startswith("log_var")):
                continue
            crit_names.append(p)
        params = [
            {"params": bb,   "lr": cfg["base_lr"] * bb_mult,   "lr_scale": bb_mult},
            {"params": guid, "lr": cfg["base_lr"] * guid_mult, "lr_scale": guid_mult},
            {"params": fus,  "lr": cfg["base_lr"] * fus_mult,  "lr_scale": fus_mult},
            {"params": crit_names,
             "lr": cfg["base_lr"], "lr_scale": 1.0, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(params, lr=cfg["base_lr"],
                                 weight_decay=cfg["weight_decay"])

    def _cs_train(self, beta: str):
        if beta not in self._cs_train_cache:
            self._cs_train_cache[beta] = build_cityscapes_foggy(
                self.cfg["cityscapes_root"], beta, self.cfg.crop, True, True)
        return self._cs_train_cache[beta]

    def _dataset_for_replay_source(self, src: str):
        """Replay subset source; clear weather uses ``leftImg8bit``."""
        s = str(src).lower()
        if s in ("clear", "0.000", "0.0", "0"):
            from dataset.segdatasets.cityscapes import Cityscapes
            clear_root = self.cfg.get("cityscapes_clear_root",
                                      self.cfg["cityscapes_root"])
            return Cityscapes(
                root=clear_root, split="train",
                is_train=True, crop_size=self.cfg.crop,
                img_folder="leftImg8bit", load_clear_reference=False)
        return self._cs_train(src)

    def _build_stage_train_dataset(self, stage):
        kind = stage["dataset"]
        if kind == "cityscapes_foggy":
            beta = stage["beta"]
            if str(beta).lower() in ("clear", "0.000", "0.0", "0"):
                from dataset.segdatasets.cityscapes import Cityscapes
                clear_root = self.cfg.get("cityscapes_clear_root",
                                          self.cfg["cityscapes_root"])
                return Cityscapes(
                    root=clear_root, split="train",
                    is_train=True, crop_size=self.cfg.crop,
                    img_folder="leftImg8bit", load_clear_reference=False)
            if str(beta).lower() == "mixed":
                betas = stage.get("mixed_betas") or ["0.005", "0.01", "0.02"]
                sets = [self._cs_train(b) for b in betas]
                levels = [float(b) for b in betas]
                return MixedFogDataset(sets, levels, seed=self.cfg["seed"])
            return self._cs_train(beta)
        if kind == "acdc":
            return build_acdc(self.cfg["acdc_root"], stage["conditions"],
                              self.cfg.crop, True, True)
        if kind == "mixed_fog":
            sets = [self._cs_train(b) for b in stage["betas"]]
            levels = [float(b) for b in stage["betas"]]
            return MixedFogDataset(sets, levels, seed=self.cfg["seed"])
        raise ValueError(f"unknown dataset kind: {kind}")

    def _build_stage_val_dataset(self, stage):
        kind = stage["dataset"]
        eval_full_res = bool(self.cfg.get("validation_full_res", True))
        if kind == "cityscapes_foggy":
            beta = stage["beta"]
            if str(beta).lower() in ("clear", "0.000", "0.0", "0"):
                from dataset.segdatasets.cityscapes import Cityscapes
                clear_root = self.cfg.get("cityscapes_clear_root",
                                          self.cfg["cityscapes_root"])
                return Cityscapes(
                    root=clear_root, split="val",
                    is_train=False, crop_size=self.cfg.crop,
                    img_folder="leftImg8bit", load_clear_reference=False,
                    eval_full_res=eval_full_res)
            if str(beta).lower() == "mixed":
                betas = stage.get("mixed_betas") or ["0.005", "0.01", "0.02"]
                return build_cityscapes_foggy(
                    self.cfg["cityscapes_root"], betas[-1], self.cfg.crop,
                    False, self.cfg.get("use_decoder", False),
                    eval_full_res=eval_full_res)
            return build_cityscapes_foggy(
                self.cfg["cityscapes_root"], beta, self.cfg.crop,
                False, self.cfg.get("use_decoder", False),
                eval_full_res=eval_full_res)
        if kind == "acdc":
            return build_acdc(self.cfg["acdc_root"], stage["conditions"],
                              self.cfg.crop, False,
                              self.cfg.get("use_decoder", False),
                              eval_full_res=eval_full_res)
        if kind == "mixed_fog":
            return build_cityscapes_foggy(
                self.cfg["cityscapes_root"], stage["betas"][-1], self.cfg.crop,
                False, self.cfg.get("use_decoder", False),
                eval_full_res=eval_full_res)
        raise ValueError(f"unknown dataset kind: {kind}")

    def _build_replay_loader(self, stage) -> Optional[object]:
        prev_betas = stage.get("replay_from", []) or []
        prev_acdc = stage.get("replay_acdc_from", []) or []
        ratio = stage.get("replay_ratio", 0.0) or 0.0
        if (not prev_betas and not prev_acdc) or ratio <= 0:
            return None
        cfg = self.cfg
        subsets = []
        for i, b in enumerate(prev_betas):
            subsets.append(make_replay_subset(
                self._dataset_for_replay_source(b), float(ratio),
                seed=cfg["seed"] + i))
        offset = len(prev_betas)
        for j, c in enumerate(prev_acdc):
            ds = build_acdc(cfg["acdc_root"], [c], cfg.crop, True, True)
            subsets.append(make_replay_subset(
                ds, float(ratio), seed=cfg["seed"] + offset + j))
        return build_replay_loader(
            subsets, batch_size=cfg["replay_batch_size"],
            num_workers=max(1, cfg["num_workers"] // 2), collate_fn=collate,
            shuffle=(self.world <= 1), seed=cfg["seed"])

    def _train_stage(self, stage_idx: int, stage: Dict) -> Dict[str, float]:
        cfg = self.cfg
        name = stage.get("name", f"stage{stage_idx}")
        epochs = int(stage["epochs"])
        n_blocks = len(self.cfg.stages)
        self.log.info("=" * 72)
        self.log.info(
            "[step %d/%d] %s  (epochs=%d)  |  checkpoints: best_stage_%d.pth / "
            "last_stage_%d.pth  (step index = YAML list order; first training "
            "block is often step 2 after a validate-only step 1)",
            stage_idx, n_blocks, name, epochs, stage_idx, stage_idx,
        )
        self.log.info("  output : %s", self.cfg.output_root)

        train_ds = self._build_stage_train_dataset(stage)
        val_ds = self._build_stage_val_dataset(stage)

        validate_only_stage = stage.get("validate_only", False) or epochs == 0

        if stage.get("use_curriculum_replay", False):
            raise ValueError(
                "use_curriculum_replay is retired: configure replay only via "
                "replay_ratio/replay_from so each step has one replay mechanism")

        bs = cfg["batch_size"]
        nw = cfg["num_workers"]
        if validate_only_stage:
            train_loader = None
        else:
            train_loader = make_loader(
                train_ds, bs, nw, self.world, self.rank, seed=cfg["seed"])
        val_loader = make_loader(val_ds, int(cfg.get("val_batch_size", 1)),
                                 max(1, nw // 2), self.world, self.rank,
                                 shuffle=False, drop_last=False)
        replay_loader = (None if validate_only_stage
                         else self._build_replay_loader(stage))
        if replay_loader is not None and self.world > 1:
            self.log.info("[replay] shuffle=OFF (world_size=%d — synchronized "
                          "replay batches across ranks)", self.world)

        model = self._build_model()

        stage_last = self.ckpt.stage_last_path(stage_idx)
        stage_best = self.ckpt.stage_best_path(stage_idx)
        if stage_best.is_file() and not stage_last.is_file():
            raise CheckpointFormatError(
                f"{stage_best} exists without {stage_last}; this stage has a "
                "selected checkpoint but no exact resume snapshot")
        resume_blob: Optional[Dict] = None
        initialization: Dict[str, object]
        if stage_last.exists():
            resume_blob = load_resume_state(
                stage_last, model, expected_contract=self.run_contract,
                stage_idx=stage_idx, log=self.log)
            provenance = resume_blob["provenance"]
            initialization = dict(provenance.get("initialization", {}))
            initialization["resumed_from"] = str(stage_last)
        else:
            init_kind = str(stage.get("init", "prev_stage"))
            if init_kind == "full_ckpt":
                initialization = load_full_ckpt(
                    model, cfg.get("pretrain_path"), self.log,
                    expected_architecture=self.architecture,
                    allow_decoder_drop=not bool(self.architecture["use_decoder"]),
                )
            elif init_kind == "pretrain":
                initialization = load_pretrained(model, cfg.get("pretrain_path"),
                                                  self.log)
            elif init_kind == "prev_stage":
                if stage_idx <= 1:
                    raise ValueError("stage 1 cannot use init=prev_stage")
                previous_selected = self.ckpt.best_path(stage_idx - 1)
                initialization = load_full_ckpt(
                    model, str(previous_selected), self.log,
                    expected_architecture=self.architecture,
                    expected_contract=self.run_contract,
                    expected_stage_idx=stage_idx - 1,
                    expected_kinds=("stage_best",),
                )
            elif init_kind == "scratch":
                self.log.info("[init] stage %d starts from random init", stage_idx)
                initialization = {"kind": "scratch"}
            else:
                raise ValueError(
                    f"stage {stage_idx} has unknown init={init_kind!r}")

        criterion = self._build_criterion()
        if resume_blob is not None:
            criterion_state = resume_blob.get("criterion")
            if not isinstance(criterion_state, dict):
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has no criterion state")
            strict_load_state_dict(
                criterion, criterion_state, path=stage_last,
                context="resume criterion")
            self.log.info("[resume] strictly restored criterion state")
        model = self._wrap_ddp(model)

        if validate_only_stage:
            if resume_blob is not None:
                restore_rng_state(resume_blob["rng_state"], rank=self.rank)
            self.log.info("[step %d/%d] validate-only on %s", stage_idx, n_blocks,
                          stage.get("dataset"))
            m = validate(model, val_loader, cfg["n_classes"], self.device)
            self.log.info("[step %d/%d] val %s", stage_idx, n_blocks,
                          _format_val_metrics(m))
            metrics = {"best_miou": m["miou/all"], "no_improve": 0,
                       "history": [{"epoch": -1, **m}]}
            # Write both paths on every rank so the manager can gather RNG state.
            self.ckpt.save_stage_best(
                stage_idx, model, criterion, None, -1, dict(m), cfg.cfg,
                stage_name=name, initialization=initialization)
            self.ckpt.save_stage_last(
                stage_idx, model, criterion, None, -1, metrics, cfg.cfg,
                stage_name=name, initialization=initialization)
            barrier_after_rank0_checkpoint()
            return {"best_miou": m["miou/all"]}

        optim = self._build_optimizer(model, criterion, stage)
        if resume_blob is not None:
            if "optim" not in resume_blob:
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has no optimizer state")
            try:
                optim.load_state_dict(resume_blob["optim"])
                self.log.info("[resume] optimizer state restored")
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CheckpointFormatError(
                    f"{stage_last}: optimizer restore failed: {exc}") from exc

        cfg.cfg["stage_lr_mult"] = stage.get("lr_multiplier", 1.0)
        cfg.cfg["active_recon"] = bool(stage.get(
            "recon", cfg.get("use_decoder", True)))
        cfg.cfg["replay_ratio"] = float(stage.get("replay_ratio", 0.0) or 0.0)

        total_iters = epochs * len(train_loader)
        engine = TrainingEngine(
            model, criterion, optim, cfg.cfg, train_loader, val_loader,
            self.log, self.device, total_iters,
            replay_loader=replay_loader, tb_writer=self.tb)

        if resume_blob is not None:
            scaler_state = resume_blob.get("scaler")
            if not isinstance(scaler_state, dict):
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has no GradScaler state")
            try:
                engine.scaler.load_state_dict(scaler_state)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CheckpointFormatError(
                    f"{stage_last}: GradScaler restore failed: {exc}") from exc
            engine_state = resume_blob.get("engine_state")
            if not isinstance(engine_state, dict):
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has no engine state")
            try:
                engine.load_state_dict(engine_state)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CheckpointFormatError(
                    f"{stage_last}: TrainingEngine restore failed: {exc}") from exc
            restore_rng_state(resume_blob["rng_state"], rank=self.rank)
            try:
                start_ep = int(resume_blob["epoch"]) + 1
            except (KeyError, TypeError, ValueError) as exc:
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has an invalid epoch") from exc
            saved_metrics = resume_blob.get("metrics")
            if not isinstance(saved_metrics, dict):
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has invalid metrics")
            try:
                best_miou = float(saved_metrics["best_miou"])
                no_improve = int(saved_metrics["no_improve"])
                history = list(saved_metrics["history"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint lacks stage progress") from exc
        else:
            start_ep = 0
            best_miou = float("-inf")
            no_improve = 0
            history: List[Dict] = []
        patience = int(cfg.get("early_stop_patience", 10))
        ve = int(cfg["val_every"])
        self.log.info(
            "[step %d/%d] early-stop: patience=%d val_every=%d "
            "(no_improve ticks each validation); resume start_ep=%d "
            "best_miou=%.4f no_improve=%d",
            stage_idx, n_blocks, patience, ve, start_ep, best_miou, no_improve)
        if start_ep >= epochs:
            if not self.ckpt.stage_best_path(stage_idx).is_file():
                raise CheckpointFormatError(
                    f"{stage_last}: completed stage has no selected "
                    f"best_stage_{stage_idx}.pth checkpoint")
            self.log.info("[step %d/%d] already at epoch %d/%d — skipping",
                          stage_idx, n_blocks, start_ep, epochs)
            return {"best_miou": best_miou}
        if start_ep > 0:
            saved_iteration = resume_blob.get("iteration") if resume_blob else None
            if isinstance(saved_iteration, bool) or not isinstance(saved_iteration, int):
                raise CheckpointFormatError(
                    f"{stage_last}: resume checkpoint has no valid iteration")
            expected_iteration = start_ep * len(train_loader)
            if saved_iteration != expected_iteration:
                raise CheckpointFormatError(
                    f"{stage_last}: iteration={saved_iteration} does not match "
                    f"epoch boundary {expected_iteration}; refuse non-exact resume")
            if engine._iter != saved_iteration:
                raise CheckpointFormatError(
                    f"{stage_last}: engine iteration={engine._iter} disagrees "
                    f"with top-level iteration={saved_iteration}")
            self.log.info("[resume] engine iteration restored to %d "
                          "(stage %d, epoch %d/%d)",
                          engine._iter, stage_idx, start_ep, epochs)
        for epoch in range(start_ep, epochs):
            epoch_stats = engine.run_epoch(epoch)
            if (epoch + 1) % cfg["val_every"] == 0:
                barrier_after_rank0_checkpoint()
                m = validate(model, val_loader, cfg["n_classes"], self.device)
                self.log.info("[step %d/%d] val ep %d  %s", stage_idx, n_blocks,
                              epoch, _format_val_metrics(m))
                self.log.info(_epoch_line(stage_idx, name, epoch, epochs,
                                                 epoch_stats, m))
                history.append({"epoch": epoch, **m})
                if self.tb is not None:
                    for k, v in m.items():
                        self.tb.add_scalar(f"val/stage{stage_idx}/{k}", v,
                                           epoch)
                if m["miou/all"] > best_miou:
                    best_miou = m["miou/all"]
                    no_improve = 0
                    self.ckpt.save_stage_best(
                        stage_idx, model, criterion, optim, epoch, m, cfg.cfg,
                        stage_name=name, initialization=initialization,
                        scaler=engine.scaler, iteration=engine._iter)
                else:
                    no_improve += 1
            self.ckpt.save_stage_last(
                stage_idx, model, criterion, optim, epoch,
                {"best_miou": best_miou, "no_improve": no_improve,
                 "history": history}, cfg.cfg,
                stage_name=name, initialization=initialization,
                scaler=engine.scaler, iteration=engine._iter,
                engine_state=engine.state_dict())
            barrier_after_rank0_checkpoint()
            if no_improve >= patience:
                self.log.info(
                    "[step %d/%d] early-stop @ ep %d (no improve after %d "
                    "validation checks, val_every=%d; best=%.2f)",
                    stage_idx, n_blocks, epoch, no_improve, ve, best_miou)
                break
        if not self.ckpt.stage_best_path(stage_idx).is_file():
            raise RuntimeError(
                f"stage {stage_idx} completed without a selected checkpoint; "
                "ensure val_every produces at least one validation")
        self.log.info("[step %d/%d] done. best mIoU = %.2f",
                      stage_idx, n_blocks, best_miou)
        return {"best_miou": best_miou}

    def run(self) -> None:
        only = int(self.cfg.args.only_stage)
        if is_done(self.cfg.output_root):
            if only:
                raise RuntimeError(
                    "--only_stage refuses to mutate a completed run; use a new "
                    "output_root or explicitly create a new experiment")
            manifest_path = self.cfg.output_root / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"{self.cfg.output_root}/DONE exists without manifest.json; "
                    "completion evidence is corrupt")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{manifest_path} cannot be read as completion evidence: {exc}") from exc
            if not isinstance(manifest, dict) or manifest.get("complete") is not True:
                raise RuntimeError(
                    f"{manifest_path} is not a complete lifecycle manifest")
            self.log.info("[skip] %s/DONE exists — training already completed",
                          self.cfg.output_root)
            return
        manifest_path = self.cfg.output_root / "manifest.json"
        if manifest_path.is_file():
            raise RuntimeError(
                "output contains manifest.json without DONE; completion state is "
                "ambiguous, so use a fresh output_root")
        stale_aliases = [
            path for path in (self.cfg.output_root / "best.pth",
                              self.cfg.output_root / "last.pth",
                              self.cfg.output_root / "checkpoint.pth")
            if path.is_file()
        ]
        if stale_aliases:
            raise RuntimeError(
                "output contains stale global checkpoint aliases without DONE; refuse to mix an "
                f"incomplete run with {', '.join(str(path) for path in stale_aliases)}")
        if is_main():
            n = len(self.cfg.stages)
            self.log.info("[curriculum] %d blocks", n)
            for i, s in enumerate(self.cfg.stages, start=1):
                ep = int(s.get("epochs", 0))
                vo = bool(s.get("validate_only", False)) or ep == 0
                tag = "validate-only" if vo else "train"
                self.log.info("  %d/%d [%s] %s", i, n, tag,
                              s.get("name", f"block{i}"))
        stage_results = {}
        for i, stage in enumerate(self.cfg.stages, start=1):
            if only and i != only:
                self.log.info("skipping stage %d (only_stage=%d)", i, only)
                continue
            stage_results[i] = self._train_stage(i, stage)

        if only:
            self.log.info(
                "[partial] stage %d completed; no aliases, manifest, or DONE "
                "marker were published", only)
            if self.tb is not None:
                self.tb.close()
            return

        trainable_stages = [
            index for index, stage in enumerate(self.cfg.stages, start=1)
            if int(stage.get("epochs", 0)) > 0
            and not bool(stage.get("validate_only", False))
        ]
        if not trainable_stages:
            raise RuntimeError(
                "a complete run must contain at least one trainable stage; "
                "use --only_stage for validation-only diagnostics")
        final_stage_idx = trainable_stages[-1]
        if is_main():
            aliases = self.ckpt.publish_final_stage_aliases(final_stage_idx)
        else:
            aliases = {}
        barrier_after_rank0_checkpoint()
        if is_main():
            alias_metadata = {
                alias_name: {
                    "path": alias_path,
                    "sha256": sha256_file(alias_path),
                }
                for alias_name, alias_path in aliases.items()
            }
            manifest = {
                "schema_version": 1,
                "complete": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "architecture": self.architecture,
                "contract_hash": self.run_contract["contract_hash"],
                "final_stage": {
                    "idx": final_stage_idx,
                    "name": self.cfg.stages[final_stage_idx - 1].get(
                        "name", f"stage{final_stage_idx}"),
                    "selected": str(
                        self.ckpt.stage_best_path(final_stage_idx)),
                    "resumable": str(
                        self.ckpt.stage_last_path(final_stage_idx)),
                },
                "final_aliases": alias_metadata,
                "stage_ids": list(range(1, len(self.cfg.stages) + 1)),
                "stages": {
                    str(i): {
                        "name": s.get("name", f"stage{i}"),
                        "best": str(self.ckpt.stage_best_path(i)),
                        "last": str(self.ckpt.stage_last_path(i)),
                        **stage_results.get(i, {}),
                    }
                    for i, s in enumerate(self.cfg.stages, start=1)
                    if self.ckpt.stage_best_path(i).is_file()
                    and self.ckpt.stage_last_path(i).is_file()
                },
            }
            self.ckpt.write_manifest(manifest)
            self.log.info("complete manifest written.")
            mark_done(self.cfg.output_root)
        barrier_after_rank0_checkpoint()
        if self.tb is not None:
            self.tb.close()


def main() -> None:
    args = parse_args()
    cfg = Configuration(args)
    if bool(cfg.get("deterministic", True)):
        # Set before torchrun selects a device.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world, local = init_distributed()
    if torch.cuda.is_available():
        allow_tf32 = bool(cfg.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    Trainer(cfg, rank, world, local).run()


if __name__ == "__main__":
    main()
