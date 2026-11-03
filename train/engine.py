"""TrainingEngine: per-iter step + per-epoch loop with AMP, replay-batch, grad norms."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from train.core.recon_metrics import recon_psnr_ssim_batch
from train.core.utils import _unwrap


def _safe_f32(t):
    """Cast a float tensor to fp32 and clamp."""
    if isinstance(t, torch.Tensor) and t.is_floating_point():
        return t.float().clamp(-50, 50)
    return t


class TrainingEngine:
    """Runs one stage's training. Persistent GradScaler across epochs."""

    GRAD_GROUPS = ("backbone", "guidance_enc", "guidance_dec", "lgf",
                   "criterion")

    def __init__(self, model, criterion, optim, cfg, train_loader, val_loader,
                 log, device, total_iters: int,
                 replay_loader: Optional[DataLoader] = None,
                 tb_writer=None):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.replay_loader = replay_loader
        self._replay_iter = None
        self._replay_buffer: Optional[Dict[str, Any]] = None
        self._replay_current_rows = 0
        self._replay_emitted_rows = 0
        replay_semantics = str(
            cfg.get("replay_ratio_semantics", "total_sample_fraction"))
        if replay_semantics != "total_sample_fraction":
            raise ValueError(
                "replay_ratio_semantics must be 'total_sample_fraction', got "
                f"{replay_semantics!r}")
        self._replay_ratio = float(cfg.get("replay_ratio", 0.0) or 0.0)
        if not 0.0 <= self._replay_ratio < 1.0:
            raise ValueError(
                "replay_ratio must be a fraction in [0, 1), got "
                f"{self._replay_ratio!r}")
        if self.replay_loader is None and self._replay_ratio > 0.0:
            raise ValueError(
                f"replay_ratio={self._replay_ratio} requires a replay_loader")
        if self.replay_loader is not None and self._replay_ratio <= 0.0:
            raise ValueError(
                "a replay_loader was supplied but replay_ratio is zero; "
                "refusing an unproven replay mixture")
        self.log = log
        self.device = device
        self.total_iters = max(1, total_iters)
        self.tb = tb_writer
        self.scaler = GradScaler("cuda", enabled=cfg["amp"], init_scale=2 ** 12)
        self._iter = 0
        self._accum_count = 0
        self._amp_skip_streak = 0

    def _reset_replay_epoch(self, epoch: int) -> None:
        """Restart the replay iterator at an epoch boundary."""
        if self.replay_loader is None:
            return
        replay_seed = getattr(self.replay_loader, "_lidseg_replay_seed", None)
        if replay_seed is not None:
            if isinstance(replay_seed, bool) or not isinstance(replay_seed, int):
                raise RuntimeError("replay loader has an invalid deterministic seed")
            generator = getattr(self.replay_loader, "generator", None)
            if not isinstance(generator, torch.Generator):
                raise RuntimeError(
                    "replay loader declares a deterministic seed without a "
                    "torch.Generator")
            generator.manual_seed(replay_seed + epoch)
        sampler = getattr(self.replay_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self._replay_iter = None
        self._replay_buffer = None

    def _reset_train_epoch(self, epoch: int) -> None:
        """Set the exact sampler and worker-RNG boundary for one train epoch."""
        train_seed = getattr(self.train_loader, "_lidseg_train_seed", None)
        if train_seed is not None:
            if isinstance(train_seed, bool) or not isinstance(train_seed, int):
                raise RuntimeError("training loader has an invalid deterministic seed")
            generator = getattr(self.train_loader, "generator", None)
            if not isinstance(generator, torch.Generator):
                raise RuntimeError(
                    "training loader declares a deterministic seed without a "
                    "torch.Generator")
            generator.manual_seed(train_seed + epoch)
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = getattr(self.train_loader, "dataset", None)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        elif hasattr(ds, "refresh_sampling"):
            # In-tree stochastic datasets implement set_epoch.
            ds.refresh_sampling()

    def _discard_replay_stream(self) -> None:
        """Drop the replay iterator and buffer."""
        self._replay_iter = None
        self._replay_buffer = None

    def state_dict(self) -> Dict[str, Any]:
        """Serializable epoch-boundary state."""
        if self._accum_count != 0:
            raise RuntimeError(
                "TrainingEngine state can only be checkpointed at an optimizer "
                "boundary (accum_count must be zero)")
        if self._replay_buffer is not None:
            raise RuntimeError(
                "TrainingEngine state can only be checkpointed after replay "
                "buffer reset at an epoch boundary")
        return {
            "schema_version": 1,
            "iteration": int(self._iter),
            "replay_current_rows": int(self._replay_current_rows),
            "replay_emitted_rows": int(self._replay_emitted_rows),
            "replay_ratio": float(self._replay_ratio),
            "replay_ratio_semantics": "total_sample_fraction",
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the deterministic scheduler portion of an epoch checkpoint."""
        if not isinstance(state, Mapping):
            raise TypeError("TrainingEngine state must be a mapping")
        if state.get("schema_version") != 1:
            raise RuntimeError(
                f"unsupported TrainingEngine state schema: "
                f"{state.get('schema_version')!r}")
        if state.get("replay_ratio_semantics") != "total_sample_fraction":
            raise RuntimeError("TrainingEngine state has incompatible replay semantics")
        saved_ratio = state.get("replay_ratio")
        if not isinstance(saved_ratio, (int, float)) or isinstance(saved_ratio, bool):
            raise RuntimeError("TrainingEngine state has invalid replay_ratio")
        if not math.isclose(float(saved_ratio), self._replay_ratio,
                            rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                "TrainingEngine state replay_ratio differs from active stage: "
                f"checkpoint={saved_ratio}, active={self._replay_ratio}")
        values: Dict[str, int] = {}
        for key in ("iteration", "replay_current_rows", "replay_emitted_rows"):
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"TrainingEngine state has invalid {key}: {value!r}")
            values[key] = int(value)
        expected_emitted = int(math.floor(
            values["replay_current_rows"] * self._replay_ratio
            / (1.0 - self._replay_ratio) + 1e-12))
        if values["replay_emitted_rows"] != expected_emitted:
            raise RuntimeError(
                "TrainingEngine state replay counters are inconsistent with the "
                "configured total-sample replay fraction")
        if self.replay_loader is None and (
                values["replay_current_rows"] or values["replay_emitted_rows"]):
            raise RuntimeError(
                "TrainingEngine state has replay counters but active stage has no "
                "replay loader")
        self._iter = values["iteration"]
        self._replay_current_rows = values["replay_current_rows"]
        self._replay_emitted_rows = values["replay_emitted_rows"]
        self._accum_count = 0
        self._replay_iter = None
        self._replay_buffer = None

    def _set_lr(self) -> float:
        """Warmup, then decay to ``min_lr_ratio`` of the base learning rate."""
        it = self._iter
        warm = self.cfg["warmup_iters"]
        if it < warm:
            scale = (it + 1) / max(1, warm)
        else:
            t = (it - warm) / max(1, self.total_iters - warm)
            t = min(max(t, 0.0), 1.0)
            sched = str(self.cfg.get("sched", "cosine")).lower()
            if sched == "poly":
                scale = (1.0 - t) ** self.cfg.get("poly_power", 0.9)
            elif sched == "cosine":
                floor = float(self.cfg.get("min_lr_ratio", 0.0))
                scale = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))
            else:
                raise ValueError(
                    f"unknown sched {sched!r}; expected 'cosine' or 'poly'")
        lr = self.cfg["base_lr"] * scale * self.cfg.get("stage_lr_mult", 1.0)
        for g in self.optim.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        return lr

    def _next_replay(self) -> Dict[str, Any]:
        if self.replay_loader is None:
            raise RuntimeError("_next_replay called without a replay_loader")
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        try:
            return next(self._replay_iter)
        except StopIteration:
            self._replay_iter = iter(self.replay_loader)
            try:
                return next(self._replay_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "replay_loader yielded no batches; check replay subset size "
                    "and replay_batch_size") from exc

    @staticmethod
    def _batch_rows(batch: Dict[str, Any], *, source: str) -> int:
        image = batch.get("image")
        if not isinstance(image, torch.Tensor) or image.ndim < 1:
            raise TypeError(
                f"{source} batch must contain a batched tensor 'image'")
        n = int(image.shape[0])
        if n < 1:
            raise RuntimeError(f"{source} batch is empty")
        semantic = batch.get("semantic")
        if not isinstance(semantic, torch.Tensor) or semantic.shape[0] != n:
            raise RuntimeError(
                f"{source} batch must contain 'semantic' with {n} rows")
        return n

    @staticmethod
    def _slice_batch(batch: Dict[str, Any], start: int,
                     stop: int) -> Dict[str, Any]:
        """Take rows from a collated batch without losing list metadata."""
        if not 0 <= start < stop:
            raise ValueError(f"invalid batch slice [{start}:{stop}]")
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value[start:stop]
            elif isinstance(value, np.ndarray):
                out[key] = value[start:stop]
            elif isinstance(value, list):
                out[key] = value[start:stop]
            elif isinstance(value, tuple):
                out[key] = value[start:stop]
            else:
                raise TypeError(
                    f"cannot slice replay batch field {key!r} of type "
                    f"{type(value).__name__}")
        return out

    def _requested_replay_rows(self, current_rows: int) -> int:
        """Integer replay rows for this batch, tracking the cumulative target fraction."""
        if current_rows < 1:
            raise ValueError(f"current_rows must be positive, got {current_rows}")
        if self.replay_loader is None:
            return 0
        # R = C*r/(1-r), floored against the cumulative target.
        self._replay_current_rows += int(current_rows)
        target = (self._replay_current_rows * self._replay_ratio
                  / (1.0 - self._replay_ratio))
        desired_total = int(math.floor(target + 1e-12))
        requested = desired_total - self._replay_emitted_rows
        if requested < 0:
            raise RuntimeError(
                "replay scheduler regressed; cumulative target must be "
                "monotonic")
        self._replay_emitted_rows += requested
        return requested

    def _take_replay_chunks(self, n_rows: int) -> List[Dict[str, Any]]:
        """Read exactly ``n_rows`` replay rows, retaining excess loader rows."""
        if n_rows < 0:
            raise ValueError(f"n_rows must be non-negative, got {n_rows}")
        chunks: List[Dict[str, Any]] = []
        remaining = n_rows
        while remaining:
            if self._replay_buffer is None:
                self._replay_buffer = self._next_replay()
            available = self._batch_rows(self._replay_buffer, source="replay")
            take = min(available, remaining)
            chunks.append(self._slice_batch(self._replay_buffer, 0, take))
            if take == available:
                self._replay_buffer = None
            else:
                self._replay_buffer = self._slice_batch(
                    self._replay_buffer, take, available)
            remaining -= take
        return chunks

    def _device_batch_tensors(
        self, batch: Dict[str, Any], *, source: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Move a collated batch's model tensors to device."""
        n_rows = self._batch_rows(batch, source=source)
        image = batch["image"].to(self.device, non_blocking=True)
        target = batch["semantic"].to(self.device, non_blocking=True)
        clear = batch.get("clear_image")
        paired = batch.get("has_paired_clear")
        if clear is not None:
            if not isinstance(clear, torch.Tensor):
                raise TypeError(f"{source} clear_image must be a tensor")
            if paired is None:
                raise RuntimeError(
                    f"{source} supplied clear_image without explicit "
                    "has_paired_clear provenance")
            clear = clear.to(self.device, non_blocking=True)
            if clear.shape != image.shape:
                raise RuntimeError(
                    f"{source} image/clear_image shapes differ: "
                    f"{tuple(image.shape)} vs {tuple(clear.shape)}")
        if paired is None:
            paired_t = torch.zeros(n_rows, dtype=torch.bool, device=self.device)
        else:
            if not isinstance(paired, torch.Tensor):
                paired = torch.as_tensor(paired)
            paired_t = paired.to(self.device, non_blocking=True).bool().reshape(-1)
            if paired_t.numel() != n_rows:
                raise RuntimeError(
                    f"{source} has_paired_clear contains {paired_t.numel()} rows, "
                    f"expected {n_rows}")
        if bool(paired_t.any()) and clear is None:
            raise RuntimeError(
                f"{source} claims paired clear rows but did not supply clear_image")
        return image, target, clear, paired_t

    def _merge_current_and_replay(
        self, batch: Dict[str, Any], replay_chunks: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Merge current and replay sources while retaining exact pair masks."""
        sources = [("current", batch)] + [
            (f"replay[{i}]", chunk) for i, chunk in enumerate(replay_chunks)
        ]
        images: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        clears: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        any_paired = False
        for source, source_batch in sources:
            image, target, clear, paired = self._device_batch_tensors(
                source_batch, source=source)
            images.append(image)
            targets.append(target)
            # Shape-compatible placeholder for an unpaired row, masked false.
            clears.append(image if clear is None else clear)
            masks.append(paired)
            any_paired = any_paired or bool(paired.any())
        clear_out = torch.cat(clears, 0) if any_paired else None
        return (torch.cat(images, 0), torch.cat(targets, 0), clear_out,
                torch.cat(masks, 0))

    def _compute_component_grad_norms(self) -> Dict[str, float]:
        raw = _unwrap(self.model)
        sums = {g: 0.0 for g in self.GRAD_GROUPS}
        for n, p in raw.named_parameters():
            if p.grad is None:
                continue
            g_sq = float(p.grad.detach().pow(2).sum().item())
            if n.startswith("backbone."):
                sums["backbone"] += g_sq
            elif n.startswith("guidance."):
                sums["guidance_enc"] += g_sq
            elif n.startswith("decoder."):
                sums["guidance_dec"] += g_sq
            elif n.startswith("lgf"):
                sums["lgf"] += g_sq
        for n, p in self.criterion.named_parameters():
            if p.grad is None:
                continue
            sums["criterion"] += float(p.grad.detach().pow(2).sum().item())
        return {f"grad/{k}": v ** 0.5 for k, v in sums.items()}

    def step(self, batch: Dict, *, force_sync: bool = False) -> Dict[str, float]:
        current_rows = self._batch_rows(batch, source="current")
        replay_rows = self._requested_replay_rows(current_rows)
        replay_chunks = self._take_replay_chunks(replay_rows) if replay_rows else []
        img, tgt, clear, has_paired_clear = self._merge_current_and_replay(
            batch, replay_chunks)

        lr = self._set_lr()
        grad_accum = max(1, int(self.cfg.get("grad_accum", 1)))
        if self._accum_count == 0:
            self.optim.zero_grad(set_to_none=True)
        should_step = force_sync or (self._accum_count + 1 >= grad_accum)
        if (not should_step) and hasattr(self.model, "no_sync"):
            _ctx = self.model.no_sync()
        else:
            _ctx = nullcontext()
        with _ctx:
            # no_sync wraps both forward and backward.
            with autocast("cuda", enabled=self.cfg["amp"]):
                out = self.model(img, x_clear=clear)

            logits = _safe_f32(out["logits"])
            aux_logits = out.get("aux_logits")
            if aux_logits:
                aux_logits = [_safe_f32(a) for a in aux_logits]
            recon = out.get("recon")
            if recon is not None:
                recon = recon.float()

            losses = self.criterion(
                logits=logits, target=tgt, aux_logits=aux_logits,
                recon=recon, clear=clear,
                has_paired_clear=has_paired_clear,
                active_recon=bool(self.cfg.get("active_recon", True)),
            )
            if not torch.isfinite(losses["total"]).all():
                raise FloatingPointError(
                    f"non-finite training loss at iteration {self._iter}: "
                    f"{float(losses['total'].detach())}")
            total_scaled = losses["total"] / grad_accum
            self.scaler.scale(total_scaled).backward()
        self._accum_count += 1

        log_d = {k: float(v.detach()) for k, v in losses.items()
                 if torch.is_tensor(v) and v.numel() == 1}
        log_d["lr"] = lr
        log_d["replay/current_rows"] = float(current_rows)
        log_d["replay/rows"] = float(replay_rows)
        total_rows = current_rows + replay_rows
        log_d["replay/fraction_batch"] = float(replay_rows / total_rows)
        mixed_total = self._replay_current_rows + self._replay_emitted_rows
        log_d["replay/fraction_cumulative"] = float(
            self._replay_emitted_rows / mixed_total) if mixed_total else 0.0

        if bool(self.cfg.get("active_recon", True)) and recon is not None \
                and clear is not None:
            rp, rs = recon_psnr_ssim_batch(recon, clear, has_paired_clear)
            if rp is not None:
                log_d["recon_psnr"] = rp
            if rs is not None:
                log_d["recon_ssim"] = rs

        if should_step:
            # The criterion's uncertainty weights sit outside DDP; average their grads
            # like DDP does for the model, or ranks clip and step differently and drift.
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                for param in self.criterion.parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            self.scaler.unscale_(self.optim)
            # The short final group was already divided by grad_accum.
            if self._accum_count != grad_accum:
                correction = grad_accum / self._accum_count
                for param in (list(self.model.parameters())
                              + list(self.criterion.parameters())):
                    if param.grad is not None:
                        param.grad.mul_(correction)
            if self.cfg.get("grad_log_every", 0) > 0 and \
                    (self._iter + 1) % self.cfg["grad_log_every"] == 0:
                log_d.update(self._compute_component_grad_norms())
            cap = float(self.cfg.get("grad_clip", 0.0))
            all_params = list(self.model.parameters()) + list(self.criterion.parameters())
            # Under AMP an fp16 overflow is routine: unscale_ records it and
            # scaler.step skips the update, so only non-AMP runs fail here.
            amp_on = self.scaler.is_enabled()
            gn = torch.nn.utils.clip_grad_norm_(
                all_params, cap if cap > 0 else float("inf"),
                error_if_nonfinite=not amp_on)
            log_d["grad_norm"] = float(gn)
            scale_before = self.scaler.get_scale() if amp_on else None
            self.scaler.step(self.optim)
            self.scaler.update()
            if amp_on:
                skipped = self.scaler.get_scale() < scale_before
                self._amp_skip_streak = self._amp_skip_streak + 1 if skipped else 0
                log_d["amp/skipped_step"] = float(skipped)
                if self._amp_skip_streak >= 20:
                    raise FloatingPointError(
                        f"20 consecutive AMP overflow skips at iteration {self._iter} "
                        f"(scale={self.scaler.get_scale():.3g}); training has diverged")
            self._accum_count = 0
        return log_d

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        self._reset_train_epoch(epoch)
        # Restart replay at the epoch boundary.
        self._reset_replay_epoch(epoch)
        self.model.train()
        t0 = time.time()
        running = defaultdict(float)
        count = 0
        recon_sum = {"psnr": 0.0, "ssim": 0.0, "n": 0}
        n_batches = len(self.train_loader)
        if n_batches < 1:
            raise RuntimeError("training loader yielded no batches")
        grad_accum = max(1, int(self.cfg.get("grad_accum", 1)))
        try:
            for batch_index, batch in enumerate(self.train_loader):
                final_partial = (
                    batch_index + 1 == n_batches
                    and self._accum_count + 1 < grad_accum
                )
                stats = self.step(batch, force_sync=final_partial)
                rp = stats.pop("recon_psnr", None)
                rs = stats.pop("recon_ssim", None)
                if rp is not None and rs is not None:
                    recon_sum["psnr"] += float(rp)
                    recon_sum["ssim"] += float(rs)
                    recon_sum["n"] += 1
                for k, v in stats.items():
                    running[k] += float(v)
                count += 1
                self._iter += 1
                if self._iter % self.cfg["log_every"] == 0:
                    avg = {k: v / count for k, v in running.items()}
                    if recon_sum["n"] > 0:
                        avg["recon_psnr"] = recon_sum["psnr"] / recon_sum["n"]
                        avg["recon_ssim"] = recon_sum["ssim"] / recon_sum["n"]
                    parts = []
                    for k, v in sorted(avg.items()):
                        if k in ("recon_psnr",):
                            parts.append(f"{k}={v:.2f}")
                        elif k in ("recon_ssim",):
                            parts.append(f"{k}={v:.4f}")
                        else:
                            parts.append(f"{k}={v:.4f}")
                    self.log.info("ep %d it %d  %s", epoch, self._iter,
                                  "  ".join(parts))
                    if self.tb is not None:
                        for k, v in avg.items():
                            self.tb.add_scalar(f"train/{k}", v, self._iter)
            self.log.info("epoch %d done in %.1fs", epoch, time.time() - t0)
            return {k: v / max(1, count) for k, v in running.items()}
        finally:
            # Drop iterator and buffer state before the stage checkpoint.
            self._discard_replay_stream()
