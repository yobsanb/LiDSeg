"""Atomic, self-describing stage checkpoints and final-stage aliases."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from train.core.torch_io import torch_load_checkpoint
from train.core.utils import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointFormatError,
    capture_rng_states_by_rank,
    contract_hash,
    is_main,
    sha256_file,
    unwrap_trainable_module,
    validate_checkpoint_provenance,
)


class CheckpointManager:
    """Per-stage checkpoints and final-stage aliases."""

    def __init__(self, output_root: str, log, *, run_contract: Mapping[str, Any]):
        self.root = Path(output_root)
        self.log = log
        self.root.mkdir(parents=True, exist_ok=True)
        self._global_best_path = self.root / "best.pth"
        self._global_last_path = self.root / "last.pth"
        self.run_contract = copy.deepcopy(dict(run_contract))
        expected_hash = self.run_contract.get("contract_hash")
        if not isinstance(expected_hash, str) or expected_hash != contract_hash(self.run_contract):
            raise ValueError("run_contract is missing or has an invalid contract_hash")
        if not isinstance(self.run_contract.get("architecture"), Mapping):
            raise ValueError("run_contract is missing architecture metadata")

    @staticmethod
    def _atomic_save(blob: Mapping[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(blob), tmp)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _build_blob(
        self,
        model,
        criterion,
        optim,
        epoch: int,
        metrics,
        cfg,
        *,
        checkpoint_kind: str,
        stage_idx: int,
        stage_name: str,
        initialization: Optional[Mapping[str, Any]],
        scaler=None,
        rng_state: Mapping[str, Any],
        iteration: Optional[int] = None,
        engine_state: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        provenance = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "contract_hash": self.run_contract["contract_hash"],
            "contract": copy.deepcopy(self.run_contract),
            "checkpoint": {
                "kind": checkpoint_kind,
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
            },
            "initialization": copy.deepcopy(dict(initialization or {})),
        }
        blob = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": int(epoch),
            "iteration": None if iteration is None else int(iteration),
            "model": unwrap_trainable_module(model).state_dict(),
            "criterion": criterion.state_dict(),
            "metrics": copy.deepcopy(metrics),
            "cfg": copy.deepcopy(cfg),
            "architecture": copy.deepcopy(self.run_contract["architecture"]),
            "contract_hash": self.run_contract["contract_hash"],
            "provenance": provenance,
            "rng_state": copy.deepcopy(dict(rng_state)),
        }
        if optim is not None:
            blob["optim"] = optim.state_dict()
        if scaler is not None:
            blob["scaler"] = scaler.state_dict()
        if engine_state is not None:
            blob["engine_state"] = copy.deepcopy(dict(engine_state))
        return blob

    def stage_last_path(self, stage_idx: int) -> Path:
        return self.root / f"last_stage_{stage_idx}.pth"

    def stage_best_path(self, stage_idx: int) -> Path:
        return self.root / f"best_stage_{stage_idx}.pth"

    def best_path(self, stage_idx: int) -> Path:
        """Path of a stage's selected checkpoint."""
        return self.stage_best_path(stage_idx)

    def save_stage_last(
        self,
        stage_idx: int,
        model,
        criterion,
        optim,
        epoch: int,
        metrics,
        cfg,
        *,
        stage_name: str,
        initialization: Optional[Mapping[str, Any]],
        scaler=None,
        iteration: Optional[int] = None,
        engine_state: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        # Runs on every rank before rank zero writes.
        rng_state = capture_rng_states_by_rank()
        path = self.stage_last_path(stage_idx)
        if is_main():
            blob = self._build_blob(
                model, criterion, optim, epoch, metrics, cfg,
                checkpoint_kind="stage_last", stage_idx=stage_idx,
                stage_name=stage_name, initialization=initialization,
                scaler=scaler, iteration=iteration, rng_state=rng_state,
                engine_state=engine_state,
            )
            self._atomic_save(blob, path)
        return path

    def save_stage_best(
        self,
        stage_idx: int,
        model,
        criterion,
        optim,
        epoch: int,
        metrics,
        cfg,
        *,
        stage_name: str,
        initialization: Optional[Mapping[str, Any]],
        scaler=None,
        iteration: Optional[int] = None,
    ) -> Path:
        # Runs on every rank before rank zero writes.
        rng_state = capture_rng_states_by_rank()
        path = self.stage_best_path(stage_idx)
        if is_main():
            blob = self._build_blob(
                model, criterion, optim, epoch, metrics, cfg,
                checkpoint_kind="stage_best", stage_idx=stage_idx,
                stage_name=stage_name, initialization=initialization,
                scaler=scaler, iteration=iteration, rng_state=rng_state,
            )
            self._atomic_save(blob, path)
        return path

    def _load_stage_blob(self, path: Path, *, stage_idx: int,
                         expected_kind: str) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"required final-stage checkpoint is missing: {path}")
        blob = torch_load_checkpoint(path)
        if not isinstance(blob, Mapping):
            raise CheckpointFormatError(f"{path}: checkpoint payload must be a mapping")
        validate_checkpoint_provenance(
            blob, path,
            expected_architecture=self.run_contract["architecture"],
            expected_contract=self.run_contract,
            expected_stage_idx=stage_idx,
            expected_kinds=(expected_kind,),
        )
        return copy.deepcopy(dict(blob))

    def publish_final_stage_aliases(self, final_stage_idx: int) -> Dict[str, str]:
        """Publish ``best.pth``/``last.pth`` as rewritten blobs of the final stage."""
        if not is_main():
            return {}
        stage_best = self.stage_best_path(final_stage_idx)
        stage_last = self.stage_last_path(final_stage_idx)
        sources = (
            (stage_best, self._global_best_path, "final_stage_best"),
            (stage_last, self._global_last_path, "final_stage_last"),
        )
        published: Dict[str, str] = {}
        for source, destination, alias_kind in sources:
            blob = self._load_stage_blob(
                source, stage_idx=final_stage_idx,
                expected_kind="stage_best" if source == stage_best else "stage_last")
            provenance = blob["provenance"]
            checkpoint_meta = dict(provenance["checkpoint"])
            checkpoint_meta.update({
                "kind": alias_kind,
                "alias_of": source.name,
                "alias_source_sha256": sha256_file(source),
            })
            provenance["checkpoint"] = checkpoint_meta
            blob["provenance"] = provenance
            self._atomic_save(blob, destination)
            published[destination.name] = str(destination)
        self.log.info("[ckpt] published final-stage aliases from stage %d", final_stage_idx)
        return published

    def write_manifest(self, payload: Dict) -> None:
        if is_main():
            self._atomic_json(payload, self.root / "manifest.json")
