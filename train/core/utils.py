"""Distributed-training utilities, checkpoint contracts, and loading."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from models.dehazeformer import DEHAZE_CHANNELS
from train.core.torch_io import torch_load_checkpoint, torch_load_weights


def unwrap_trainable_module(model: nn.Module) -> nn.Module:
    """Strip transparent distributed ``.module`` wrappers."""
    m = model
    for _ in range(16):
        inner = getattr(m, "module", None)
        if isinstance(inner, nn.Module):
            m = inner
        else:
            break
    return m


def barrier_after_rank0_checkpoint() -> None:
    """No-op on single-GPU; barrier across the active process group otherwise."""
    if dist.is_available() and dist.is_initialized() \
            and dist.get_world_size() > 1:
        dist.barrier()


CONTRACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint errors."""


class CheckpointFormatError(CheckpointError):
    """Checkpoint lacks the required self-describing structure."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint tensors mismatch the receiving module."""


class ResumeContractError(CheckpointError):
    """Resume attempt crosses a different protocol or source revision."""


def _jsonable(value: Any) -> Any:
    """Convert supported config/provenance values into stable JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical JSON encoding used for reproducible provenance digests."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    """Return a file's SHA-256, streamed in chunks."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint/source file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str) -> Optional[bytes]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def source_provenance(
    repo_root: str | os.PathLike,
    *,
    exclude_paths: Sequence[str | os.PathLike] = (),
) -> dict:
    """Fingerprint the tracked source tree from working-tree bytes."""
    root = Path(repo_root).resolve()
    excluded = set()
    for value in exclude_paths:
        candidate = Path(value).resolve()
        try:
            excluded.add(candidate.relative_to(root).as_posix())
        except ValueError:
            pass
    revision_raw = _run_git(root, "rev-parse", "HEAD")
    files_raw = _run_git(root, "ls-files", "-z")
    if revision_raw is None or files_raw is None:
        raise RuntimeError(
            f"cannot establish Git source provenance under {root}; train from a "
            "Git checkout so resume contracts can be verified")

    digest = hashlib.sha256()
    count = 0
    missing_tracked = []
    for encoded in files_raw.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", "surrogateescape")
        if relative in excluded:
            continue
        candidate = root / relative
        digest.update(relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        if candidate.is_file():
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            # Hash a sentinel for a tracked deletion.
            digest.update(b"<MISSING-TRACKED-FILE>")
            missing_tracked.append(relative)
        digest.update(b"\0")
        count += 1

    status_raw = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    untracked_raw = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    if status_raw is None or untracked_raw is None:
        raise RuntimeError(f"cannot inspect Git working-tree status under {root}")
    untracked_digest = hashlib.sha256()
    untracked_count = 0
    for encoded in untracked_raw.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", "surrogateescape")
        if relative in excluded:
            continue
        candidate = root / relative
        if not candidate.is_file():
            raise RuntimeError(
                f"untracked provenance file is not a regular file: {candidate}")
        untracked_digest.update(relative.encode("utf-8", "surrogateescape"))
        untracked_digest.update(b"\0")
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                untracked_digest.update(block)
        untracked_digest.update(b"\0")
        untracked_count += 1
    return {
        "git_revision": revision_raw.decode("ascii").strip(),
        "tracked_tree_sha256": digest.hexdigest(),
        "tracked_file_count": count,
        "missing_tracked_files": missing_tracked,
        "untracked_tree_sha256": untracked_digest.hexdigest(),
        "untracked_file_count": untracked_count,
        "excluded_paths": sorted(excluded),
        "audit": {
            "dirty": bool(status_raw.strip()),
            "status_sha256": sha256_bytes(status_raw),
        },
    }


def architecture_from_config(cfg: Mapping[str, Any]) -> dict:
    """The complete constructor contract for a ``LiDSegModel`` checkpoint."""
    guidance_encoder_kwargs = cfg.get(
        "guidance_encoder_kwargs", cfg.get("guidance_encoder", {})) or {}
    backbone_kwargs = cfg.get("backbone_kwargs", {}) or {}
    return {
        "backbone": str(cfg["backbone"]),
        "n_classes": int(cfg["n_classes"]),
        "guidance": bool(cfg.get("guidance", True)),
        "use_decoder": bool(cfg.get("use_decoder", True)),
        "guidance_channels": list(cfg.get("guidance_channels", DEHAZE_CHANNELS)),
        "guidance_depths": list(cfg.get("guidance_depths", (4, 6, 6))),
        "guidance_encoder_kwargs": _jsonable(dict(guidance_encoder_kwargs)),
        "backbone_kwargs": _jsonable(dict(backbone_kwargs)),
    }


# Fields excluded from the resume-bound protocol.
_MUTABLE_RUN_CONFIG_KEYS = {
    "exp_name", "log_every", "grad_log_every", "tensorboard",
    "val_batch_size",
}
_MUTABLE_STAGE_CONFIG_KEYS = {"epochs"}


def _resume_protocol_config(cfg: Mapping[str, Any]) -> dict:
    protocol = copy.deepcopy(dict(cfg))
    for key in _MUTABLE_RUN_CONFIG_KEYS:
        protocol.pop(key, None)
    stages = protocol.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict):
                for key in _MUTABLE_STAGE_CONFIG_KEYS:
                    stage.pop(key, None)
    # Per-stage fields derived at runtime.
    protocol.pop("stage_lr_mult", None)
    protocol.pop("active_recon", None)
    protocol.pop("replay_ratio", None)
    return _jsonable(protocol)


def contract_hash(contract: Mapping[str, Any]) -> str:
    """Digest a contract, ignoring its stored digest field."""
    payload = copy.deepcopy(dict(contract))
    payload.pop("contract_hash", None)
    # The submitted YAML as resolved for this run.
    payload.pop("audit", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def build_run_contract(
    cfg: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike,
    world_size: int,
    config_path: Optional[str | os.PathLike] = None,
) -> dict:
    """Build the immutable contract required to resume a training stage."""
    raw_config = None
    config_exclusions: list[Path] = []
    if config_path is not None:
        config_file = Path(config_path).resolve()
        raw_config = {
            "path": str(config_file),
            "sha256": sha256_file(config_file),
        }
        config_exclusions.append(config_file)
    source = source_provenance(repo_root, exclude_paths=config_exclusions)
    source_audit = source.pop("audit", {})
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "architecture": architecture_from_config(cfg),
        "protocol": _resume_protocol_config(cfg),
        "runtime": {
            "world_size": int(world_size),
            "seed": int(cfg["seed"]),
            "deterministic": bool(cfg.get("deterministic", True)),
        },
        "source": source,
    }
    # The resolved config and source fingerprint for this run.
    contract["audit"] = {"source_worktree": source_audit}
    if raw_config is not None:
        contract["audit"]["config_source"] = raw_config
    contract["contract_hash"] = contract_hash(contract)
    return contract


def _format_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        lines: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                lines.append(f"{path}: unexpected in checkpoint")
            elif key not in actual:
                lines.append(f"{path}: missing from checkpoint")
            else:
                lines.extend(_format_differences(expected[key], actual[key], path))
        return lines
    if expected != actual:
        return [f"{prefix}: checkpoint={actual!r}, current={expected!r}"]
    return []


def _checkpoint_provenance(blob: Mapping[str, Any], path: str | os.PathLike) -> dict:
    if blob.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"{path}: unsupported or missing checkpoint schema version")
    provenance = blob.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CheckpointFormatError(
            f"{path}: missing provenance; only self-describing LiDSeg checkpoints "
            "may be used for full initialization or resume")
    if provenance.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"{path}: unsupported or missing provenance schema version")
    contract = provenance.get("contract")
    stored_hash = provenance.get("contract_hash", blob.get("contract_hash"))
    if not isinstance(contract, Mapping) or not isinstance(stored_hash, str):
        raise CheckpointFormatError(
            f"{path}: malformed provenance contract or contract_hash")
    if blob.get("contract_hash") != stored_hash:
        raise CheckpointFormatError(
            f"{path}: top-level/provenance contract hashes disagree")
    computed = contract_hash(contract)
    if computed != stored_hash:
        raise CheckpointFormatError(
            f"{path}: provenance contract hash is invalid "
            f"(stored={stored_hash}, computed={computed})")
    top_level_architecture = blob.get("architecture")
    if not isinstance(top_level_architecture, Mapping):
        raise CheckpointFormatError(f"{path}: missing top-level architecture")
    if _jsonable(top_level_architecture) != _jsonable(contract.get("architecture")):
        raise CheckpointFormatError(
            f"{path}: top-level architecture disagrees with provenance contract")
    return dict(provenance)


def validate_checkpoint_provenance(
    blob: Mapping[str, Any],
    path: str | os.PathLike,
    *,
    expected_architecture: Optional[Mapping[str, Any]] = None,
    expected_contract: Optional[Mapping[str, Any]] = None,
    expected_stage_idx: Optional[int] = None,
    expected_kinds: Optional[Sequence[str]] = None,
    allow_decoder_drop: bool = False,
) -> dict:
    """Validate a self-produced checkpoint's provenance."""
    provenance = _checkpoint_provenance(blob, path)
    contract = provenance["contract"]
    if expected_architecture is not None:
        expected_arch = _jsonable(expected_architecture)
        actual_arch = _jsonable(contract["architecture"])
        differences = _format_differences(expected_arch, actual_arch,
                                          "architecture")
        if allow_decoder_drop and differences:
            expected_without_decoder = dict(expected_arch)
            actual_without_decoder = dict(actual_arch)
            target_decoder = expected_without_decoder.pop("use_decoder", None)
            source_decoder = actual_without_decoder.pop("use_decoder", None)
            permitted = (
                target_decoder is False
                and source_decoder is True
                and not _format_differences(expected_without_decoder,
                                            actual_without_decoder,
                                            "architecture")
            )
            if permitted:
                differences = []
        if differences:
            raise CheckpointCompatibilityError(
                f"{path}: architecture mismatch: " + "; ".join(differences[:8]))
    if expected_contract is not None:
        expected_hash = expected_contract.get("contract_hash")
        if not isinstance(expected_hash, str) or expected_hash != contract_hash(expected_contract):
            raise ResumeContractError(
                "current resume contract is malformed or its contract_hash is invalid")
        actual_hash = str(provenance["contract_hash"])
        if expected_hash != actual_hash:
            differences = _format_differences(
                _jsonable(expected_contract), _jsonable(contract), "contract")
            detail = "; ".join(differences[:8]) or "contract hash differs"
            raise ResumeContractError(
                f"{path}: resume contract drift detected: {detail}")
    checkpoint_meta = provenance.get("checkpoint")
    if not isinstance(checkpoint_meta, Mapping):
        raise CheckpointFormatError(f"{path}: missing checkpoint provenance metadata")
    if expected_stage_idx is not None:
        actual_stage = checkpoint_meta.get("stage_idx")
        if actual_stage != expected_stage_idx:
            raise ResumeContractError(
                f"{path}: checkpoint belongs to stage {actual_stage!r}, expected "
                f"stage {expected_stage_idx}")
    if expected_kinds is not None:
        actual_kind = checkpoint_meta.get("kind")
        if actual_kind not in set(expected_kinds):
            raise ResumeContractError(
                f"{path}: checkpoint kind {actual_kind!r} is not valid here; "
                f"expected one of {sorted(set(expected_kinds))}")
    return provenance


def normalize_checkpoint_state_dict(sd: dict) -> dict:
    """Strip distributed ``module.`` prefixes from keys."""
    if not isinstance(sd, dict):
        return sd
    out = {}
    for key, value in sd.items():
        normalized = key
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in out:
            raise CheckpointFormatError(
                "checkpoint state dict has colliding keys after transparent "
                f"module-prefix normalization: {key!r} -> {normalized!r}")
        out[normalized] = value
    return out


def extract_checkpoint_state_dict(
    blob: Any,
    path: str | os.PathLike,
    *,
    require_model_key: bool = False,
) -> dict:
    """Extract the tensor state mapping from a checkpoint blob."""
    if not isinstance(blob, Mapping):
        raise CheckpointFormatError(f"{path}: checkpoint payload must be a mapping")
    if require_model_key:
        state = blob.get("model")
        if not isinstance(state, Mapping):
            raise CheckpointFormatError(
                f"{path}: full LiDSeg checkpoint must contain a 'model' state dict")
        return normalize_checkpoint_state_dict(dict(state))
    for key in ("state_dict", "model", "model_state_dict"):
        state = blob.get(key)
        if isinstance(state, Mapping):
            return normalize_checkpoint_state_dict(dict(state))
    if blob and all(isinstance(key, str) for key in blob):
        return normalize_checkpoint_state_dict(dict(blob))
    raise CheckpointFormatError(f"{path}: no supported state_dict/model mapping found")


def strict_load_state_dict(
    module: nn.Module,
    state: Mapping[str, Any],
    *,
    path: str | os.PathLike,
    context: str,
    allowed_missing_prefixes: Sequence[str] = (),
    allowed_unexpected_prefixes: Sequence[str] = (),
) -> None:
    """Load a state dict with exact key and shape matching."""
    target = module.state_dict()
    source = dict(state)
    missing = sorted(
        key for key in set(target) - set(source)
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes))
    unexpected = sorted(
        key for key in set(source) - set(target)
        if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes))
    shape_mismatches = []
    for key in sorted(set(target) & set(source)):
        source_value, target_value = source[key], target[key]
        if not torch.is_tensor(source_value):
            shape_mismatches.append(
                f"{key}: source is {type(source_value).__name__}, not tensor")
        elif tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                f"{key}: checkpoint{tuple(source_value.shape)} != "
                f"model{tuple(target_value.shape)}")
    if missing or unexpected or shape_mismatches:
        pieces = []
        if missing:
            pieces.append(f"missing={len(missing)} ({missing[:3]})")
        if unexpected:
            pieces.append(f"unexpected={len(unexpected)} ({unexpected[:3]})")
        if shape_mismatches:
            pieces.append(f"shape_mismatch={len(shape_mismatches)} "
                          f"({shape_mismatches[:2]})")
        raise CheckpointCompatibilityError(
            f"{path}: incompatible {context}: " + "; ".join(pieces))
    try:
        # Filter transfer-only keys before loading.
        accepted = {
            key: value for key, value in source.items()
            if key in target
        }
        module.load_state_dict(accepted, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"{path}: strict {context} load failed: {exc}") from exc


def _serialize_numpy_rng_state() -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": str(algorithm),
        "keys": torch.from_numpy(np.asarray(keys, dtype=np.uint32).copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(payload: Mapping[str, Any]) -> None:
    try:
        keys = payload["keys"]
        if not torch.is_tensor(keys):
            raise TypeError("keys is not a tensor")
        state = (
            str(payload["algorithm"]),
            keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
        np.random.set_state(state)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError(f"invalid NumPy RNG state: {exc}") from exc


def capture_rng_state() -> dict:
    """Capture RNG state as primitives and tensors."""
    py_version, py_state, py_gauss = random.getstate()
    payload = {
        "python": {
            "version": int(py_version),
            "state": [int(v) for v in py_state],
            "gauss_next": None if py_gauss is None else float(py_gauss),
        },
        "numpy": _serialize_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = list(torch.cuda.get_rng_state_all())
    return payload


def capture_rng_states_by_rank() -> dict:
    """Gather rank-local RNG states before rank zero atomically saves a checkpoint."""
    local = capture_rng_state()
    if dist.is_available() and dist.is_initialized():
        states: list[Optional[dict]] = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
    else:
        states = [local]
    return {"schema_version": 1, "by_rank": states}


def restore_rng_state(payload: Mapping[str, Any], *, rank: int = 0) -> None:
    """Restore the saved rank-local random streams after all objects are built."""
    states = payload.get("by_rank") if isinstance(payload, Mapping) else None
    if not isinstance(states, Sequence) or rank < 0 or rank >= len(states):
        raise CheckpointFormatError(
            f"checkpoint RNG state lacks an entry for rank {rank}")
    state = states[rank]
    if not isinstance(state, Mapping):
        raise CheckpointFormatError(f"checkpoint RNG state for rank {rank} is invalid")
    try:
        py = state["python"]
        if not isinstance(py, Mapping):
            raise TypeError("python state is not a mapping")
        random.setstate((int(py["version"]), tuple(py["state"]), py["gauss_next"]))
        _restore_numpy_rng_state(state["numpy"])
        cpu_state = state["torch_cpu"]
        if not torch.is_tensor(cpu_state):
            raise TypeError("torch_cpu is not a tensor")
        torch.set_rng_state(cpu_state)
        if "torch_cuda" in state:
            cuda_states = state["torch_cuda"]
            if not isinstance(cuda_states, Sequence) or not all(
                    torch.is_tensor(item) for item in cuda_states):
                raise TypeError("torch_cuda is not a tensor sequence")
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(list(cuda_states))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointFormatError(f"invalid checkpoint RNG state: {exc}") from exc


def is_done(output_root) -> bool:
    return (Path(output_root) / "DONE").is_file()


def mark_done(output_root) -> None:
    """Atomically publish the DONE marker."""
    path = Path(output_root) / "DONE"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("complete\n", encoding="utf-8")
    os.replace(tmp, path)


def load_resume_state(
    path: Path,
    model: nn.Module,
    *,
    expected_contract: Mapping[str, Any],
    stage_idx: int,
    log=None,
) -> dict:
    """Load a resumable stage snapshot after contract validation."""
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    blob = torch_load_checkpoint(path)
    if not isinstance(blob, Mapping):
        raise CheckpointFormatError(f"{path}: resume checkpoint must be a mapping")
    validate_checkpoint_provenance(
        blob, path,
        expected_architecture=expected_contract["architecture"],
        expected_contract=expected_contract,
        expected_stage_idx=stage_idx,
        expected_kinds=("stage_last",),
    )
    state = extract_checkpoint_state_dict(blob, path, require_model_key=True)
    core = unwrap_trainable_module(model)
    strict_load_state_dict(core, state, path=path, context="resume model")
    if not isinstance(blob.get("criterion"), Mapping):
        raise CheckpointFormatError(f"{path}: resume checkpoint has no criterion state")
    if "rng_state" not in blob:
        raise CheckpointFormatError(f"{path}: resume checkpoint has no RNG state")
    if log is not None:
        log.info("[resume] strictly restored stage %d model from %s", stage_idx,
                 path)
    return dict(blob)


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1


def is_main() -> bool:
    return get_rank() == 0


def init_distributed() -> Tuple[int, int, int]:
    """Initialise torch.distributed via torchrun env. Returns (rank, world, local)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun/DDP requires CUDA for this NCCL trainer")
        torch.cuda.set_device(local)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        return rank, world, local
    return 0, 1, 0


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if is_ddp():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def set_seed(seed: int, rank: int = 0, *, deterministic: bool = True) -> None:
    """Seed every RNG and set the determinism flags."""
    value = int(seed) + int(rank)
    if deterministic:
        # Set before any CUDA work.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def build_logger(name: str, log_file: str = None) -> logging.Logger:
    """Build a rank-zero logger writing to an optional file."""
    log = logging.getLogger(name)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%m/%d %I:%M:%S %p")
    if is_main():
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        log.addHandler(stream)
        if log_file:
            path = os.path.abspath(os.path.expanduser(log_file))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(fmt)
            log.addHandler(handler)
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)
    return log


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _require_checkpoint_file(path: Optional[str], *, kind: str) -> Path:
    if not path:
        raise FileNotFoundError(f"{kind} checkpoint path is required but was empty")
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{kind} checkpoint does not exist: {candidate}")
    return candidate.resolve()


def _strip_backbone_wrappers(state: Mapping[str, Any]) -> dict:
    cleaned = {}
    for key, value in state.items():
        normalized = key
        while normalized.startswith(("module.", "model.", "backbone.")):
            for prefix in ("module.", "model.", "backbone."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
        cleaned[normalized] = value
    return cleaned


_OUTPUT_KEY_PREFIXES = (
    # BiSeNetV2
    "head.", "aux2.", "aux3.", "aux4.", "aux5_4.",
    # DDRNet
    "final_layer.", "seghead_extra.",
    # PIDNet / SCTNet
    "conv_seg.", "p_cls_seg.", "d_cls_seg.",
    # STDC segmentation outputs
    "conv_out.", "conv_out16.", "conv_out32.",
    "conv_out_sp2.", "conv_out_sp4.", "conv_out_sp8.", "conv_out_sp16.",
    # Official names for segmentation outputs
    "decode_head.", "classifier.", "classification_head.",
)


def _is_output_key(key: str) -> bool:
    return key.lower().startswith(_OUTPUT_KEY_PREFIXES)


def _validate_partial_backbone_load(
    target: torch.nn.Module,
    source: Mapping[str, Any],
    *,
    path: Path,
    context: str,
) -> dict:
    """Compare a pretrain state dict against the backbone, ignoring output heads."""
    target_state = target.state_dict()
    matched: dict = {}
    missing = []
    unexpected = []
    mismatched = []
    for key, value in target_state.items():
        if key not in source:
            missing.append(key)
        elif not torch.is_tensor(source[key]) or tuple(source[key].shape) != tuple(value.shape):
            mismatched.append(key)
        else:
            matched[key] = source[key]
    for key in source:
        if key not in target_state:
            unexpected.append(key)

    bad_missing = [key for key in missing if not _is_output_key(key)]
    bad_mismatched = [key for key in mismatched if not _is_output_key(key)]
    bad_unexpected = [key for key in unexpected if not _is_output_key(key)]
    if bad_missing or bad_mismatched or bad_unexpected:
        issues = []
        if bad_missing:
            issues.append(f"missing non-output={bad_missing[:4]}")
        if bad_mismatched:
            issues.append(f"mismatched non-output={bad_mismatched[:4]}")
        if bad_unexpected:
            issues.append(f"unexpected non-output={bad_unexpected[:4]}")
        raise CheckpointCompatibilityError(
            f"{path}: incompatible {context}: " + "; ".join(issues))
    if not matched:
        raise CheckpointCompatibilityError(
            f"{path}: incompatible {context}: no backbone tensors matched")

    total_numel = sum(tensor.numel() for tensor in target_state.values())
    matched_numel = sum(target_state[key].numel() for key in matched)
    coverage = matched_numel / max(1, total_numel)
    if coverage < 0.90:
        raise CheckpointCompatibilityError(
            f"{path}: incompatible {context}: only {coverage:.1%} of backbone "
            "parameters matched; expected at least 90%")
    try:
        result = target.load_state_dict(matched, strict=False)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"{path}: {context} load failed: {exc}") from exc
    residual_missing = [key for key in result.missing_keys if not _is_output_key(key)]
    residual_unexpected = [key for key in result.unexpected_keys if not _is_output_key(key)]
    if residual_missing or residual_unexpected:
        raise CheckpointCompatibilityError(
            f"{path}: {context} left non-output tensors unloaded: "
            f"missing={residual_missing[:4]}, unexpected={residual_unexpected[:4]}")
    return {
        "matched_tensors": len(matched),
        "target_tensors": len(target_state),
        "matched_parameter_fraction": coverage,
        "allowed_output_missing": sorted(missing),
        "allowed_output_mismatched": sorted(mismatched),
        "allowed_output_unexpected": sorted(unexpected),
    }


def _load_official_backbone(
    target: torch.nn.Module,
    source: Mapping[str, Any],
    *,
    path: Path,
) -> Optional[dict]:
    """Apply a model-owned key mapper and check coverage."""
    loader = getattr(target, "load_official_state_dict", None)
    if not callable(loader):
        return None
    try:
        result = loader(dict(source), strict=False)
    except TypeError:
        result = loader(dict(source))
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"{path}: official backbone key mapping failed: {exc}") from exc
    if result is None:
        raise CheckpointCompatibilityError(
            f"{path}: official backbone loader returned no compatibility report")
    missing = list(getattr(result, "missing_keys", result[0] if isinstance(result, tuple) else []))
    unexpected = list(getattr(result, "unexpected_keys", result[1] if isinstance(result, tuple) and len(result) > 1 else []))
    bad_missing = [key for key in missing if not _is_output_key(key)]
    bad_unexpected = [key for key in unexpected if not _is_output_key(key)]
    if bad_missing or bad_unexpected:
        raise CheckpointCompatibilityError(
            f"{path}: official backbone mapping left non-output tensors unmatched: "
            f"missing={bad_missing[:4]}, unexpected={bad_unexpected[:4]}")
    total = len(target.state_dict())
    matched = total - len(missing)
    if matched <= 0 or matched / max(1, total) < 0.90:
        raise CheckpointCompatibilityError(
            f"{path}: official backbone mapping coverage is insufficient "
            f"({matched}/{total} tensor keys)")
    return {
        "matched_tensors": matched,
        "target_tensors": total,
        "allowed_output_missing": sorted(missing),
        "allowed_output_unexpected": sorted(unexpected),
        "loader": "official",
    }


def load_pretrained(model, ckpt_path: str, log) -> dict:
    """Initialize a backbone from an external state dict."""
    path = _require_checkpoint_file(ckpt_path, kind="pretrained")
    blob = torch_load_weights(path)
    state = extract_checkpoint_state_dict(blob, path)
    raw = unwrap_trainable_module(model)
    if not hasattr(raw, "backbone"):
        raise TypeError("pretrained initialization expects a model with .backbone")
    target = raw.backbone

    # Model-owned key mappings for official PIDNet/SCTNet naming.
    official = _load_official_backbone(target, state, path=path)
    if official is None:
        candidates = [
            ("as-is", dict(state)),
            ("prefix-stripped", _strip_backbone_wrappers(state)),
        ]
        scored = []
        target_state = target.state_dict()
        for name, candidate in candidates:
            score = sum(
                1 for key, value in candidate.items()
                if key in target_state and torch.is_tensor(value)
                and tuple(value.shape) == tuple(target_state[key].shape))
            scored.append((score, name, candidate))
        _, mode, chosen = max(scored, key=lambda item: item[0])
        evidence = _validate_partial_backbone_load(
            target, chosen, path=path, context=f"pretrained backbone ({mode})")
        evidence["loader"] = mode
    else:
        evidence = official
    artifact = {
        "kind": "pretrain",
        "path": str(path),
        "sha256": sha256_file(path),
        **evidence,
    }
    log.info("[ckpt] strict pretrained backbone load from %s (%s; %d/%d tensors)",
             path, artifact["loader"], artifact["matched_tensors"],
             artifact["target_tensors"])
    return artifact


def load_full_ckpt(
    model,
    ckpt_path: str,
    log,
    *,
    expected_architecture: Mapping[str, Any],
    expected_contract: Optional[Mapping[str, Any]] = None,
    expected_stage_idx: Optional[int] = None,
    expected_kinds: Optional[Sequence[str]] = None,
    allow_decoder_drop: bool = False,
) -> dict:
    """Load a self-describing LiDSeg checkpoint with explicit transfer rules."""
    path = _require_checkpoint_file(ckpt_path, kind="full")
    blob = torch_load_checkpoint(path)
    if not isinstance(blob, Mapping):
        raise CheckpointFormatError(f"{path}: full checkpoint must be a mapping")
    provenance = validate_checkpoint_provenance(
        blob, path,
        expected_architecture=expected_architecture,
        expected_contract=expected_contract,
        expected_stage_idx=expected_stage_idx,
        expected_kinds=expected_kinds,
        allow_decoder_drop=allow_decoder_drop,
    )
    source_architecture = provenance["contract"]["architecture"]
    documented_decoder_drop = (
        bool(allow_decoder_drop)
        and expected_architecture.get("use_decoder") is False
        and source_architecture.get("use_decoder") is True
    )
    state = extract_checkpoint_state_dict(blob, path, require_model_key=True)
    strict_load_state_dict(
        unwrap_trainable_module(model), state, path=path, context="full model",
        allowed_unexpected_prefixes=("decoder.",) if documented_decoder_drop else (),
    )
    checkpoint_meta = provenance["checkpoint"]
    artifact = {
        "kind": "full_ckpt",
        "path": str(path),
        "sha256": sha256_file(path),
        "source_contract_hash": provenance["contract_hash"],
        "source_checkpoint_kind": checkpoint_meta.get("kind"),
        "source_stage_idx": checkpoint_meta.get("stage_idx"),
        "decoder_dropped": documented_decoder_drop,
    }
    log.info("[ckpt] strict full-model load from %s (kind=%s stage=%s)", path,
             artifact["source_checkpoint_kind"], artifact["source_stage_idx"])
    return artifact
