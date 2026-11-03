"""Shared helpers for the evaluation entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.paths import paths
from dataset.loader import VALID_HW_KEY, build_cityscapes_foggy, collate
from dataset.segdatasets.acdc import ACDC
from dataset.segdatasets.cityscapes import Cityscapes
from models.dehazeformer import DEHAZE_CHANNELS
from models.lidseg import LiDSegModel
from train.core.metrics import (fast_hist, jsonable_iou, mean_iou,
                                per_class_iou)
from train.core.utils import contract_hash, sha256_file
from train.core.torch_io import torch_load_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[1]

CS_CLASSES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]
NUM_CLASSES = 19
IGNORE_INDEX = 255

CS_FOGGY_BETAS: Tuple[str, ...] = ("clear", "0.005", "0.01", "0.02")

ACDC_CONDITIONS: Tuple[str, ...] = ("fog", "night", "rain", "snow")

VARIANT_SPEC: Dict[str, Tuple[bool, bool]] = {
    "baseline": (False, False),
    "guided":   (True,  True),
    "norecon":  (True,  False),
}
VARIANT_NAMES: Tuple[str, ...] = tuple(VARIANT_SPEC.keys())

_ARCH_REQUIRED: Tuple[str, ...] = (
    "backbone",
    "n_classes",
    "guidance",
    "use_decoder",
    "guidance_channels",
    "guidance_depths",
    "guidance_encoder_kwargs",
    "backbone_kwargs",
)

ACDC_EXP_SUFFIXES: Tuple[str, ...] = ("uber",) + ACDC_CONDITIONS


def repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def normalize_backbone(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def parse_exp_name(
    name: str, dataset: str
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Split an experiment dir name into (backbone, variant, acdc_suffix).

    ACDC names may carry one trailing run tag without underscores, e.g.
    ``pidnetl_norecon_uber_bblr0.1``; the tag is ignored.
    """
    head = name
    suffix: Optional[str] = None
    if dataset == "acdc":
        candidates = [head]
        if "_" in head:
            candidates.append(head.rsplit("_", 1)[0])
        for cand in candidates:
            for s in ACDC_EXP_SUFFIXES:
                if cand.endswith("_" + s):
                    suffix = s
                    head = cand[: -(len(s) + 1)]
                    break
            if suffix is not None:
                break
        if suffix is None:
            return None
    for v in sorted(VARIANT_NAMES, key=len, reverse=True):
        if head.endswith("_" + v):
            return head[: -(len(v) + 1)], v, suffix
    return None


def build_model(
    backbone: str, variant: str, num_classes: int = NUM_CLASSES,
    *, guidance_channels: Tuple[int, ...] = tuple(DEHAZE_CHANNELS),
    guidance_depths: Tuple[int, ...] = (4, 6, 6),
    guidance_encoder_kwargs: Optional[Mapping[str, Any]] = None,
    backbone_kwargs: Optional[Mapping[str, Any]] = None,
) -> torch.nn.Module:
    """Rebuild the model a variant's checkpoint was saved from."""
    if variant not in VARIANT_SPEC:
        raise KeyError(f"unknown variant {variant!r}; expected one of "
                       f"{sorted(VARIANT_SPEC)}")
    guidance, use_decoder = VARIANT_SPEC[variant]
    return LiDSegModel(
        backbone_name=normalize_backbone(backbone),
        n_classes=num_classes,
        guidance=guidance,
        use_decoder=use_decoder,
        guidance_channels=guidance_channels,
        guidance_depths=guidance_depths,
        guidance_encoder_kwargs=dict(guidance_encoder_kwargs or {}),
        backbone_kwargs=dict(backbone_kwargs or {}),
    )


def variant_from_architecture(architecture: Mapping[str, Any]) -> str:
    """Derive the experiment variant from model provenance."""
    guidance = architecture.get("guidance")
    decoder = architecture.get("use_decoder")
    if guidance is False and decoder is False:
        return "baseline"
    if guidance is True and decoder is True:
        return "guided"
    if guidance is True and decoder is False:
        return "norecon"
    raise RuntimeError(
        "invalid LiDSeg architecture provenance: use_decoder requires "
        "guidance=True")


def _normalise_architecture(raw: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Validate the architecture contract."""
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{source} architecture must be a mapping")
    missing = [key for key in _ARCH_REQUIRED if key not in raw]
    if missing:
        raise RuntimeError(
            f"{source} lacks complete architecture provenance; missing {missing}. "
            "Refusing to infer a model from a directory name or partial state "
            "dict.")
    backbone = raw["backbone"]
    if not isinstance(backbone, str) or not backbone.strip():
        raise RuntimeError(f"{source} architecture.backbone must be a non-empty string")
    n_classes = raw["n_classes"]
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise RuntimeError(
            f"{source} architecture.n_classes must be an integer >= 2, got "
            f"{n_classes!r}")
    guidance = raw["guidance"]
    use_decoder = raw["use_decoder"]
    if type(guidance) is not bool or type(use_decoder) is not bool:
        raise RuntimeError(
            f"{source} architecture guidance/use_decoder must be booleans")
    if use_decoder and not guidance:
        raise RuntimeError(
            f"{source} architecture is invalid: use_decoder=True with guidance=False")
    channels = raw["guidance_channels"]
    if (not isinstance(channels, (list, tuple)) or len(channels) != 3
            or any(isinstance(c, bool) or not isinstance(c, int) or c < 1
                   for c in channels)):
        raise RuntimeError(
            f"{source} architecture.guidance_channels must contain three positive "
            f"integers, got {channels!r}")
    depths = raw["guidance_depths"]
    if (not isinstance(depths, (list, tuple)) or len(depths) != 3
            or any(isinstance(d, bool) or not isinstance(d, int) or d < 1
                   for d in depths)):
        raise RuntimeError(
            f"{source} architecture.guidance_depths must contain three positive "
            f"integers, got {depths!r}")
    encoder_kwargs = raw["guidance_encoder_kwargs"]
    if not isinstance(encoder_kwargs, Mapping):
        raise RuntimeError(
            f"{source} architecture.guidance_encoder_kwargs must be a mapping")
    backbone_kwargs = raw["backbone_kwargs"]
    if not isinstance(backbone_kwargs, Mapping):
        raise RuntimeError(
            f"{source} architecture.backbone_kwargs must be a mapping")
    out = {
        "backbone": normalize_backbone(backbone),
        "n_classes": int(n_classes),
        "guidance": guidance,
        "use_decoder": use_decoder,
        "guidance_channels": tuple(int(c) for c in channels),
        "guidance_depths": tuple(int(d) for d in depths),
        "guidance_encoder_kwargs": dict(encoder_kwargs),
        "backbone_kwargs": dict(backbone_kwargs),
    }
    # Canonical variant from the guidance/decoder combination.
    out["variant"] = variant_from_architecture(out)
    return out


def _architecture_from_metadata(meta: Mapping[str, Any], *, ckpt_path: str) -> Dict[str, Any]:
    """Read top-level architecture metadata from a checkpoint."""
    explicit = meta.get("architecture")
    if not isinstance(explicit, Mapping):
        raise RuntimeError(
            f"checkpoint {ckpt_path} lacks explicit top-level architecture "
            "metadata; release evaluators do not infer architecture from cfg")
    return _normalise_architecture(explicit, source=f"checkpoint {ckpt_path}")


def architecture_for_report(architecture: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-safe architecture record included beside every evaluator result."""
    return {
        "backbone": architecture["backbone"],
        "n_classes": int(architecture["n_classes"]),
        "guidance": bool(architecture["guidance"]),
        "use_decoder": bool(architecture["use_decoder"]),
        "guidance_channels": list(architecture["guidance_channels"]),
        "guidance_depths": list(architecture["guidance_depths"]),
        "guidance_encoder_kwargs": dict(architecture["guidance_encoder_kwargs"]),
        "backbone_kwargs": dict(architecture["backbone_kwargs"]),
        "variant": architecture["variant"],
    }


def _validate_expected_architecture(
    architecture: Mapping[str, Any], *, expected_backbone: Optional[str] = None,
    expected_variant: Optional[str] = None,
    expected_n_classes: Optional[int] = None,
    expected_architecture: Optional[Mapping[str, Any]] = None,
) -> None:
    """Check command-line and directory claims against checkpoint data."""
    if expected_backbone is not None:
        if normalize_backbone(expected_backbone) != architecture["backbone"]:
            raise RuntimeError(
                "checkpoint backbone provenance disagrees with requested/name "
                f"backbone: checkpoint={architecture['backbone']!r}, "
                f"expected={expected_backbone!r}")
    if expected_variant is not None and expected_variant != architecture["variant"]:
        raise RuntimeError(
            "checkpoint variant provenance disagrees with requested/name "
            f"variant: checkpoint={architecture['variant']!r}, "
            f"expected={expected_variant!r}")
    if expected_n_classes is not None and int(expected_n_classes) != architecture["n_classes"]:
        raise RuntimeError(
            "checkpoint n_classes provenance disagrees with evaluator: "
            f"checkpoint={architecture['n_classes']}, "
            f"expected={expected_n_classes}")
    if expected_architecture is not None:
        expected = _normalise_architecture(
            expected_architecture, source="expected architecture")
        fields = (
            "backbone", "n_classes", "guidance", "use_decoder",
            "guidance_channels", "guidance_depths",
            "guidance_encoder_kwargs", "backbone_kwargs",
        )
        mismatches = [
            key for key in fields if architecture[key] != expected[key]
        ]
        if mismatches:
            raise RuntimeError(
                "checkpoint architecture differs from the model/checkpoint it is "
                f"being compared against in fields {mismatches}")


def build_model_from_architecture(architecture: Mapping[str, Any]) -> torch.nn.Module:
    """Instantiate the model described by checkpoint metadata."""
    arch = _normalise_architecture(architecture, source="architecture")
    return LiDSegModel(
        backbone_name=arch["backbone"],
        n_classes=arch["n_classes"],
        guidance=arch["guidance"],
        use_decoder=arch["use_decoder"],
        guidance_channels=arch["guidance_channels"],
        guidance_depths=arch["guidance_depths"],
        guidance_encoder_kwargs=arch["guidance_encoder_kwargs"],
        backbone_kwargs=arch["backbone_kwargs"],
    )


def _torch_load_state(ckpt_path: str):
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {ckpt_path}")
    blob = torch_load_checkpoint(ckpt_path)
    if isinstance(blob, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in blob and isinstance(blob[k], dict):
                state = blob[k]
                meta = {kk: vv for kk, vv in blob.items() if kk != k}
                return state, meta
    return blob, {}


def _normalise_state_dict(state: Any, *, ckpt_path: str) -> Dict[str, Any]:
    """Remove a DDP prefix from state-dict keys."""
    if not isinstance(state, dict):
        raise RuntimeError(f"unrecognised checkpoint format: {ckpt_path}")
    if not state:
        raise RuntimeError(f"checkpoint contains an empty model state: {ckpt_path}")
    if any(not isinstance(k, str) for k in state):
        raise RuntimeError(
            f"checkpoint {ckpt_path} has a non-string state-dict key")
    if any(k.startswith("dehazeformer.") for k in state.keys()):
        raise RuntimeError(
            f"{ckpt_path}: this checkpoint has 'dehazeformer.*' tensors, so it "
            "was not saved by train/train.py. The evaluation entry points build "
            "LiDSegModel and expect 'backbone.*' / 'guidance.*' / 'lgf.*' keys."
        )
    normalized: Dict[str, Any] = {}
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.")
        if key in normalized:
            raise RuntimeError(
                f"checkpoint {ckpt_path} has colliding keys after DDP prefix "
                f"normalization: {raw_key!r} -> {key!r}")
        normalized[key] = value
    return normalized


def _validate_checkpoint_provenance(
    meta: Mapping[str, Any], architecture: Mapping[str, Any], *, ckpt_path: str,
) -> None:
    """Check the architecture is bound to a trainer contract."""
    provenance = meta.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            f"checkpoint {ckpt_path} lacks provenance metadata; release "
            "evaluators require a contract-bound checkpoint, not a bare state "
            "dict or unverifiable legacy artifact")
    contract = provenance.get("contract")
    declared_hash = provenance.get("contract_hash")
    if not isinstance(contract, Mapping) or not isinstance(declared_hash, str):
        raise RuntimeError(
            f"checkpoint {ckpt_path} has incomplete provenance contract metadata")
    if not declared_hash or contract.get("contract_hash") != declared_hash:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has inconsistent provenance contract hashes")
    top_level_hash = meta.get("contract_hash")
    if top_level_hash != declared_hash:
        raise RuntimeError(
            f"checkpoint {ckpt_path} top-level/provenance contract hashes differ")
    try:
        computed_hash = contract_hash(contract)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has a non-canonical provenance contract") from exc
    if computed_hash != declared_hash:
        raise RuntimeError(
            f"checkpoint {ckpt_path} provenance contract hash does not match its "
            "contract payload")
    contract_architecture = contract.get("architecture")
    contract_architecture = _normalise_architecture(
        contract_architecture,
        source=f"checkpoint {ckpt_path} provenance contract",
    )
    _validate_expected_architecture(
        architecture, expected_architecture=contract_architecture)
    checkpoint_info = provenance.get("checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        raise RuntimeError(
            f"checkpoint {ckpt_path} provenance lacks checkpoint kind/stage metadata")
    stage_idx = checkpoint_info.get("stage_idx")
    kind = checkpoint_info.get("kind")
    if (isinstance(stage_idx, bool) or not isinstance(stage_idx, int)
            or stage_idx < 1 or not isinstance(kind, str) or not kind):
        raise RuntimeError(
            f"checkpoint {ckpt_path} has invalid checkpoint kind/stage provenance")


def _read_checkpoint(ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    state, meta = _torch_load_state(ckpt_path)
    state = _normalise_state_dict(state, ckpt_path=ckpt_path)
    architecture = _architecture_from_metadata(meta, ckpt_path=ckpt_path)
    _validate_checkpoint_provenance(meta, architecture, ckpt_path=ckpt_path)
    return state, meta, architecture


def checkpoint_architecture(ckpt_path: str) -> Dict[str, Any]:
    """Return normalized architecture provenance."""
    _, _, architecture = _read_checkpoint(ckpt_path)
    return architecture


def _strict_load_state(
    model: torch.nn.Module, state: Mapping[str, Any], *, ckpt_path: str,
    log=print,
) -> None:
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"strict checkpoint load failed for {ckpt_path}; architecture and "
            "state tensors must match exactly. This evaluator will not score a "
            f"partially initialized model.\n{exc}") from exc
    log(f"  loaded strictly: {len(state)} model tensors")


def load_checkpoint(
    model: torch.nn.Module, ckpt_path: str, log=print, *,
    expected_backbone: Optional[str] = None,
    expected_variant: Optional[str] = None,
    expected_n_classes: Optional[int] = None,
    expected_architecture: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Load a trainer checkpoint after validating its provenance."""
    state, meta, architecture = _read_checkpoint(ckpt_path)
    _validate_expected_architecture(
        architecture,
        expected_backbone=expected_backbone,
        expected_variant=expected_variant,
        expected_n_classes=expected_n_classes,
        expected_architecture=expected_architecture,
    )
    _strict_load_state(model, state, ckpt_path=ckpt_path, log=log)
    return meta


def build_model_from_checkpoint(
    ckpt_path: str, *, expected_backbone: Optional[str] = None,
    expected_variant: Optional[str] = None,
    expected_n_classes: Optional[int] = None,
    expected_architecture: Optional[Mapping[str, Any]] = None,
    log=print,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Build and load the model recorded in a checkpoint."""
    state, meta, architecture = _read_checkpoint(ckpt_path)
    _validate_expected_architecture(
        architecture,
        expected_backbone=expected_backbone,
        expected_variant=expected_variant,
        expected_n_classes=expected_n_classes,
        expected_architecture=expected_architecture,
    )
    model = build_model_from_architecture(architecture)
    _strict_load_state(model, state, ckpt_path=ckpt_path, log=log)
    return model, architecture, meta


def _stage_idx_from_tag(tag: str) -> int:
    s = tag.removeprefix("stage")
    for suf in ("_best", "_last"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return int(s)


def collect_checkpoints(exp_dir: Path) -> Dict[str, Path]:
    """All checkpoints in an experiment dir, per stage plus the globals."""
    out: Dict[str, Path] = {}
    if (exp_dir / "best.pth").is_file():
        out["best"] = exp_dir / "best.pth"
    if (exp_dir / "last.pth").is_file():
        out["last"] = exp_dir / "last.pth"
    for f in sorted(exp_dir.glob("stage*_best.pth")):
        out[f"stage{_stage_idx_from_tag(f.stem)}_best"] = f
    for f in sorted(exp_dir.glob("stage*_last.pth")):
        out[f"stage{_stage_idx_from_tag(f.stem)}_last"] = f
    for f in sorted(exp_dir.glob("best_stage_*.pth")):
        idx = int(f.stem.removeprefix("best_stage_"))
        out[f"stage{idx}_best"] = f
    for f in sorted(exp_dir.glob("last_stage_*.pth")):
        idx = int(f.stem.removeprefix("last_stage_"))
        out[f"stage{idx}_last"] = f
    return out


def _manifest_path(root: Path, raw_path: Any, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"manifest {field} must be a non-empty path string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"manifest {field} escapes experiment directory: {raw_path!r}") from exc
    return path


def _load_completed_manifest(exp_dir: Path) -> Tuple[Dict[str, Any], int]:
    """Read a completed, self-consistent experiment manifest."""
    root = Path(exp_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    path = root / "manifest.json"
    if not path.is_file():
        raise RuntimeError(
            f"{root} has no manifest.json; experiment-directory evaluation only "
            "accepts completed release runs. Use --checkpoint for an explicit "
            "partial-checkpoint evaluation.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid experiment manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"experiment manifest must be a JSON object: {path}")
    schema = manifest.get("schema_version")
    if isinstance(schema, bool) or schema != 1:
        raise RuntimeError(
            f"unsupported or missing manifest schema_version in {path}: "
            f"{schema!r}")
    if manifest.get("complete") is not True:
        raise RuntimeError(
            f"experiment {root} is incomplete; refusing to report a partial run")
    if not isinstance(manifest.get("completed_at"), str):
        raise RuntimeError(f"completed manifest is missing completed_at: {path}")
    final = manifest.get("final_stage")
    if not isinstance(final, Mapping):
        raise RuntimeError(f"completed manifest is missing final_stage: {path}")
    stage_idx = final.get("idx")
    if isinstance(stage_idx, bool) or not isinstance(stage_idx, int) or stage_idx < 1:
        raise RuntimeError(f"completed manifest has invalid final_stage.idx: {path}")
    if not isinstance(final.get("name"), str) or not final["name"]:
        raise RuntimeError(f"completed manifest has invalid final_stage.name: {path}")
    manifest_architecture = _normalise_architecture(
        manifest.get("architecture"), source=f"manifest {path}")
    manifest_hash = manifest.get("contract_hash")
    if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in manifest_hash.lower())):
        raise RuntimeError(f"completed manifest has invalid contract_hash: {path}")

    final_paths: Dict[str, Path] = {}
    for key, expected_kind in (("selected", "stage_best"),
                               ("resumable", "stage_last")):
        checkpoint_path = _manifest_path(
            root, final.get(key), field=f"final_stage.{key}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"completed manifest references missing {key} checkpoint: "
                f"{checkpoint_path}")
        _, checkpoint_meta = _torch_load_state(str(checkpoint_path))
        checkpoint_architecture = _architecture_from_metadata(
            checkpoint_meta, ckpt_path=str(checkpoint_path))
        _validate_checkpoint_provenance(
            checkpoint_meta, checkpoint_architecture, ckpt_path=str(checkpoint_path))
        _validate_expected_architecture(
            checkpoint_architecture, expected_architecture=manifest_architecture)
        if checkpoint_meta.get("contract_hash") != manifest_hash:
            raise RuntimeError(
                f"completed manifest {key} checkpoint contract hash disagrees "
                f"with manifest: {checkpoint_path}")
        info = checkpoint_meta["provenance"]["checkpoint"]
        if (info.get("kind") != expected_kind
                or info.get("stage_idx") != stage_idx):
            raise RuntimeError(
                f"completed manifest {key} checkpoint does not describe final "
                f"stage {stage_idx}: {checkpoint_path}")
        final_paths[key] = checkpoint_path

    stage_ids = manifest.get("stage_ids")
    if (not isinstance(stage_ids, list) or stage_idx not in stage_ids
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 1
                   for i in stage_ids)
            or len(set(stage_ids)) != len(stage_ids)):
        raise RuntimeError(
            f"completed manifest has invalid stage_ids/final_stage relation: {path}")
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise RuntimeError(f"completed manifest lacks per-stage records: {path}")
    final_record = stages.get(str(stage_idx))
    if not isinstance(final_record, Mapping):
        raise RuntimeError(
            f"completed manifest lacks a record for final stage {stage_idx}: {path}")
    for record_key, final_key in (("best", "selected"), ("last", "resumable")):
        record_path = _manifest_path(
            root, final_record.get(record_key),
            field=f"stages.{stage_idx}.{record_key}")
        if record_path != final_paths[final_key]:
            raise RuntimeError(
                f"completed manifest final stage {record_key} path disagrees "
                f"with final_stage.{final_key}: {path}")

    aliases = manifest.get("final_aliases")
    if not isinstance(aliases, Mapping):
        raise RuntimeError(f"completed manifest lacks final_aliases: {path}")
    for alias_name, expected_kind in (("best.pth", "final_stage_best"),
                                      ("last.pth", "final_stage_last")):
        entry = aliases.get(alias_name)
        if not isinstance(entry, Mapping):
            raise RuntimeError(
                f"completed manifest lacks final alias metadata for {alias_name}")
        alias_path = _manifest_path(
            root, entry.get("path"), field=f"final_aliases.{alias_name}.path")
        if alias_path != (root / alias_name).resolve():
            raise RuntimeError(
                f"final alias {alias_name} must resolve to {root / alias_name}, "
                f"not {alias_path}")
        expected_sha = entry.get("sha256")
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in expected_sha.lower())):
            raise RuntimeError(
                f"completed manifest has invalid SHA-256 for {alias_name}")
        if not alias_path.is_file():
            raise FileNotFoundError(f"completed final alias is missing: {alias_path}")
        if sha256_file(alias_path) != expected_sha:
            raise RuntimeError(
                f"final alias SHA-256 disagrees with manifest: {alias_path}")
        _, alias_meta = _torch_load_state(str(alias_path))
        alias_architecture = _architecture_from_metadata(
            alias_meta, ckpt_path=str(alias_path))
        _validate_checkpoint_provenance(
            alias_meta, alias_architecture, ckpt_path=str(alias_path))
        _validate_expected_architecture(
            alias_architecture, expected_architecture=manifest_architecture)
        if alias_meta.get("contract_hash") != manifest_hash:
            raise RuntimeError(
                f"final alias contract hash disagrees with manifest: {alias_path}")
        info = alias_meta["provenance"]["checkpoint"]
        if info.get("kind") != expected_kind or info.get("stage_idx") != stage_idx:
            raise RuntimeError(
                f"final alias {alias_path} does not identify final stage {stage_idx}")
    return manifest, stage_idx


def final_stage_checkpoint(
    exp_dir: Path, *, prefer_best: bool = True
) -> Tuple[str, Path]:
    """Return the final-stage alias of a completed experiment."""
    manifest, _ = _load_completed_manifest(exp_dir)
    alias_name = "best.pth" if prefer_best else "last.pth"
    alias = manifest["final_aliases"][alias_name]
    return ("final_stage_best" if prefer_best else "final_stage_last",
            _manifest_path(Path(exp_dir).resolve(), alias["path"],
                           field=f"final_aliases.{alias_name}.path"))


def build_cs_loader(
    beta: str,
    crop: Tuple[int, int],
    batch_size: int = 2,
    num_workers: int = 2,
    full_res: bool = True,
) -> DataLoader:
    """Build a Cityscapes or Foggy Cityscapes validation loader."""
    if str(beta).lower() in ("clear", "0.000", "0.0", "0"):
        clear_root = paths.get("CITYSCAPES_CLEAR_ROOT") or paths["CITYSCAPES_ROOT"]
        ds = Cityscapes(
            root=clear_root, split="val", is_train=False, crop_size=crop,
            img_folder="leftImg8bit", load_clear_reference=False,
            eval_full_res=full_res,
        )
    else:
        ds = build_cityscapes_foggy(
            paths["CITYSCAPES_ROOT"], beta, crop,
            is_train=False, load_clear=False, eval_full_res=full_res,
        )
    if len(ds) < 1:
        raise FileNotFoundError(
            f"Cityscapes-Foggy evaluation set is empty for beta={beta!r}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate,
    )


def build_acdc_val_loader(
    condition: str,
    crop: Tuple[int, int],
    batch_size: int = 2,
    num_workers: int = 2,
    full_res: bool = True,
) -> DataLoader:
    """ACDC validation loader for a single condition (has labels)."""
    if condition not in ACDC_CONDITIONS:
        raise ValueError(
            f"unknown ACDC condition {condition!r}; "
            f"expected one of {ACDC_CONDITIONS}"
        )
    ds = ACDC(
        root=paths["ACDC_ROOT"], split="val", is_train=False, crop_size=crop,
        conditions=[condition], load_clear_reference=False,
        eval_full_res=full_res,
    )
    if len(ds) < 1:
        raise FileNotFoundError(
            f"ACDC validation set is empty for condition={condition!r}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate,
    )


def _logits_from_model_output(out) -> torch.Tensor:
    if isinstance(out, dict):
        out = out["logits"]
    if isinstance(out, (tuple, list)):
        out = out[0]
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    desc: str = "",
    return_hist: bool = False,
) -> Dict:
    if hasattr(loader, "dataset") and len(loader.dataset) < 1:
        raise RuntimeError(f"{desc or 'evaluation'} loader has an empty dataset")
    was_training = model.training
    model.eval()
    try:
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        n_images = 0
        saw_spatial_padding = False
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            labels = batch.get("semantic")
            if not isinstance(labels, torch.Tensor):
                raise TypeError("evaluation semantic labels must be tensors")
            y = labels.cpu().numpy()
            if y.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"evaluation image/label row mismatch: {x.shape[0]} vs "
                    f"{y.shape[0]}")
            valid_hw = batch.get(VALID_HW_KEY)
            if valid_hw is None:
                valid_hw_np = np.tile(
                    np.asarray(y.shape[-2:], dtype=np.int64), (x.shape[0], 1))
            else:
                if not isinstance(valid_hw, torch.Tensor):
                    valid_hw = torch.as_tensor(valid_hw)
                valid_hw_np = valid_hw.cpu().numpy()
                if valid_hw_np.shape != (x.shape[0], 2):
                    raise RuntimeError(
                        "evaluation valid-H/W provenance must have shape "
                        f"({x.shape[0]}, 2), got {valid_hw_np.shape}")
            raw_out = model.predict(x) if hasattr(model, "predict") else model(x)
            logits = _logits_from_model_output(raw_out)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                raise RuntimeError(
                    "model evaluation output must resolve to [N, C, H, W] logits")
            if logits.shape[0] != x.shape[0] or logits.shape[1] != num_classes:
                raise RuntimeError(
                    "model logits disagree with evaluation contract: "
                    f"got {tuple(logits.shape)}, expected N x {num_classes} x H x W")
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear",
                    align_corners=False,
                )
            pred = logits.argmax(1).cpu().numpy()
            for i in range(pred.shape[0]):
                h, w = (int(valid_hw_np[i, 0]), int(valid_hw_np[i, 1]))
                if not 0 < h <= y.shape[-2] or not 0 < w <= y.shape[-1]:
                    raise RuntimeError(
                        f"invalid evaluation extent {(h, w)} for padded label "
                        f"shape {y.shape[-2:]}")
                saw_spatial_padding = saw_spatial_padding or (
                    (h, w) != tuple(y.shape[-2:]))
                hist += fast_hist(
                    y[i, :h, :w].flatten(), pred[i, :h, :w].flatten(),
                    num_classes,
                )
                n_images += 1
        if n_images < 1:
            raise RuntimeError(
                f"{desc or 'evaluation'} yielded zero images; refusing to emit "
                "a fabricated metric report")
        if hist.sum() < 1:
            raise RuntimeError(
                f"{desc or 'evaluation'} yielded no valid labelled pixels; "
                "check label encoding and ignore-index handling")
        iou = per_class_iou(hist)
        miou = mean_iou(hist)
        pacc = float(np.diag(hist).sum() / hist.sum())
        native = bool(getattr(getattr(loader, "dataset", None),
                              "eval_full_res", False))
        if native:
            resolution_protocol = (
                "native_with_right_bottom_padding" if saw_spatial_padding
                else "native")
        else:
            resolution_protocol = "deterministic_center_crop"
        out = {
            "desc": desc,
            "num_images": int(n_images),
            "miou": miou,
            "miou_percent": miou * 100.0,
            "pixel_acc": pacc,
            "pixel_acc_percent": pacc * 100.0,
            "per_class_iou": jsonable_iou(iou),
            "classes_present": int(np.count_nonzero(~np.isnan(iou))),
            "class_names": CS_CLASSES,
            "resolution_protocol": resolution_protocol,
        }
        if return_hist:
            out["hist"] = hist.tolist()
        return out
    finally:
        if was_training:
            model.train()


__all__ = [
    "ACDC_CONDITIONS",
    "ACDC_EXP_SUFFIXES",
    "CS_CLASSES",
    "CS_FOGGY_BETAS",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "VARIANT_NAMES",
    "VARIANT_SPEC",
    "architecture_for_report",
    "build_acdc_val_loader",
    "build_cs_loader",
    "build_model",
    "build_model_from_architecture",
    "build_model_from_checkpoint",
    "checkpoint_architecture",
    "collect_checkpoints",
    "evaluate",
    "final_stage_checkpoint",
    "load_checkpoint",
    "normalize_backbone",
    "parse_exp_name",
    "repo_relative",
    "variant_from_architecture",
]
