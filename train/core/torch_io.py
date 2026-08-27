"""Checkpoint deserialization helpers."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import torch


def _validate_path(path: str | os.PathLike) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist or is not a file: {candidate}")
    return candidate


def _torch_load(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device,
    weights_only: bool,
) -> Any:
    candidate = _validate_path(path)
    kwargs: dict[str, Any] = {"map_location": map_location}
    supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    if supports_weights_only:
        kwargs["weights_only"] = weights_only
    elif weights_only:
        raise RuntimeError(
            "this PyTorch build has no weights_only loader; refusing to "
            f"unsafely deserialize external checkpoint {candidate}")
    try:
        return torch.load(os.fspath(candidate), **kwargs)
    except Exception as exc:
        mode = "trusted pickle" if not weights_only else "safe weights-only"
        raise RuntimeError(
            f"could not load {candidate} with {mode} deserialization: {exc}") from exc


def torch_load_weights(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load external weights with ``weights_only=True``."""
    return _torch_load(path, map_location=map_location, weights_only=True)


def torch_load_checkpoint(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device = "cpu",
    trusted: bool = False,
) -> Any:
    """Load a checkpoint; ``trusted=True`` allows a full pickle load."""
    return _torch_load(path, map_location=map_location, weights_only=not trusted)
