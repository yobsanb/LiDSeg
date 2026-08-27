"""Dataset builders, replay sampling and the collate used by the trainer."""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              DistributedSampler, Sampler, Subset)

from dataset.segdatasets.acdc import ACDC
from dataset.segdatasets.cityscapes import Cityscapes



# Private batch keys consumed by train.val and the evaluation helpers.
EVAL_SAMPLE_INDEX_KEY = "__lidseg_eval_sample_index__"
EVAL_PADDING_KEY = "__lidseg_eval_padding__"
VALID_HW_KEY = "__lidseg_valid_hw__"


class _EvalSampleIndex(int):
    """An index carrying whether DDP inserted it solely as iteration padding."""

    def __new__(cls, value: int, *, is_padding: bool):
        obj = int.__new__(cls, value)
        obj.is_padding = bool(is_padding)
        return obj


class PaddedDistributedEvalSampler(Sampler[int]):
    """Distributed validation sampler that marks its padding rows."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(np.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if n == 0:
            return iter(())
        indices = list(range(n))
        padding = self.total_size - n
        if padding > 0:
            repeats = int(np.ceil(padding / n))
            indices.extend((indices * repeats)[:padding])
        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        positions = range(self.rank, self.total_size, self.num_replicas)
        return iter([
            _EvalSampleIndex(index, is_padding=position >= n)
            for index, position in zip(rank_indices, positions)
        ])

    def __len__(self) -> int:
        return self.num_samples


class EpochRandomSampler(Sampler[int]):
    """A replay sampler whose order is a pure function of seed and epoch."""

    def __init__(self, dataset: Dataset, *, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"replay sampler epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def _seed_epoch_worker(worker_id: int) -> None:
    """Seed Python/NumPy augmentation RNGs from a loader generator."""
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class _IndexedEvalDataset(Dataset):
    """Adds provenance keys needed to remove DDP sampler padding at metric time."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        sample_index = int(index)
        sample = self.dataset[sample_index]
        if not isinstance(sample, dict):
            raise TypeError(
                "DDP evaluation requires mapping samples so padding provenance "
                f"can be carried; got {type(sample).__name__}")
        out = dict(sample)
        out[EVAL_SAMPLE_INDEX_KEY] = sample_index
        out[EVAL_PADDING_KEY] = bool(getattr(index, "is_padding", False))
        return out

    def __getattr__(self, name):
        # Forward optional dataset hooks such as refresh_sampling.
        return getattr(self.dataset, name)


def _stack_tensors(key: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack tensors, right/bottom-padding spatial samples when necessary."""
    try:
        return torch.stack(list(values), 0)
    except RuntimeError as exc:
        if not values or any(v.ndim < 2 for v in values):
            raise RuntimeError(
                f"cannot collate field {key!r}: incompatible tensor shapes "
                f"{[tuple(v.shape) for v in values]}") from exc
        prefix = values[0].shape[:-2]
        if any(v.shape[:-2] != prefix for v in values):
            raise RuntimeError(
                f"cannot spatially pad field {key!r}: incompatible non-spatial "
                f"shapes {[tuple(v.shape) for v in values]}") from exc
        max_h = max(int(v.shape[-2]) for v in values)
        max_w = max(int(v.shape[-1]) for v in values)
        fill = 255 if key in {"semantic", "raw_label"} else 0
        out = values[0].new_full(
            (len(values), *prefix, max_h, max_w), fill_value=fill)
        for i, value in enumerate(values):
            h, w = value.shape[-2:]
            out[i, ..., :h, :w] = value
        return out


def _validate_sample_spatial_contract(sample: Dict, index: int) -> Tuple[int, int]:
    """Return image H/W after checking paired and label geometry."""
    image = sample.get("image")
    if not isinstance(image, torch.Tensor) or image.ndim < 2:
        raise TypeError(
            f"batch sample {index} must provide image as a spatial tensor; "
            f"got {type(image).__name__}")
    h, w = (int(image.shape[-2]), int(image.shape[-1]))
    semantic = sample.get("semantic")
    if isinstance(semantic, torch.Tensor) and semantic.shape[-2:] != (h, w):
        raise RuntimeError(
            f"batch sample {index} has image shape {(h, w)} but semantic shape "
            f"{tuple(semantic.shape[-2:])}")
    if bool(sample.get("has_paired_clear", False)):
        clear = sample.get("clear_image")
        if not isinstance(clear, torch.Tensor):
            raise RuntimeError(
                f"batch sample {index} claims has_paired_clear=True without "
                "a tensor clear_image")
        if clear.shape[-2:] != (h, w):
            raise RuntimeError(
                f"batch sample {index} has paired image shape {(h, w)} but "
                f"clear_image shape {tuple(clear.shape[-2:])}")
    return h, w


def collate(batch):
    if not batch:
        raise ValueError("cannot collate an empty batch")
    if any(not isinstance(b, dict) for b in batch):
        raise TypeError("LiDSeg collate expects mapping samples")

    spatial_sizes = [_validate_sample_spatial_contract(b, i)
                     for i, b in enumerate(batch)]
    out = {}
    keys = set(batch[0])
    for b in batch[1:]:
        keys &= set(b)
    had_pair_key = "has_paired_clear" in keys
    # Keep genuine pairs; insert a masked placeholder for unpaired rows.
    keys.discard("clear_image")
    keys.discard("has_paired_clear")
    for k in (k for k in batch[0] if k in keys):
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            if not all(isinstance(v, torch.Tensor) for v in vals):
                raise TypeError(f"mixed tensor/non-tensor values for {k!r}")
            out[k] = _stack_tensors(k, vals)
        elif isinstance(v0, bool):
            out[k] = torch.tensor(vals, dtype=torch.bool)
        elif isinstance(v0, (int, float)):
            out[k] = torch.tensor(vals)
        elif isinstance(v0, str):
            out[k] = vals
        elif hasattr(v0, "__array__"):
            t = [torch.from_numpy(np.asarray(v)) for v in vals]
            try:
                out[k] = torch.stack(t, 0)
            except (RuntimeError, TypeError, ValueError):
                out[k] = vals
        else:
            out[k] = vals

    pair_mask = [bool(b.get("has_paired_clear", False)) for b in batch]
    any_clear = any("clear_image" in b for b in batch)
    if any_clear:
        clear_values = []
        for i, sample in enumerate(batch):
            clear = sample.get("clear_image")
            if clear is None:
                # Shape-compatible placeholder, masked false below.
                clear = sample["image"]
            if not isinstance(clear, torch.Tensor):
                raise TypeError(
                    f"batch sample {i} clear_image must be a tensor when present")
            clear_values.append(clear)
        out["clear_image"] = _stack_tensors("clear_image", clear_values)
    elif any(pair_mask):
        raise RuntimeError(
            "has_paired_clear=True was supplied without clear_image")
    if any_clear or had_pair_key:
        out["has_paired_clear"] = torch.tensor(pair_mask, dtype=torch.bool)

    # Evaluators crop predictions back to these extents.
    out[VALID_HW_KEY] = torch.tensor(spatial_sizes, dtype=torch.int64)
    return out


def build_cityscapes_foggy(root: str, beta: str, crop: Tuple[int, int],
                           is_train: bool, load_clear: bool,
                           eval_full_res: bool = True) -> Cityscapes:
    return Cityscapes(
        root=root, split="train" if is_train else "val",
        is_train=is_train, crop_size=crop,
        img_folder=f"Foggy_{beta}/leftImg8bit",
        load_clear_reference=load_clear,
        eval_full_res=eval_full_res,
    )


def build_acdc(root: str, conditions: List[str], crop: Tuple[int, int],
               is_train: bool, load_clear: bool,
               eval_full_res: bool = True) -> ACDC:
    return ACDC(
        root=root, split="train" if is_train else "val",
        is_train=is_train, crop_size=crop, conditions=conditions,
        load_clear_reference=load_clear,
        eval_full_res=eval_full_res,
    )


def make_loader(ds: Dataset, batch_size: int, num_workers: int,
                world_size: int, rank: int,
                shuffle: bool = True, drop_last: bool = True,
                *, seed: Optional[int] = None) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    is_train_loader = bool(shuffle)
    sampler = None
    loader_ds = ds
    if world_size > 1:
        if not shuffle and not drop_last:
            loader_ds = _IndexedEvalDataset(ds)
            sampler = PaddedDistributedEvalSampler(
                loader_ds, num_replicas=world_size, rank=rank)
        else:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                         shuffle=shuffle, drop_last=drop_last,
                                         seed=0 if seed is None else int(seed))
        shuffle = False
    resamples = hasattr(ds, "refresh_sampling")
    generator = None
    worker_init_fn = None
    if is_train_loader:
        # Derive augmentation RNG from (seed, rank, epoch).
        epoch_seed = (0 if seed is None else int(seed)) + int(rank)
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
        worker_init_fn = _seed_epoch_worker
    loader = DataLoader(
        loader_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
        collate_fn=collate,
        persistent_workers=num_workers > 0 and not resamples and not is_train_loader,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if is_train_loader:
        loader._lidseg_train_seed = epoch_seed
    return loader



def make_replay_subset(
    dataset: Dataset, fraction: float = 0.1, seed: int = 0,
) -> Subset:
    """Random subset of `dataset` (uniform sampling, no replacement)."""
    n = len(dataset)
    if n < 1:
        raise ValueError("cannot build a replay subset from an empty dataset")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(
            f"replay subset fraction must be in (0, 1], got {fraction!r}")
    k = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False).tolist()
    return Subset(dataset, idx)


def build_replay_loader(
    replay_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    collate_fn,
    shuffle: bool = True,
    seed: int = 0,
) -> Optional[DataLoader]:
    """Concatenate replay subsets from prior stages into a DataLoader."""
    if not replay_datasets:
        return None
    if len(replay_datasets) == 1:
        ds = replay_datasets[0]
    else:
        ds = ConcatDataset(replay_datasets)
    if len(ds) < 1:
        raise ValueError("cannot build a replay loader from an empty dataset")
    if batch_size < 1:
        raise ValueError(f"replay batch_size must be >= 1, got {batch_size}")
    if len(ds) < batch_size:
        raise ValueError(
            "replay subset is smaller than replay_batch_size while drop_last=True; "
            "it would yield no replay batches")
    replay_seed = int(seed)
    generator = torch.Generator()
    generator.manual_seed(replay_seed)
    sampler = EpochRandomSampler(ds, seed=replay_seed) if shuffle else None
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
        # Recreate workers each epoch from the epoch-seeded generator.
        persistent_workers=False,
        worker_init_fn=_seed_epoch_worker,
        generator=generator,
    )
    # TrainingEngine sets these before each epoch.
    loader._lidseg_replay_seed = replay_seed
    return loader



class CurriculumReplayDataset(Dataset):
    """Mix current-stage data with previous stages via deterministic sampling."""

    def __init__(self, current_dataset: Dataset,
                 previous_datasets: Optional[List[Dataset]] = None,
                 replay_ratio: float = 0.2, seed: int = 12345,
                 stage_id: int = 0, stage_name: str = ""):
        if not 0.0 <= float(replay_ratio) < 1.0:
            raise ValueError(
                f"replay_ratio must be in [0, 1), got {replay_ratio!r}")
        if len(current_dataset) < 1:
            raise ValueError("current curriculum dataset is empty")
        self.current_dataset = current_dataset
        self.previous_datasets = [d for d in (previous_datasets or [])
                                  if len(d) > 0]
        self.replay_ratio = float(replay_ratio)
        self.current_ratio = 1.0 - replay_ratio
        self.seed = seed
        self.epoch = 0
        self.stage_id = stage_id
        self.stage_name = stage_name or f"Stage{stage_id}"
        if self.replay_ratio > 0.0 and not self.previous_datasets:
            raise ValueError(
                f"stage {self.stage_name!r} requests replay_ratio="
                f"{self.replay_ratio} but has no non-empty replay source")
        self.effective_length = len(self.current_dataset)
        self._precompute()

    def _precompute(self):
        rng = random.Random(self.seed + self.epoch)
        n = self.effective_length
        n_cur = int(round(n * self.current_ratio))
        n_rep = n - n_cur
        plan = [("cur", rng.randrange(len(self.current_dataset)))
                for _ in range(n_cur)]
        if self.previous_datasets and n_rep > 0:
            for _ in range(n_rep):
                ds_idx = rng.randrange(len(self.previous_datasets))
                in_idx = rng.randrange(len(self.previous_datasets[ds_idx]))
                plan.append(("prev", ds_idx, in_idx))
        else:
            for _ in range(n_rep):
                plan.append(("cur", rng.randrange(len(self.current_dataset))))
        rng.shuffle(plan)
        self.sampling_plan = plan

    def refresh_sampling(self):
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                f"curriculum epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch
        self._precompute()

    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        rec = self.sampling_plan[idx]
        if rec[0] == "cur":
            return self.current_dataset[rec[1]]
        return self.previous_datasets[rec[1]][rec[2]]



class MixedFogDataset(Dataset):
    """Randomly sample one fog level per item from a list of fog datasets."""

    def __init__(self, fog_datasets: List[Dataset], fog_levels: List[float],
                 seed: int = 12345):
        self.fog_datasets = fog_datasets
        self.fog_levels = fog_levels
        self.seed = seed
        self.epoch = 0
        self.length = len(fog_datasets[0])
        for i, ds in enumerate(fog_datasets):
            if len(ds) != self.length:
                logging.warning("MixedFog: dataset %d size %d != %d",
                                i, len(ds), self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fog_idx = random.Random(f"{self.seed}:{self.epoch}:{idx}").randrange(
            len(self.fog_datasets))
        sample = self.fog_datasets[fog_idx][idx]
        sample["fog_level"] = self.fog_levels[fog_idx]
        return sample

    def refresh_sampling(self):
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                f"mixed-fog epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch
