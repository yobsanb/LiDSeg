"""Histogram-based mIoU helpers for segmentation validation."""

from __future__ import annotations

import numpy as np


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n * n).reshape(n, n)


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """Per-class IoU; NaN where the union is empty."""
    union = hist.sum(1) + hist.sum(0) - np.diag(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(hist).astype(np.float64) / union
    return np.where(union > 0, iou, np.nan)


def mean_iou(hist: np.ndarray) -> float:
    """Mean IoU over the classes present in ``hist`` (NaN-safe)."""
    iou = per_class_iou(hist)
    return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0


def jsonable_iou(iou: np.ndarray) -> list:
    """Per-class IoU as JSON, with NaN encoded as null."""
    return [None if np.isnan(v) else float(v) for v in iou]
