"""Portable environment snapshot comparison."""

import math

import numpy as np
import torch


def snapshots_equal(left, right):
    """Exact adapters use torch-serializable scalar, tensor and container state."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and bool(torch.isfinite(left).all())
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray):
        finite = (
            np.isfinite(left).all()
            if left.dtype.kind in "biufc"
            else left.dtype.kind in "US"
        )
        return (
            left.dtype == right.dtype and bool(finite) and np.array_equal(left, right)
        )
    if isinstance(left, np.generic):
        return snapshots_equal(left.item(), right.item())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            snapshots_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            snapshots_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return math.isfinite(left) and left == right
    if isinstance(left, (str, int, bool, type(None))):
        return left == right
    raise ValueError(
        "exact adapter snapshots must use scalar, tensor, NumPy array, dict, list, or tuple values"
    )
