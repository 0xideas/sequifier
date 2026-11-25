"""Lookup of processed per-sequence features for model windows."""

import os
from functools import lru_cache
from typing import Any

import polars as pl
import torch
from torch import Tensor

from sequifier.helpers import PANDAS_TO_TORCH_TYPES, normalize_path
from sequifier.io.sequence import validate_sequence_ids
from sequifier.io.tensors import gather_tensor


class StaticFeatureTable:
    def __init__(self, path: str, columns: tuple[str, ...], dtypes: tuple[str, ...]):
        frame = pl.read_parquet(path).sort("sequenceId")
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Static table lacks columns: {sorted(missing)}")
        validate_sequence_ids(frame)
        self.sequence_ids = torch.tensor(
            frame.get_column("sequenceId").to_numpy(), dtype=torch.int64
        )
        if self.sequence_ids.numel() != torch.unique(self.sequence_ids).numel():
            raise ValueError("Processed static table has duplicate sequenceId values")
        self.values = {}
        for column, dtype in zip(columns, dtypes):
            self.values[column] = torch.tensor(
                frame.get_column(column).to_numpy(), dtype=PANDAS_TO_TORCH_TYPES[dtype]
            )

    def lookup(self, sequence_ids: Tensor) -> dict[str, Tensor]:
        requested = sequence_ids.to(device="cpu", dtype=torch.int64)
        positions = torch.searchsorted(self.sequence_ids, requested)
        if self.sequence_ids.numel() == 0:
            raise ValueError("Processed static table is empty")
        safe = positions.clamp(max=self.sequence_ids.numel() - 1)
        if not torch.equal(self.sequence_ids[safe], requested):
            missing = requested[self.sequence_ids[safe] != requested].unique().tolist()
            raise ValueError(f"Static table lacks sequenceId values: {missing[:10]}")
        return {
            column: gather_tensor(value, safe) for column, value in self.values.items()
        }


@lru_cache(maxsize=8)
def _load(
    path: str,
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    _file_signature: tuple[int, int, int, int, int],
) -> StaticFeatureTable:
    return StaticFeatureTable(path, columns, dtypes)


def table_for_config(config: Any) -> StaticFeatureTable | None:
    try:
        columns = tuple(config.static_columns)
    except AttributeError:
        return None
    if not columns:
        return None
    try:
        path = config.static_features_path
    except AttributeError:
        path = None
    if path is None:
        raise ValueError("Static conditioning requires static_features_path")
    dtypes = tuple(config.column_data_types[column] for column in columns)
    resolved_path = normalize_path(path, config.project_root)
    stat = os.stat(resolved_path)
    signature = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    return _load(resolved_path, columns, dtypes, signature)
