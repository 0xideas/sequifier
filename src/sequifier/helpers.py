import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

PANDAS_TO_TORCH_TYPES = {"int64": torch.int64, "float64": torch.float32}


def construct_index_maps(
    id_maps: Dict[str, Dict],
    target_columns_index_map: List[str],
    map_to_id: Optional[bool]
) -> Dict[str, Dict]:
    """Construct index maps for target columns."""
    index_map = {}
    if map_to_id is not None:
        for target_column in target_columns_index_map:
            map_ = (
                {v: k for k, v in id_maps[target_column].items()} if map_to_id else None
            )
            if isinstance(next(iter(map_.values())), str):
                map_[0] = "unknown"
            else:
                map_[0] = min(map_.values()) - 1
            index_map[target_column] = map_
    return index_map


def read_data(path: str, read_format: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read data from CSV or Parquet file."""
    if read_format == "csv":
        return pd.read_csv(path, sep=",", decimal=".", index_col=False)
    if read_format == "parquet":
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


def write_data(data: pd.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Write data to CSV or Parquet file."""
    if write_format == "csv":
        data.to_csv(path, sep=",", decimal=".", index=False, **kwargs)
    elif write_format == "parquet":
        data.to_parquet(path)
    else:
        raise ValueError(f"Unsupported write format: {write_format}")


def subset_to_selected_columns(data: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """Subset data to selected columns."""
    column_filters = [
        (data["inputCol"].values == input_col) for input_col in selected_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


def numpy_to_pytorch(
    data: pd.DataFrame,
    column_types: Dict[str, torch.dtype],
    selected_columns: List[str],
    target_columns: List[str],
    seq_length: int,
    device: torch.device,
    to_device: bool
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Convert numpy data to PyTorch tensors."""
    targets = {}
    for target_column in target_columns:
        target = torch.tensor(
            data.query(f"inputCol=='{target_column}'")[
                [str(c) for c in range(seq_length - 1, -1, -1)]
            ].values
        ).to(column_types[target_column])
        if to_device:
            target = target.to(device)
        targets[target_column] = target

    sequence = {}
    for col in selected_columns:
        f = data["inputCol"].values == col
        data_subset = data.loc[f, [str(c) for c in range(seq_length, 0, -1)]].values

        tens = torch.tensor(data_subset).to(column_types[col])

        if to_device:
            tens = tens.to(device)

        sequence[col] = tens

    return sequence, targets


class LogFile:
    """A class for logging to multiple files with different levels."""

    def __init__(self, path: str, open_mode: str):
        self.levels = [2, 3]
        self._files = {
            level: open(path.replace("[NUMBER]", str(level)), open_mode)
            for level in self.levels
        }
        self._path = path

    def write(self, string: str, level: int = 3) -> None:
        """Write a string to log files of appropriate levels."""
        for level2 in self.levels:
            if level2 <= level:
                self._files[level2].write(f"{string}\n")
                self._files[level2].flush()
        if level >= 3:
            print(string)

    def close(self) -> None:
        """Close all open log files."""
        for file in self._files.values():
            file.close()


def normalize_path(path: str, project_path: str) -> str:
    """Normalize a path relative to the project path."""
    project_path_normalized = os.path.normpath(project_path) + os.sep
    return os.path.normpath(os.path.join(project_path, path.replace(project_path_normalized, "")))
