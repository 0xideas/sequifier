import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from torch import Tensor

PANDAS_TO_TORCH_TYPES = {"int64": torch.int64, "float64": torch.float32}


@beartype
def construct_index_maps(
    id_maps: Optional[dict[str, dict[Union[str, int], int]]],
    target_columns_index_map: list[str],
    map_to_id: Optional[bool],
) -> dict[str, dict[int, Union[str, int]]]:
    """Construct index maps for target columns."""
    index_map = {}
    if map_to_id is not None and map_to_id:
        assert id_maps is not None
        for target_column in target_columns_index_map:
            map_ = {v: k for k, v in id_maps[target_column].items()}
            val = next(iter(map_.values()))
            if isinstance(val, str):
                map_[0] = "unknown"
            else:
                assert isinstance(val, int)
                map_[0] = min(map_.values()) - 1  # type: ignore
            index_map[target_column] = map_
    return index_map


@beartype
def read_data(
    path: str, read_format: str, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Read data from CSV or Parquet file."""
    if read_format == "csv":
        return pd.read_csv(path, sep=",", decimal=".", index_col=False)
    if read_format == "parquet":
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


@beartype
def write_data(data: pd.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Write data to CSV or Parquet file."""
    if write_format == "csv":
        data.to_csv(path, sep=",", decimal=".", index=False, **kwargs)
    elif write_format == "parquet":
        data.to_parquet(path)
    else:
        raise ValueError(f"Unsupported write format: {write_format}")


@beartype
def subset_to_selected_columns(
    data: pd.DataFrame, selected_columns: list[str]
) -> pd.DataFrame:
    """Subset data to selected columns."""
    column_filters = [
        (data["inputCol"].values == input_col) for input_col in selected_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


@beartype
def numpy_to_pytorch(
    data: pd.DataFrame,
    column_types: dict[str, torch.dtype],
    selected_columns: list[str],
    target_columns: list[str],
    seq_length: int,
    device: str,
    to_device: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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

    @beartype
    def __init__(self, path: str, open_mode: str):
        self.levels = [2, 3]
        self._files = {
            level: open(path.replace("[NUMBER]", str(level)), open_mode)
            for level in self.levels
        }
        self._path = path

    @beartype
    def write(self, string: str, level: int = 3) -> None:
        """Write a string to log files of appropriate levels."""
        for level2 in self.levels:
            if level2 <= level:
                self._files[level2].write(f"{string}\n")
                self._files[level2].flush()
        if level >= 3:
            print(string)

    @beartype
    def close(self) -> None:
        """Close all open log files."""
        for file in self._files.values():
            file.close()


@beartype
def normalize_path(path: str, project_path: str) -> str:
    """Normalize a path relative to the project path."""
    project_path_normalized = (project_path + os.sep).replace(os.sep + os.sep, os.sep)
    path2 = os.path.join(project_path, path.replace(project_path_normalized, ""))
    return path2
