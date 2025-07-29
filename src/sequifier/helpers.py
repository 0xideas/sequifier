import os
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
from beartype import beartype
from torch import Tensor

PANDAS_TO_TORCH_TYPES = {
    "Int64": torch.int64,
    "Float64": torch.float32,
    "int64": torch.int64,
    "float64": torch.float32,
}


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
) -> pl.DataFrame:
    """Read data from CSV or Parquet file."""
    if read_format == "csv":
        return pl.read_csv(path, separator=",")
    if read_format == "parquet":
        return pl.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


@beartype
def write_data(data: pl.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Write data to CSV or Parquet file."""
    if isinstance(data, pl.DataFrame):
        if write_format == "csv":
            data.write_csv(path, **kwargs)
        elif write_format == "parquet":
            data.write_parquet(path)
        else:
            raise ValueError(
                f"Unsupported write format for Polars DataFrame: {write_format}"
            )
        return

    if write_format == "csv":
        data.to_csv(path, separator=",", index=False, **kwargs)
    elif write_format == "parquet":
        data.to_parquet(path)
    else:
        raise ValueError(f"Unsupported write format: {write_format}")


@beartype
def subset_to_selected_columns(
    data: Union[pl.DataFrame, pl.LazyFrame], selected_columns: list[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Subset data to selected columns."""
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data.filter(pl.col("inputCol").is_in(selected_columns))

    column_filters = [
        (data["inputCol"].values == input_col) for input_col in selected_columns
    ]
    filter_ = np.logical_or.reduce(column_filters)
    return data.loc[filter_, :]


@beartype
def numpy_to_pytorch(
    data: pl.DataFrame,
    column_types: dict[str, torch.dtype],
    selected_columns: list[str],
    target_columns: list[str],
    seq_length: int,
    device: str,
    to_device: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Converts data from a Polars DataFrame to PyTorch Tensors based on specified columns.

    This function processes a DataFrame that is structured in a "long" format,
    where one column ('inputCol') identifies the feature or target type, and other
    columns ('0', '1', '2', ...) contain the sequence data.
    """
    targets = {}
    # Define the column names for the target sequences, e.g., ['29', '28', ..., '0']
    target_seq_cols = [str(c) for c in range(seq_length - 1, -1, -1)]

    for target_column in target_columns:
        # Filter for the target, select sequence columns, and convert to a tensor
        target_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == target_column)
            .select(target_seq_cols)
            .to_numpy(),
            dtype=column_types[target_column],
        )

        if to_device:
            target_tensor = target_tensor.to(device)
        targets[target_column] = target_tensor

    sequence = {}
    # Define the column names for the input sequences, e.g., ['30', '29', ..., '1']
    input_seq_cols = [str(c) for c in range(seq_length, 0, -1)]

    for col in selected_columns:
        # Filter for the feature, select sequence columns, and convert to a tensor
        feature_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col).select(input_seq_cols).to_numpy(),
            dtype=column_types[col],
        )

        if to_device:
            feature_tensor = feature_tensor.to(device)
        sequence[col] = feature_tensor

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
