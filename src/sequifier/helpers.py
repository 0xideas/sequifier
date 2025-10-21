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
    all_columns: list[str],  # Changed from selected_columns, target_columns
    seq_length: int,
) -> dict[str, Tensor]:  # Now returns a single dictionary
    """Converts a Polars DataFrame to a dictionary of PyTorch tensors.

    Args:
        data: The Polars DataFrame to convert.
        column_types: A dictionary mapping column names to PyTorch dtypes.
        all_columns: A list of all columns to include.
        seq_length: The sequence length.

    Returns:
        A dictionary of PyTorch tensors.
    """
    # Define both input and target sequence column names
    input_seq_cols = [str(c) for c in range(seq_length, 0, -1)]
    target_seq_cols = [str(c) for c in range(seq_length - 1, -1, -1)]

    # We will create a unified dictionary
    unified_tensors = {}

    for col_name in all_columns:
        # Create the input sequence tensor (e.g., from t=1 to t=L)
        input_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(input_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[col_name] = input_tensor

        # Create the target sequence tensor (e.g., from t=0 to t=L-1)
        # We'll store it with a "_target" suffix to distinguish it
        target_tensor = torch.tensor(
            data.filter(pl.col("inputCol") == col_name)
            .select(target_seq_cols)
            .to_numpy(),
            dtype=column_types[col_name],
        )
        unified_tensors[f"{col_name}_target"] = target_tensor

    return unified_tensors


class LogFile:
    """A class for logging to multiple files with different levels."""

    @beartype
    def __init__(self, path: str, open_mode: str, rank: Optional[int] = None):
        """Initializes the LogFile.

        Args:
            path: The path to the log file.
            open_mode: The open mode for the log file.
            rank: The rank of the current process.
        """
        self.rank = rank
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
            if self.rank is None or self.rank == 0:
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
