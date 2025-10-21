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
    """Constructs reverse index maps (int index to original ID).

    This function creates reverse mappings from the integer indices back to
    the original string or integer identifiers. It only performs this
    operation if `map_to_id` is True and `id_maps` is provided.

    A special mapping for index 0 is added:
    - If original IDs are strings, 0 maps to "unknown".
    - If original IDs are integers, 0 maps to (minimum original ID) - 1.

    Args:
        id_maps: A nested dictionary mapping column names to their
            respective ID-to-index maps (e.g.,
            `{'col_name': {'original_id': 1, ...}}`). Expected to be
            provided if `map_to_id` is True.
        target_columns_index_map: A list of column names for which to
            construct the reverse maps.
        map_to_id: A boolean flag. If True, the reverse maps are
            constructed. If False or None, an empty dictionary is returned.

    Returns:
        A dictionary where keys are column names from
        `target_columns_index_map` and values are the reverse maps
        (index-to-original-ID). Returns an empty dict if `map_to_id`
        is not True.

    Raises:
        AssertionError: If `map_to_id` is True but `id_maps` is None.
        AssertionError: If the values of a map are not consistently
            string or integer (excluding the added '0' key).
    """
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
    """Reads data from a CSV or Parquet file into a Polars DataFrame.

    Args:
        path: The file path to read from.
        read_format: The format of the file. Supported formats are
            "csv" and "parquet".
        columns: An optional list of column names to read. This argument
            is only used when `read_format` is "parquet".

    Returns:
        A Polars DataFrame containing the data from the file.

    Raises:
        ValueError: If `read_format` is not "csv" or "parquet".
    """
    if read_format == "csv":
        return pl.read_csv(path, separator=",")
    if read_format == "parquet":
        return pl.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported read format: {read_format}")


@beartype
def write_data(data: pl.DataFrame, path: str, write_format: str, **kwargs) -> None:
    """Writes a Polars (or Pandas) DataFrame to a CSV or Parquet file.

    This function detects the type of the input DataFrame.
    - For Polars DataFrames, it uses `.write_csv()` or `.write_parquet()`.
    - For other DataFrame types (presumably Pandas), it uses `.to_csv()`
      or `.to_parquet()`.

    Note: The type hint specifies `pl.DataFrame`, but the implementation
    includes a fallback path that suggests compatibility with Pandas
    DataFrames.

    Args:
        data: The Polars (or Pandas) DataFrame to write.
        path: The destination file path.
        write_format: The format to write. Supported formats are
            "csv" and "parquet".
        **kwargs: Additional keyword arguments passed to the underlying
            write function (e.g., `write_csv` for Polars, `to_csv` for
            Pandas).

    Returns:
        None.

    Raises:
        ValueError: If `write_format` is not "csv" or "parquet".
    """
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
    """Filters a DataFrame to rows where 'inputCol' is in a selected list.

    This function supports both Polars (DataFrame, LazyFrame) and Pandas
    DataFrames, dispatching to the appropriate filtering method.

    - For Polars objects, it uses `data.filter(pl.col("inputCol").is_in(...))`.
    - For other objects (presumably Pandas), it builds a numpy boolean
      mask and filters using `data.loc[...]`.

    Note: The type hint only specifies Polars objects, but the
    implementation includes a fallback path for Pandas-like objects.

    Args:
        data: The Polars (DataFrame, LazyFrame) or Pandas DataFrame to
            filter. It must contain a column named "inputCol".
        selected_columns: A list of values. Rows will be kept if their
            value in "inputCol" is present in this list.

    Returns:
        A filtered DataFrame or LazyFrame of the same type as the input.
    """
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
    """Converts a long-format Polars DataFrame to a dict of sequence tensors.

    This function assumes the input DataFrame `data` is in a long format
    where each row represents a sequence for a specific feature. It expects
    a column named "inputCol" that contains the feature name (e.g.,
    'price', 'volume') and other columns representing time steps (e.g.,
    "0", "1", ..., "L").

    It generates two tensors for each column in `all_columns`:
    1.  An "input" tensor (from time steps L down to 1).
    2.  A "target" tensor (from time steps L-1 down to 0).

    Example:
        For `seq_length = 3` and `all_columns = ['price']`, it will create:
        - 'price': Tensor from columns ["3", "2", "1"]
        - 'price_target': Tensor from columns ["2", "1", "0"]

    Args:
        data: The long-format Polars DataFrame. Must contain "inputCol"
            and columns named as strings of integers for time steps.
        column_types: A dictionary mapping feature names (from "inputCol")
            to their desired `torch.dtype`.
        all_columns: A list of all feature names (from "inputCol") to
            be processed and converted into tensors.
        seq_length: The total sequence length (L). This determines the
            column names for time steps (e.g., "0" to "L").

    Returns:
        A dictionary mapping feature names to their corresponding PyTorch
        tensors. Target tensors are stored with a `_target` suffix
        (e.g., `{'price': <tensor>, 'price_target': <tensor>}`).
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
    """Manages logging to multiple files based on verbosity levels.

    This class opens multiple log files based on a path template and a
    hardcoded list of levels (2 and 3). Messages are written to files
    based on their assigned level, and high-level messages are also
    printed to the console on the main process (rank 0).

    Attributes:
        rank (Optional[int]): The rank of the current process, used to
            control console output.
        levels (list[int]): The hardcoded list of log levels [2, 3]
            for which files are created.
        _files (dict[int, io.TextIOWrapper]): A dictionary mapping log
            levels to their open file handlers.
        _path (str): The original path template provided.
    """

    @beartype
    def __init__(self, path: str, open_mode: str, rank: Optional[int] = None):
        """Initializes the LogFile and opens log files.

        The `path` argument should be a template containing "[NUMBER]",
        which will be replaced by the log levels (2 and 3) to create
        separate log files.

        Args:
            path: The path template for the log files (e.g.,
                "run_log_[NUMBER].txt").
            open_mode: The mode for opening the log files (e.g., "a", "w").
            rank: The rank of the current process (e.g., in distributed
                training). If None or 0, high-level messages will be
                printed to stdout.
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
        """Writes a string to log files and potentially the console.

        The string is written to all log files whose level is less than
        or equal to the specified `level`.

        - A message with `level=2` goes to file 2.
        - A message with `level=3` goes to file 2 and file 3.

        If `level` is 3 or greater, the message is also printed to stdout
        if `self.rank` is None or 0.

        Args:
            string: The message to log.
            level: The verbosity level of the message. Defaults to 3.
        """
        for level2 in self.levels:
            if level2 <= level:
                self._files[level2].write(f"{string}\n")
                self._files[level2].flush()
        if level >= 3:
            if self.rank is None or self.rank == 0:
                print(string)

    @beartype
    def close(self) -> None:
        """Closes all open log file handlers."""
        for file in self._files.values():
            file.close()


@beartype
def normalize_path(path: str, project_path: str) -> str:
    """Normalizes a path to be relative to a project path, then joins them.

    This function ensures that a given `path` is correctly expressed as
    an absolute path rooted at `project_path`. It does this by first
    removing the `project_path` prefix from `path` (if it exists)
    and then joining the result back to `project_path`.

    This is useful for handling paths that might be provided as either
    relative (e.g., "data/file.txt") or absolute
    (e.g., "/abs/path/to/project/data/file.txt").

    Args:
        path: The path to normalize.
        project_path: The absolute path to the project's root directory.

    Returns:
        A normalized, absolute path.
    """
    project_path_normalized = (project_path + os.sep).replace(os.sep + os.sep, os.sep)
    path2 = os.path.join(project_path, path.replace(project_path_normalized, ""))
    return path2
