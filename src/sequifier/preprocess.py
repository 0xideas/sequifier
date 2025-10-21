import json
import math
import multiprocessing
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from beartype import beartype

from sequifier.config.preprocess_config import load_preprocessor_config
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, read_data, write_data


@beartype
def preprocess(args: Any, args_config: dict[str, Any]) -> None:
    """Main preprocessing function.

    Args:
        args: Command line arguments.
        args_config: Configuration dictionary.
    """
    print("--- Starting Preprocessing ---")
    config_path = args.config_path or "configs/preprocess.yaml"
    config = load_preprocessor_config(config_path, args_config)
    Preprocessor(**config.dict())
    print("--- Preprocessing Complete ---")


class Preprocessor:
    """A class for preprocessing data for the sequifier model.

    This class handles loading, preprocessing, and saving data. It supports
    single-file and multi-file processing, and can handle large datasets by
    processing them in batches.

    Attributes:
        project_path (str): The path to the sequifier project directory.
        batches_per_file (int): The number of batches to process per file.
        data_name_root (str): The root name of the data file.
        combine_into_single_file (bool): Whether to combine the output into a single file.
        target_dir (str): The target directory for temporary files.
        seed (int): The random seed for reproducibility.
        n_cores (int): The number of cores to use for parallel processing.
        split_paths (list[str]): The paths to the output split files.
    """

    @beartype
    def __init__(
        self,
        project_path: str,
        data_path: str,
        read_format: str,
        write_format: str,
        combine_into_single_file: bool,
        selected_columns: Optional[list[str]],
        group_proportions: list[float],
        seq_length: int,
        seq_step_sizes: list[int],
        max_rows: Optional[int],
        seed: int,
        n_cores: Optional[int],
        batches_per_file: int,
        process_by_file: bool,
    ):
        """Initializes the Preprocessor with the given parameters.

        Args:
            project_path: The path to the sequifier project directory.
            data_path: The path to the input data file.
            read_format: The file type of the input data.
            write_format: The file type for the preprocessed output data.
            combine_into_single_file: Whether to combine the output into a single file.
            selected_columns: A list of columns to be included in the preprocessing.
            group_proportions: A list of floats that define the relative sizes of data splits.
            seq_length: The sequence length for the model inputs.
            seq_step_sizes: A list of step sizes for creating subsequences.
            max_rows: The maximum number of input rows to process.
            seed: A random seed for reproducibility.
            n_cores: The number of CPU cores to use for parallel processing.
            batches_per_file: The number of batches to process per file.
            process_by_file: A flag to indicate if processing should be done file by file.
        """
        self.project_path = project_path
        self.batches_per_file = batches_per_file

        self.data_name_root = os.path.splitext(os.path.basename(data_path))[0]
        self.combine_into_single_file = combine_into_single_file
        if self.combine_into_single_file:
            self.target_dir = "temp"
        else:
            assert write_format == "pt"
            self.target_dir = f"{self.data_name_root}-temp"

        self.seed = seed
        np.random.seed(seed)
        self.n_cores = n_cores or multiprocessing.cpu_count()
        self._setup_directories()

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns

        self._setup_split_paths(write_format, len(group_proportions))

        if os.path.isfile(data_path):
            data = _load_and_preprocess_data(
                data_path, read_format, selected_columns, max_rows
            )
            data_columns = [
                col for col in data.columns if col not in ["sequenceId", "itemPosition"]
            ]
            id_maps, selected_columns_statistics = {}, {}
            id_maps, selected_columns_statistics = _get_column_statistics(
                data, data_columns, id_maps, selected_columns_statistics, 0
            )

            data, n_classes, col_types = _apply_column_statistics(
                data, data_columns, id_maps, selected_columns_statistics
            )

            self._export_metadata(
                id_maps, n_classes, col_types, selected_columns_statistics
            )

            schema = self._create_schema(col_types, seq_length)

            data = data.sort(["sequenceId", "itemPosition"])
            n_batches = _process_batches_single_file(
                self.project_path,
                self.data_name_root,
                data,
                schema,
                self.n_cores,
                seq_length,
                seq_step_sizes,
                data_columns,
                col_types,
                group_proportions,
                write_format,
                self.split_paths,
                self.target_dir,
                self.batches_per_file,
            )

            if self.combine_into_single_file:
                input_files = create_file_paths_for_single_file(
                    self.project_path,
                    self.target_dir,
                    len(group_proportions),
                    n_batches,
                    self.data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    self.project_path,
                    self.target_dir,
                    len(group_proportions),
                    input_files,
                    self.data_name_root,
                    write_format,
                    in_target_dir=False,
                )
                delete_files(input_files)

        else:
            n_classes, id_maps, selected_columns_statistics, col_types, data_columns = (
                self._get_column_metadata_across_files(
                    data_path, read_format, max_rows, selected_columns
                )
            )
            self._export_metadata(
                id_maps, n_classes, col_types, selected_columns_statistics
            )
            schema = self._create_schema(col_types, seq_length)

            files_to_process = self._get_files_to_process(data_path, read_format)

            self._process_batches_multiple_files(
                files_to_process,
                read_format,
                selected_columns,
                max_rows,
                schema,
                self.n_cores,
                seq_length,
                seq_step_sizes,
                data_columns,
                n_classes,
                id_maps,
                selected_columns_statistics,
                col_types,
                group_proportions,
                write_format,
                process_by_file,
            )

        self._cleanup(write_format)

    @beartype
    def _get_files_to_process(self, data_path: str, read_format: str) -> list[str]:
        """Gets a list of files to process from a given data path.

        Args:
            data_path: The path to the data directory.
            read_format: The file format to look for.

        Returns:
            A list of file paths to process.
        """
        paths_to_process = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(read_format):
                    paths_to_process.append(os.path.join(root, file))
        return paths_to_process

    @beartype
    def _create_schema(
        self, col_types: dict[str, str], seq_length: int
    ) -> dict[str, Any]:
        """Creates the schema for the preprocessed data.

        Args:
            col_types: A dictionary of column types.
            seq_length: The sequence length.

        Returns:
            A dictionary representing the schema.
        """
        schema = {
            "sequenceId": pl.Int64,
            "subsequenceId": pl.Int64,
            "inputCol": pl.String,
        }

        if (np.unique(list(col_types.values())) == np.array(["Int64"]))[0]:
            sequence_position_type = pl.Int64
        else:
            assert np.all(
                [
                    type_.startswith("Int") or type_.startswith("Float")
                    for type_ in col_types.values()
                ]
            )
            sequence_position_type = pl.Float64

        schema.update(
            {str(i): sequence_position_type for i in range(seq_length - 1, -1, -1)}
        )

        return schema

    @beartype
    def _get_column_metadata_across_files(
        self,
        data_path: str,
        read_format: str,
        max_rows: Optional[int],
        selected_columns: Optional[list[str]],
    ) -> tuple[
        dict[str, int],
        dict[str, dict[Union[str, int], int]],
        dict[str, dict[str, float]],
        dict[str, str],
        list[str],
    ]:
        """Gets column metadata across multiple files.

        Args:
            data_path: The path to the data directory.
            read_format: The file format to look for.
            max_rows: The maximum number of rows to process.
            selected_columns: A list of columns to be included in the preprocessing.

        Returns:
            A tuple containing the number of classes, id maps, column statistics, column types, and data columns.
        """
        n_rows_running_count = 0
        id_maps, selected_columns_statistics = {}, {}
        col_types, data_columns = None, None

        print(f"[INFO] Data path: {data_path}")
        for root, dirs, files in os.walk(data_path):
            print(f"[INFO] N Files : {len(files)}")
            for file in files:
                if file.endswith(read_format) and (
                    max_rows is None or n_rows_running_count < max_rows
                ):
                    print(f"[INFO] Preprocessing: reading {file}")
                    max_rows_inner = (
                        None
                        if max_rows is None
                        else max(0, max_rows - n_rows_running_count)
                    )
                    data = _load_and_preprocess_data(
                        os.path.join(root, file),
                        read_format,
                        selected_columns,
                        max_rows_inner,
                    )
                    data_columns = [
                        col
                        for col in data.columns
                        if col not in ["sequenceId", "itemPosition"]
                    ]

                    if col_types is None:
                        col_types = {col: str(data.schema[col]) for col in data_columns}
                    else:
                        for col in data_columns:
                            assert (
                                str(data.schema[col]) == col_types[col]
                            ), f"{str(data.schema[col]) = } != {col_types[col] = }"

                    id_maps, selected_columns_statistics = _get_column_statistics(
                        data,
                        data_columns,
                        id_maps,
                        selected_columns_statistics,
                        n_rows_running_count,
                    )
                    n_rows_running_count += data.shape[0]
        assert data_columns is not None
        n_classes = {col: len(id_maps[col]) + 1 for col in id_maps}
        assert col_types is not None
        return (
            n_classes,
            id_maps,
            selected_columns_statistics,
            col_types,
            data_columns,
        )

    @beartype
    def _setup_directories(self) -> None:
        """Sets up the directories for the preprocessed data."""
        os.makedirs(os.path.join(self.project_path, "data"), exist_ok=True)
        temp_path = os.path.join(self.project_path, "data", self.target_dir)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)

    @beartype
    def _setup_split_paths(self, write_format: str, n_splits: int) -> None:
        """Sets up the paths for the output split files.

        Args:
            write_format: The file format for the output files.
            n_splits: The number of splits.
        """
        split_paths = [
            os.path.join(
                self.project_path,
                "data",
                f"{self.data_name_root}-split{i}.{write_format}",
            )
            for i in range(n_splits)
        ]

        self.split_paths = split_paths

    @beartype
    def _process_batches_multiple_files(
        self,
        file_paths: list[str],
        read_format: str,
        selected_columns: Optional[list[str]],
        max_rows: Optional[int],
        schema: Any,
        n_cores: int,
        seq_length: int,
        seq_step_sizes: list[int],
        data_columns: list[str],
        n_classes: dict[str, int],
        id_maps: dict[str, dict[Union[int, str], int]],
        selected_columns_statistics: dict[str, dict[str, float]],
        col_types: dict[str, str],
        group_proportions: list[float],
        write_format: str,
        process_by_file: bool = True,
    ) -> None:
        """Processes batches of data from multiple files.

        Args:
            file_paths: A list of file paths to process.
            read_format: The file format to read.
            selected_columns: A list of columns to be included in the preprocessing.
            max_rows: The maximum number of rows to process.
            schema: The schema for the preprocessed data.
            n_cores: The number of cores to use for parallel processing.
            seq_length: The sequence length for the model inputs.
            seq_step_sizes: A list of step sizes for creating subsequences.
            data_columns: A list of data columns.
            n_classes: A dictionary containing the number of classes for each categorical column.
            id_maps: A dictionary containing the id maps for each categorical column.
            selected_columns_statistics: A dictionary containing the statistics for each numerical column.
            col_types: A dictionary containing the column types.
            group_proportions: A list of floats that define the relative sizes of data splits.
            write_format: The file format for the output files.
            process_by_file: A flag to indicate if processing should be done file by file.
        """
        if process_by_file:
            _process_batches_multiple_files_inner(
                project_path=self.project_path,
                data_name_root=self.data_name_root,
                process_id=0,
                file_paths=file_paths,
                read_format=read_format,
                selected_columns=selected_columns,
                max_rows=max_rows,
                schema=schema,
                n_cores=n_cores,
                seq_length=seq_length,
                seq_step_sizes=seq_step_sizes,
                data_columns=data_columns,
                n_classes=n_classes,
                id_maps=id_maps,
                selected_columns_statistics=selected_columns_statistics,
                col_types=col_types,
                group_proportions=group_proportions,
                write_format=write_format,
                split_paths=self.split_paths,
                target_dir=self.target_dir,
                batches_per_file=self.batches_per_file,
                combine_into_single_file=self.combine_into_single_file,
            )
            input_files = create_file_paths_for_multiple_files2(
                self.project_path,
                self.target_dir,
                len(group_proportions),
                1,
                {0: len(file_paths)},
                self.data_name_root,
                write_format,
            )
        else:
            assert process_by_file is False
            n_file_sets = (len(file_paths) // n_cores) + 1

            file_sets = [
                file_paths[i : i + n_file_sets]
                for i in range(0, len(file_paths), n_file_sets)
            ]

            kwargs_1 = {
                "project_path": self.project_path,
                "data_name_root": self.data_name_root,
            }
            kwargs_2 = {
                "read_format": read_format,
                "selected_columns": selected_columns,
                "max_rows": max_rows,
                "schema": schema,
                "n_cores": 1,
                "seq_length": seq_length,
                "seq_step_sizes": seq_step_sizes,
                "data_columns": data_columns,
                "n_classes": n_classes,
                "id_maps": id_maps,
                "selected_columns_statistics": selected_columns_statistics,
                "col_types": col_types,
                "group_proportions": group_proportions,
                "write_format": write_format,
                "split_paths": self.split_paths,
                "target_dir": self.target_dir,
                "batches_per_file": self.batches_per_file,
                "combine_into_single_file": self.combine_into_single_file,
            }

            job_params = [
                list(kwargs_1.values())
                + [process_id, file_set]
                + list(kwargs_2.values())
                for process_id, file_set in enumerate(file_sets)
            ]
            print(f"[INFO] _process_batches_multiple_files n_cores: {n_cores}")
            print(f"[INFO] _process_batches_multiple_files {len(job_params) = }")

            with multiprocessing.get_context("spawn").Pool(
                processes=len(job_params)
            ) as pool:
                pool.starmap(_process_batches_multiple_files_inner, job_params)

            input_files = create_file_paths_for_multiple_files2(
                self.project_path,
                self.target_dir,
                len(group_proportions),
                len(job_params),
                {i: len(file_sets[i]) for i in range(len(file_sets))},
                self.data_name_root,
                write_format,
            )
        if self.combine_into_single_file:
            combine_multiprocessing_outputs(
                self.project_path,
                self.target_dir,
                len(group_proportions),
                input_files,
                self.data_name_root,
                write_format,
                in_target_dir=False,
            )
            delete_files(input_files)

    @beartype
    def _cleanup(self, write_format: str) -> None:
        """Cleans up the temporary files and directories.

        Args:
            write_format: The file format of the output files.
        """
        temp_output_path = os.path.join(self.project_path, "data", self.target_dir)
        directory = Path(temp_output_path)

        if not self.target_dir == "temp":
            assert write_format == "pt"
            for i, split_path in enumerate(self.split_paths):
                split = f"split{i}"
                folder_path = os.path.join(
                    self.project_path, "data", f"{self.data_name_root}-{split}"
                )
                assert folder_path in split_path
                os.makedirs(folder_path, exist_ok=True)

                pattern = re.compile(rf".+split{i}-\d+-\d+\.\w+")

                for file_path in directory.iterdir():
                    if file_path.is_file() and pattern.match(file_path.name):
                        destination = Path(folder_path) / file_path.name
                        shutil.move(str(file_path), str(destination))

                self._create_metadata_for_folder(folder_path)

        if not os.listdir(directory) or self.target_dir == "temp":
            shutil.rmtree(directory)

    @beartype
    def _export_metadata(
        self,
        id_maps: dict[str, dict[Union[str, int], int]],
        n_classes: dict[str, int],
        col_types: dict[str, str],
        selected_columns_statistics: dict[str, dict[str, float]],
    ) -> None:
        """Exports the metadata to a JSON file.

        Args:
            id_maps: A dictionary containing the id maps for each categorical column.
            n_classes: A dictionary containing the number of classes for each categorical column.
            col_types: A dictionary containing the column types.
            selected_columns_statistics: A dictionary containing the statistics for each numerical column.
        """
        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "split_paths": [
                split_path.replace(".pt", "") for split_path in self.split_paths
            ],
            "column_types": col_types,
            "selected_columns_statistics": selected_columns_statistics,
        }
        os.makedirs(
            os.path.join(self.project_path, "configs", "ddconfigs"), exist_ok=True
        )

        with open(
            os.path.join(
                self.project_path, "configs", "ddconfigs", f"{self.data_name_root}.json"
            ),
            "w",
        ) as f:
            json.dump(data_driven_config, f)

    @beartype
    def _create_metadata_for_folder(self, folder_path: str) -> None:
        """Scans a directory for .pt files, counts samples in each, and writes metadata.json."""
        batch_files_metadata = []
        total_samples = 0
        directory = Path(folder_path)

        # Find all .pt files in the target folder
        pt_files = sorted(
            [f for f in directory.iterdir() if f.is_file() and f.suffix == ".pt"]
        )

        for file_path in pt_files:
            try:
                # Load the tensor file to inspect its contents
                sequences_dict, _, _ = torch.load(file_path)
                if sequences_dict:
                    # All tensors in the dict have the same number of samples (batch size)
                    n_samples = sequences_dict[list(sequences_dict.keys())[0]].shape[0]

                    # Store the file's name (relative path) and its sample count
                    batch_files_metadata.append(
                        {"path": file_path.name, "samples": n_samples}
                    )
                    total_samples += n_samples
            except Exception as e:
                # Add a warning for robustness in case a file is corrupted
                print(f"[WARNING] Could not process file {file_path} for metadata: {e}")

        # Final metadata structure required by SequifierDatasetFromFolder
        metadata = {
            "total_samples": total_samples,
            "batch_files": batch_files_metadata,
        }

        # Write the metadata to a json file in the same folder
        metadata_path = directory / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)


@beartype
def _apply_column_statistics(
    data: pl.DataFrame,
    data_columns: list[str],
    id_maps: dict[str, dict[Union[str, int], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    n_classes: Optional[dict[str, int]] = None,
    col_types: Optional[dict[str, str]] = None,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, str]]:
    """Applies the column statistics to the data.

    Args:
        data: The data to apply the statistics to.
        data_columns: A list of data columns.
        id_maps: A dictionary containing the id maps for each categorical column.
        selected_columns_statistics: A dictionary containing the statistics for each numerical column.
        n_classes: A dictionary containing the number of classes for each categorical column.
        col_types: A dictionary containing the column types.

    Returns:
        A tuple containing the transformed data, the number of classes, and the column types.
    """
    if n_classes is None:
        n_classes = {col: len(id_maps[col]) + 1 for col in id_maps}

    if col_types is None:
        col_types = {col: str(data.schema[col]) for col in data_columns}

    for col in data_columns:
        if col in id_maps:
            data = data.with_columns(pl.col(col).replace(id_maps[col]))
        elif col in selected_columns_statistics:
            data = data.with_columns(
                (
                    (pl.col(col) - selected_columns_statistics[col]["mean"])
                    / (selected_columns_statistics[col]["std"] + 1e-9)
                ).alias(col)
            )

    return (data, n_classes, col_types)


@beartype
def _get_column_statistics(
    data: pl.DataFrame,
    data_columns: list[str],
    id_maps: dict[str, dict[Union[str, int], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    n_rows_running_count: int,
) -> tuple[
    dict[str, dict[Union[str, int], int]],
    dict[str, dict[str, float]],
]:
    """Gets the column statistics for the given data.

    Args:
        data: The data to get the statistics from.
        data_columns: A list of data columns.
        id_maps: A dictionary containing the id maps for each categorical column.
        selected_columns_statistics: A dictionary containing the statistics for each numerical column.
        n_rows_running_count: The running count of the number of rows.

    Returns:
        A tuple containing the id maps and the column statistics.
    """
    for data_col in data_columns:
        dtype = data.schema[data_col]
        if isinstance(dtype, (pl.String, pl.Utf8)) or isinstance(
            dtype, (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        ):
            new_id_map = create_id_map(data, column=data_col)
            id_maps[data_col] = combine_maps(new_id_map, id_maps.get(data_col, {}))
        elif isinstance(dtype, (pl.Float32, pl.Float64)):
            combined_mean, combined_std = get_combined_statistics(
                data.shape[0],
                data.get_column(data_col).mean(),
                data.get_column(data_col).std(),
                n_rows_running_count,
                selected_columns_statistics.get(data_col, {"mean": 0.0})["mean"],
                selected_columns_statistics.get(data_col, {"std": 0.0})["std"],
            )

            selected_columns_statistics[data_col] = {
                "std": combined_std,
                "mean": combined_mean,
            }
        else:
            raise ValueError(f"Column {data_col} has unsupported dtype: {dtype}")

    return id_maps, selected_columns_statistics


@beartype
def _load_and_preprocess_data(
    data_path: str,
    read_format: str,
    selected_columns: Optional[list[str]],
    max_rows: Optional[int],
) -> pl.DataFrame:
    """Loads and preprocesses the data.

    Args:
        data_path: The path to the data file.
        read_format: The file format to read.
        selected_columns: A list of columns to be included in the preprocessing.
        max_rows: The maximum number of rows to process.

    Returns:
        A polars DataFrame containing the preprocessed data.
    """
    print(f"[INFO] Reading data from '{data_path}'...")
    data = read_data(data_path, read_format, columns=selected_columns)
    assert (
        data.null_count().sum().sum_horizontal().item() == 0
    ), f"NaN or null values not accepted: {data.null_count()}"

    if selected_columns:
        selected_columns_filtered = [
            col for col in selected_columns if col not in ["sequenceId", "itemPosition"]
        ]
        data = data.select(["sequenceId", "itemPosition"] + selected_columns_filtered)

    if max_rows:
        data = data.slice(0, int(max_rows))

    return data


@beartype
def _process_batches_multiple_files_inner(
    project_path: str,
    data_name_root: str,
    process_id: int,
    file_paths: list[str],
    read_format: str,
    selected_columns: Optional[list[str]],
    max_rows: Optional[int],
    schema: Any,
    n_cores: int,
    seq_length: int,
    seq_step_sizes: list[int],
    data_columns: list[str],
    n_classes: dict[str, int],
    id_maps: dict[str, dict[Union[int, str], int]],
    selected_columns_statistics: dict[str, dict[str, float]],
    col_types: dict[str, str],
    group_proportions: list[float],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    combine_into_single_file: bool,
):
    """Inner function for processing batches of data from multiple files.

    Args:
        project_path: The path to the sequifier project directory.
        data_name_root: The root name of the data file.
        process_id: The id of the process.
        file_paths: A list of file paths to process.
        read_format: The file format to read.
        selected_columns: A list of columns to be included in the preprocessing.
        max_rows: The maximum number of rows to process.
        schema: The schema for the preprocessed data.
        n_cores: The number of cores to use for parallel processing.
        seq_length: The sequence length for the model inputs.
        seq_step_sizes: A list of step sizes for creating subsequences.
        data_columns: A list of data columns.
        n_classes: A dictionary containing the number of classes for each categorical column.
        id_maps: A dictionary containing the id maps for each categorical column.
        selected_columns_statistics: A dictionary containing the statistics for each numerical column.
        col_types: A dictionary containing the column types.
        group_proportions: A list of floats that define the relative sizes of data splits.
        write_format: The file format for the output files.
        split_paths: The paths to the output split files.
        target_dir: The target directory for temporary files.
        batches_per_file: The number of batches to process per file.
        combine_into_single_file: Whether to combine the output into a single file.
    """
    n_rows_running_count = 0
    for file_index, path in enumerate(file_paths):
        max_rows_inner = None if max_rows is None else max_rows - n_rows_running_count
        if max_rows_inner is None or max_rows_inner > 0:
            data = _load_and_preprocess_data(
                path, read_format, selected_columns, max_rows_inner
            )
            data, _, _ = _apply_column_statistics(
                data,
                data_columns,
                id_maps,
                selected_columns_statistics,
                n_classes,
                col_types,
            )

            adjusted_split_paths = [
                path.replace(
                    data_name_root, f"{data_name_root}-{process_id}-{file_index}"
                )
                for path in split_paths
            ]

            data_name_root_inner = f"{data_name_root}-{process_id}-{file_index}"

            n_batches = _process_batches_single_file(
                project_path,
                data_name_root_inner,
                data,
                schema,
                n_cores,
                seq_length,
                seq_step_sizes,
                data_columns,
                col_types,
                group_proportions,
                write_format,
                adjusted_split_paths,
                target_dir,
                batches_per_file,
            )

            if combine_into_single_file:
                input_files = create_file_paths_for_multiple_files1(
                    project_path,
                    target_dir,
                    len(group_proportions),
                    n_batches,
                    process_id,
                    file_index,
                    data_name_root,
                    write_format,
                )
                combine_multiprocessing_outputs(
                    project_path,
                    target_dir,
                    len(group_proportions),
                    input_files,
                    data_name_root,
                    write_format,
                    in_target_dir=True,
                    pre_split_str=f"{process_id}-{file_index}",
                )

                delete_files(input_files)

            n_rows_running_count += data.shape[0]


@beartype
def _process_batches_single_file(
    project_path: str,
    data_name_root: str,
    data: pl.DataFrame,
    schema: Any,
    n_cores: Optional[int],
    seq_length: int,
    seq_step_sizes: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    group_proportions: list[float],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
) -> int:
    """Processes batches of data from a single file.

    Args:
        project_path: The path to the sequifier project directory.
        data_name_root: The root name of the data file.
        data: The data to process.
        schema: The schema for the preprocessed data.
        n_cores: The number of cores to use for parallel processing.
        seq_length: The sequence length for the model inputs.
        seq_step_sizes: A list of step sizes for creating subsequences.
        data_columns: A list of data columns.
        col_types: A dictionary containing the column types.
        group_proportions: A list of floats that define the relative sizes of data splits.
        write_format: The file format for the output files.
        split_paths: The paths to the output split files.
        target_dir: The target directory for temporary files.
        batches_per_file: The number of batches to process per file.

    Returns:
        The number of batches processed.
    """
    n_cores = n_cores or multiprocessing.cpu_count()
    batch_limits = get_batch_limits(data, n_cores)
    batches = [
        (
            project_path,
            data_name_root,
            process_id,
            data.slice(start, end - start),
            schema,
            split_paths,
            seq_length,
            seq_step_sizes,
            data_columns,
            col_types,
            group_proportions,
            target_dir,
            write_format,
            batches_per_file,
        )
        for process_id, (start, end) in enumerate(batch_limits)
        if (end - start) > 0
    ]

    if len(batches) > 1:
        with multiprocessing.get_context("spawn").Pool(processes=len(batches)) as pool:
            pool.starmap(preprocess_batch, batches)
    else:
        preprocess_batch(*batches[0])

    return len(batches)


@beartype
def get_combined_statistics(
    n1: int, mean1: float, std1: float, n2: int, mean2: float, std2: float
) -> tuple[float, float]:
    """Calculates the combined standard deviation of two data subsets.

    Args:
      n1 (int): Number of samples in subset 1.
      mean1 (float): Mean of subset 1.
      std1 (float): Standard deviation of subset 1.
      n2 (int): Number of samples in subset 2.
      mean2 (float): Mean of subset 2.
      std2 (float): Standard deviation of subset 2.

    Returns:
      A tuple of floats containing the combined standard deviation of the two subsets.
    """
    # Step 1: Calculate the combined mean.
    combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)

    # Step 2: Calculate the pooled sum of squared differences.
    # This includes the internal variance of each subset and the variance
    # between the subset mean and the combined mean.
    sum_of_squares1 = (n1 - 1) * std1**2 + n1 * (mean1 - combined_mean) ** 2
    sum_of_squares2 = (n2 - 1) * std2**2 + n2 * (mean2 - combined_mean) ** 2

    # Step 3: Calculate the combined standard deviation.
    combined_std = math.sqrt((sum_of_squares1 + sum_of_squares2) / (n1 + n2 - 1))

    return combined_mean, combined_std


@beartype
def create_id_map(data: pl.DataFrame, column: str) -> dict[Union[str, int], int]:
    """Creates a map from unique values in a column to an integer index.

    Args:
        data: The DataFrame containing the column.
        column: The name of the column.

    Returns:
        A dictionary mapping unique values to an integer index.
    """
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )  # type: ignore
    id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
    return dict(id_map)


@beartype
def get_batch_limits(data: pl.DataFrame, n_batches: int) -> list[tuple[int, int]]:
    """Calculates the batch limits for a given DataFrame.

    Args:
        data: The DataFrame to split into batches.
        n_batches: The number of batches to create.

    Returns:
        A list of tuples, where each tuple contains the start and end index of a batch.
    """
    sequence_ids = data.get_column("sequenceId").to_numpy()
    new_sequence_id_indices = np.concatenate(
        [
            [0],
            np.where(
                np.concatenate([[False], sequence_ids[1:] != sequence_ids[:-1]], axis=0)
            )[0],
        ]
    )

    ideal_step = math.ceil(data.shape[0] / n_batches)
    ideal_limits = np.array(
        [ideal_step * m for m in range(n_batches)] + [data.shape[0]]
    )
    distances = [
        np.abs(new_sequence_id_indices - ideal_limit)
        for ideal_limit in ideal_limits[:-1]
    ]
    actual_limit_indices = [
        np.where(distance == np.min(distance))[0] for distance in distances
    ]
    actual_limits = [
        int(new_sequence_id_indices[limit_index[0]])
        for limit_index in actual_limit_indices
    ] + [data.shape[0]]
    return list(zip(actual_limits[:-1], actual_limits[1:]))


@beartype
def combine_maps(
    map1: dict[Union[str, int], int], map2: dict[Union[str, int], int]
) -> dict[Union[str, int], int]:
    """Combines two id maps.

    Args:
        map1: The first id map.
        map2: The second id map.

    Returns:
        The combined id map.
    """
    combined_keys = sorted(list(set(list(map1.keys())).union(list(set(map2.keys())))))
    id_map = {id_: i + 1 for i, id_ in enumerate(combined_keys)}
    return id_map


@beartype
def get_group_bounds(data_subset: pl.DataFrame, group_proportions: list[float]):
    """Calculates the group bounds for a given data subset.

    Args:
        data_subset: The data subset to calculate the group bounds for.
        group_proportions: A list of floats that define the relative sizes of data splits.

    Returns:
        A list of tuples, where each tuple contains the start and end index of a group.
    """
    n = data_subset.shape[0]
    upper_bounds = list((np.cumsum(group_proportions) * n).astype(int))
    lower_bounds = [0] + list(upper_bounds[:-1])
    group_bounds = list(zip(lower_bounds, upper_bounds))
    return group_bounds


@beartype
def process_and_write_data_pt(
    data: pl.DataFrame, seq_length: int, path: str, column_types: dict[str, str]
):
    """Processes and writes the data to a .pt file.

    Args:
        data: The data to process and write.
        seq_length: The sequence length.
        path: The path to write the file to.
        column_types: A dictionary containing the column types.
    """
    if data.is_empty():
        return

    sequence_cols = [str(c) for c in range(seq_length - 1, -1, -1)]

    all_feature_cols = data.get_column("inputCol").unique().to_list()

    aggs = [
        pl.concat_list(sequence_cols)
        .filter(pl.col("inputCol") == col_name)
        .flatten()
        .alias(f"seq_{col_name}")
        for col_name in all_feature_cols
    ]

    aggregated_data = (
        data.group_by(["sequenceId", "subsequenceId"])
        .agg(aggs)
        .sort(["sequenceId", "subsequenceId"])
    )

    if aggregated_data.is_empty():
        return

    sequence_ids_tensor = torch.tensor(
        aggregated_data.get_column("sequenceId").to_numpy(), dtype=torch.int64
    )

    sequences_dict = {}
    targets_dict = {}

    for col_name in all_feature_cols:
        torch_dtype = PANDAS_TO_TORCH_TYPES[column_types[col_name]]

        sequences_np = np.vstack(
            aggregated_data.get_column(f"seq_{col_name}").to_numpy(writable=True)
        )

        sequences_dict[col_name] = torch.tensor(sequences_np[:, :-1], dtype=torch_dtype)
        targets_dict[col_name] = torch.tensor(sequences_np[:, 1:], dtype=torch_dtype)

    if not sequences_dict:
        return

    print(f"[INFO] Writing preprocessed data to '{path}'...")
    data_to_save = (sequences_dict, targets_dict, sequence_ids_tensor)
    torch.save(data_to_save, path)


@beartype
def _write_accumulated_sequences(
    sequences_to_write: list[pl.DataFrame],
    split_path: str,
    write_format: str,
    process_id: int,
    file_index: int,
    target_dir: str,
    seq_length: int,
    col_types: dict[str, str],
):
    """Helper to write a batch of accumulated sequences to a single file."""
    if not sequences_to_write:
        return

    combined_df = pl.concat(sequences_to_write)

    # Construct a unique filename for the batched file
    split_path_batch_seq = split_path.replace(
        f".{write_format}", f"-{process_id}-{file_index}.{write_format}"
    )
    out_path = insert_top_folder(split_path_batch_seq, target_dir)

    # Write the combined data
    process_and_write_data_pt(combined_df, seq_length, out_path, col_types)


@beartype
def preprocess_batch(
    project_path: str,
    data_name_root: str,
    process_id: int,
    batch: pl.DataFrame,
    schema: Any,
    split_paths: list[str],
    seq_length: int,
    seq_step_sizes: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    group_proportions: list[float],
    target_dir: str,
    write_format: str,
    batches_per_file: int,
) -> None:
    """Processes a batch of data.

    Args:
        project_path: The path to the sequifier project directory.
        data_name_root: The root name of the data file.
        process_id: The id of the process.
        batch: The batch of data to process.
        schema: The schema for the preprocessed data.
        split_paths: The paths to the output split files.
        seq_length: The sequence length for the model inputs.
        seq_step_sizes: A list of step sizes for creating subsequences.
        data_columns: A list of data columns.
        col_types: A dictionary containing the column types.
        group_proportions: A list of floats that define the relative sizes of data splits.
        target_dir: The target directory for temporary files.
        write_format: The file format for the output files.
        batches_per_file: The number of batches to process per file.
    """
    sequence_ids = sorted(batch.get_column("sequenceId").unique().to_list())

    if write_format == "pt":
        # New logic for batching sequences into files for .pt format
        sequences_by_split = {i: [] for i in range(len(split_paths))}
        file_indices = {i: 0 for i in range(len(split_paths))}

        for i, sequence_id in enumerate(sequence_ids):
            data_subset = batch.filter(pl.col("sequenceId") == sequence_id)
            group_bounds = get_group_bounds(data_subset, group_proportions)
            sequences = {
                i: cast_columns_to_string(
                    extract_sequences(
                        data_subset.slice(lb, ub - lb),
                        schema,
                        seq_length,
                        seq_step_sizes[i],
                        data_columns,
                    )
                )
                for i, (lb, ub) in enumerate(group_bounds)
            }

            for group, split_df in sequences.items():
                if not split_df.is_empty():
                    sequences_by_split[group].append(split_df)

                # Check if the accumulator for this split has reached the desired size
                if len(sequences_by_split[group]) >= batches_per_file:
                    _write_accumulated_sequences(
                        sequences_by_split[group],
                        split_paths[group],
                        write_format,
                        process_id,
                        file_indices[group],
                        target_dir,
                        seq_length,
                        col_types,
                    )
                    # Reset the accumulator and increment the file index
                    sequences_by_split[group] = []
                    file_indices[group] += 1

        # After the loop, write any remaining sequences that didn't fill a full batch
        for group in range(len(split_paths)):
            _write_accumulated_sequences(
                sequences_by_split[group],
                split_paths[group],
                write_format,
                process_id,
                file_indices[group],
                target_dir,
                seq_length,
                col_types,
            )

    else:
        written_files: dict[int, list[str]] = {i: [] for i in range(len(split_paths))}
        for i, sequence_id in enumerate(sequence_ids):
            data_subset = batch.filter(pl.col("sequenceId") == sequence_id)
            group_bounds = get_group_bounds(data_subset, group_proportions)
            sequences = {
                j: cast_columns_to_string(
                    extract_sequences(
                        data_subset.slice(lb, ub - lb),
                        schema,
                        seq_length,
                        seq_step_sizes[j],
                        data_columns,
                    )
                )
                for j, (lb, ub) in enumerate(group_bounds)
            }
            post_split_str = f"{process_id}-{i}"

            for split_path, (group, split) in zip(split_paths, sequences.items()):
                split_path_batch_seq = split_path.replace(
                    f".{write_format}", f"-{post_split_str}.{write_format}"
                )
                split_path_batch_seq = insert_top_folder(
                    split_path_batch_seq, target_dir
                )

                if write_format == "csv":
                    write_data(split, split_path_batch_seq, "csv")
                elif write_format == "parquet":
                    write_data(split, split_path_batch_seq, "parquet")

                written_files[group].append(split_path_batch_seq)

        combine_multiprocessing_outputs(
            project_path,
            target_dir,
            len(split_paths),
            written_files,
            data_name_root,
            write_format,
            in_target_dir=True,
            post_split_str=f"{process_id}",
        )


@beartype
def extract_sequences(
    data: pl.DataFrame,
    schema: Any,
    seq_length: int,
    seq_step_size: int,
    columns: list[str],
) -> pl.DataFrame:
    """Extracts sequences from the data.

    Args:
        data: The data to extract sequences from.
        schema: The schema for the preprocessed data.
        seq_length: The sequence length for the model inputs.
        seq_step_size: The step size for creating subsequences.
        columns: A list of data columns.

    Returns:
        A polars DataFrame containing the extracted sequences.
    """
    if data.is_empty():
        return pl.DataFrame(schema=schema)

    raw_sequences = data.group_by("sequenceId", maintain_order=True).agg(
        [pl.col(c) for c in columns]
    )

    rows = []
    for in_row in raw_sequences.iter_rows(named=True):
        in_seq_lists_only = {col: in_row[col] for col in columns}

        subsequences = extract_subsequences(
            in_seq_lists_only, seq_length, seq_step_size, columns
        )

        for subsequence_id in range(len(subsequences[columns[0]])):
            for col, subseqs in subsequences.items():
                row = [in_row["sequenceId"], subsequence_id, col] + subseqs[
                    subsequence_id
                ]
                assert len(row) == (seq_length + 3), f"{row = }"
                rows.append(row)

    sequences = pl.DataFrame(
        rows,
        schema=schema,
        orient="row",
    )
    return sequences


@beartype
def get_subsequence_starts(
    in_seq_length: int, seq_length: int, seq_step_size: int
) -> np.ndarray:
    """Calculates the start indices for subsequences.

    Args:
        in_seq_length: The length of the input sequence.
        seq_length: The length of the subsequences.
        seq_step_size: The step size for creating subsequences.

    Returns:
        A numpy array containing the start indices of the subsequences.
    """
    nseq_adjusted = math.ceil((in_seq_length - seq_length) / seq_step_size)
    seq_step_size_adjusted = math.floor(
        (in_seq_length - seq_length) / max(1, nseq_adjusted)
    )
    increments = [0] + [max(1, seq_step_size_adjusted)] * nseq_adjusted
    while np.sum(increments) < (in_seq_length - seq_length):
        increments[np.argmin(increments[1:]) + 1] += 1

    return np.cumsum(increments)


@beartype
def extract_subsequences(
    in_seq: dict[str, list],
    seq_length: int,
    seq_step_size: int,
    columns: list[str],
) -> dict[str, list[list[Union[float, int]]]]:
    """Extracts subsequences from a sequence.

    Args:
        in_seq: The input sequence.
        seq_length: The length of the subsequences.
        seq_step_size: The step size for creating subsequences.
        columns: A list of data columns.

    Returns:
        A dictionary containing the extracted subsequences.
    """
    if not in_seq[columns[0]]:
        return {col: [] for col in columns}

    in_seq_len = len(in_seq[columns[0]])
    if in_seq_len < seq_length:
        pad_len = seq_length - in_seq_len
        in_seq = {col: ([0] * pad_len) + in_seq[col] for col in columns}
    in_seq_length = len(in_seq[columns[0]])

    subsequence_starts = get_subsequence_starts(
        in_seq_length, seq_length, seq_step_size
    )

    return {
        col: [list(in_seq[col][i : i + seq_length]) for i in subsequence_starts]
        for col in columns
    }


@beartype
def insert_top_folder(path: str, folder_name: str) -> str:
    """Inserts a folder into a path.

    Args:
        path: The path to insert the folder into.
        folder_name: The name of the folder to insert.

    Returns:
        The new path.
    """
    components = os.path.split(path)
    new_components = list(components[:-1]) + [folder_name] + [components[-1]]
    return os.path.join(*new_components)


@beartype
def cast_columns_to_string(data: pl.DataFrame) -> pl.DataFrame:
    """Casts the columns of a DataFrame to strings.

    Args:
        data: The DataFrame to cast the columns of.

    Returns:
        The DataFrame with the columns cast to strings.
    """
    data.columns = [str(col) for col in data.columns]
    return data


@beartype
def delete_files(files: Union[list[str], dict[int, list[str]]]) -> None:
    """Deletes a list of files.

    Args:
        files: A list of files to delete.
    """
    if isinstance(files, dict):
        files = [x for y in list(files.values()) for x in y]
    for file in files:
        os.remove(file)


@beartype
def create_file_paths_for_multiple_files1(
    project_path: str,
    target_dir: str,
    n_splits: int,
    n_batches: int,
    process_id: int,
    file_index: int,
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Creates file paths for multiple files.

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The target directory for temporary files.
        n_splits: The number of splits.
        n_batches: The number of batches.
        process_id: The id of the process.
        file_index: The index of the file.
        dataset_name: The name of the dataset.
        write_format: The file format for the output files.

    Returns:
        A dictionary containing the file paths for each split.
    """
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_path,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{file_index}-split{split}-{batch_id}.{write_format}",
            )
            for batch_id in range(n_batches)
        ]
        files[split] = files_for_split
    return files


@beartype
def create_file_paths_for_single_file(
    project_path: str,
    target_dir: str,
    n_splits: int,
    n_batches: int,
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Creates file paths for a single file.

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The target directory for temporary files.
        n_splits: The number of splits.
        n_batches: The number of batches.
        dataset_name: The name of the dataset.
        write_format: The file format for the output files.

    Returns:
        A dictionary containing the file paths for each split.
    """
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_path,
                "data",
                target_dir,
                f"{dataset_name}-split{split}-{core_id}.{write_format}",
            )
            for core_id in range(n_batches)
        ]
        files[split] = files_for_split
    return files


@beartype
def create_file_paths_for_multiple_files2(
    project_path: str,
    target_dir: str,
    n_splits: int,
    n_processes: int,
    n_files: dict[int, int],
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Creates file paths for multiple files.

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The target directory for temporary files.
        n_splits: The number of splits.
        n_processes: The number of processes.
        n_files: A dictionary containing the number of files for each process.
        dataset_name: The name of the dataset.
        write_format: The file format for the output files.

    Returns:
        A dictionary containing the file paths for each split.
    """
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_path,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{file_index}-split{split}.{write_format}",
            )
            for process_id in range(n_processes)
            for file_index in range(n_files[process_id])
        ]
        files[split] = files_for_split

    return files


@beartype
def combine_multiprocessing_outputs(
    project_path: str,
    target_dir: str,
    n_splits: int,
    input_files: dict[int, list[str]],
    dataset_name: str,
    write_format: str,
    in_target_dir: bool = False,
    pre_split_str: Optional[str] = None,
    post_split_str: Optional[str] = None,
) -> None:
    """Combines the outputs of multiple processes.

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The target directory for temporary files.
        n_splits: The number of splits.
        input_files: A dictionary containing the input files for each split.
        dataset_name: The name of the dataset.
        write_format: The file format for the output files.
        in_target_dir: Whether the output files are in the target directory.
        pre_split_str: A string to prepend to the split number.
        post_split_str: A string to append to the split number.
    """
    for split in range(n_splits):
        if pre_split_str is None and post_split_str is None:
            file_name = f"{dataset_name}-split{split}.{write_format}"
        elif pre_split_str is not None and post_split_str is None:
            file_name = f"{dataset_name}-{pre_split_str}-split{split}.{write_format}"
        elif post_split_str is not None and pre_split_str is None:
            file_name = f"{dataset_name}-split{split}-{post_split_str}.{write_format}"
        else:
            file_name = f"{dataset_name}-{pre_split_str}-split{split}-{post_split_str}.{write_format}"

        out_path = os.path.join(project_path, "data", file_name)
        if in_target_dir:
            out_path = insert_top_folder(out_path, target_dir)

        print(f"[INFO] writing to: {out_path}")
        if write_format == "csv":
            command = " ".join(["csvstack"] + input_files[split] + [f"> {out_path}"])
            result = os.system(command)
            assert result == 0, f"command '{command}' failes: {result = }"
        elif write_format == "parquet":
            combine_parquet_files(input_files[split], out_path)


@beartype
def combine_parquet_files(files: list[str], out_path: str) -> None:
    """Combines multiple parquet files into a single file.

    Args:
        files: A list of parquet files to combine.
        out_path: The path to the output file.
    """
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))
