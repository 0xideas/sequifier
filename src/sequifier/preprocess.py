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
    """Runs the main data preprocessing pipeline.

    This function loads the preprocessing configuration, initializes the
    `Preprocessor` class, and executes the preprocessing steps based on the
    loaded configuration.

    Args:
        args: An object containing command-line arguments. Expected to have
            a `config_path` attribute specifying the path to the YAML
            configuration file.
        args_config: A dictionary containing additional configuration parameters
            that may override or supplement the settings loaded from the
            config file.
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
        continue_preprocessing: bool,
        data_path: str,
        read_format: str,
        write_format: str,
        combine_into_single_file: bool,
        selected_columns: Optional[list[str]],
        group_proportions: list[float],
        seq_length: int,
        stride_by_split: list[int],
        max_rows: Optional[int],
        seed: int,
        n_cores: Optional[int],
        batches_per_file: int,
        process_by_file: bool,
        subsequence_start_mode: str,
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
            stride_by_split: A list of step sizes for creating subsequences.
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
        self.continue_preprocessing = continue_preprocessing
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

            schema = self._create_schema(col_types, seq_length + 1)

            data = data.sort(["sequenceId", "itemPosition"])
            n_batches = _process_batches_single_file(
                self.project_path,
                self.data_name_root,
                data,
                schema,
                self.n_cores,
                seq_length,
                stride_by_split,
                data_columns,
                col_types,
                group_proportions,
                write_format,
                self.split_paths,
                self.target_dir,
                self.batches_per_file,
                subsequence_start_mode,
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
            (
                files_to_process,
                n_classes,
                id_maps,
                selected_columns_statistics,
                col_types,
                data_columns,
            ) = self._get_column_metadata_across_files(
                data_path, read_format, max_rows, selected_columns
            )
            self._export_metadata(
                id_maps, n_classes, col_types, selected_columns_statistics
            )
            schema = self._create_schema(col_types, seq_length + 1)

            self._process_batches_multiple_files(
                files_to_process,
                read_format,
                selected_columns,
                max_rows,
                schema,
                self.n_cores,
                seq_length,
                stride_by_split,
                data_columns,
                n_classes,
                id_maps,
                selected_columns_statistics,
                col_types,
                group_proportions,
                write_format,
                process_by_file,
                subsequence_start_mode,
            )

        self._cleanup(write_format)

    @beartype
    def _create_schema(
        self, col_types: dict[str, str], seq_length: int
    ) -> dict[str, Any]:
        """Creates the Polars schema for the intermediate sequence DataFrame.

        This schema defines the structure of the DataFrame after sequence
        extraction, which includes sequence identifiers, the start item position
        within the original sequence, the input column name, and columns for
        each item in the sequence (named '0', '1', ..., 'seq_length-1').

        Args:
            col_types: A dictionary mapping data column names to their Polars
                string representations (e.g., "Int64", "Float64").
            seq_length: The length of the sequences being extracted.

        Returns:
            A dictionary defining the Polars schema. Keys are column names
            (e.g., "sequenceId", "subsequenceId", "startItemPosition", "inputCol", "0", "1", ...)
            and values are Polars data types (e.g., `pl.Int64`).
        """
        schema = {
            "sequenceId": pl.Int64,
            "subsequenceId": pl.Int64,
            "startItemPosition": pl.Int64,
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
        list[str],
        dict[str, int],
        dict[str, dict[Union[str, int], int]],
        dict[str, dict[str, float]],
        dict[str, str],
        list[str],
    ]:
        """Scans multiple data files to compute combined column metadata.

        This method iterates through all files in `data_path` matching `read_format`,
        loading each one to incrementally build up metadata. It computes:
        1.  ID maps for categorical/string columns.
        2.  Mean and standard deviation for numerical (float) columns.
        3.  The total number of unique classes for mapped columns.
        4.  The data types of all columns.

        This is used when the dataset is split into multiple files to get a
        consistent global view of the data.

        Args:
            data_path: The path to the root data directory.
            read_format: The file extension (e.g., "csv", "parquet") to read.
            max_rows: The maximum total number of rows to process across all
                files. If `None`, all rows are processed.
            selected_columns: A list of columns to include. If `None`, all
                columns (except "sequenceId" and "itemPosition") are used.

        Returns:
            A tuple containing:
                - n_classes (dict[str, int]): Map of column name to its
                  number of unique classes (including padding/unknown).
                - id_maps (dict[str, dict[Union[str, int], int]]): Nested map
                  from column name to its value-to-integer-ID map.
                - selected_columns_statistics (dict[str, dict[str, float]]):
                  Nested map from numerical column name to its 'mean' and 'std'.
                - col_types (dict[str, str]): Map of column name to its
                  Polars string data type.
                - data_columns (list[str]): List of all processed data
                  column names.
        """
        n_rows_running_count = 0
        id_maps, selected_columns_statistics = {}, {}
        col_types, data_columns = None, None
        files_to_process = []
        print(f"[INFO] Data path: {data_path}")
        for root, dirs, files in os.walk(data_path):
            print(f"[INFO] N Files : {len(files)}")
            for file in sorted(list(files)):
                if file.endswith(read_format) and (
                    max_rows is None or n_rows_running_count < max_rows
                ):
                    print(f"[INFO] Preprocessing: reading {file}")
                    files_to_process.append(os.path.join(root, file))
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
        files_to_process = sorted(files_to_process)
        return (
            files_to_process,
            n_classes,
            id_maps,
            selected_columns_statistics,
            col_types,
            data_columns,
        )

    @beartype
    def _setup_directories(self) -> None:
        """Sets up the output directories for preprocessed data.

        This method creates the base `data/` directory within the `project_path`
        if it doesn't exist. It also creates a temporary directory (defined by
        `self.target_dir`) for storing intermediate batch files, removing it
        first if it already exists to ensure a clean run.
        """

        temp_path = os.path.join(self.project_path, "data", self.target_dir)

        if self.continue_preprocessing:
            if not os.path.exists(temp_path):
                raise Exception(f"temp folder at '{temp_path}' does not exist")
        else:
            os.makedirs(os.path.join(self.project_path, "data"), exist_ok=True)
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            os.makedirs(temp_path)

    @beartype
    def _setup_split_paths(self, write_format: str, n_splits: int) -> None:
        """Sets up the final output paths for the data splits.

        This method constructs the full file paths for each data split
        (e.g., train, validation, test) based on the `data_name_root` and
        `write_format`. The paths are stored in the `self.split_paths` attribute.

        Args:
            write_format: The file extension for the output files (e.g., "pt", "parquet").
            n_splits: The number of splits to create (e.g., 3 for train/val/test).
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
        stride_by_split: list[int],
        data_columns: list[str],
        n_classes: dict[str, int],
        id_maps: dict[str, dict[Union[int, str], int]],
        selected_columns_statistics: dict[str, dict[str, float]],
        col_types: dict[str, str],
        group_proportions: list[float],
        write_format: str,
        process_by_file: bool = True,
        subsequence_start_mode: str = "distribute",
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
            stride_by_split: A list of step sizes for creating subsequences.
            data_columns: A list of data columns.
            n_classes: A dictionary containing the number of classes for each categorical column.
            id_maps: A dictionary containing the id maps for each categorical column.
            selected_columns_statistics: A dictionary containing the statistics for each numerical column.
            col_types: A dictionary containing the column types.
            group_proportions: A list of floats that define the relative sizes of data splits.
            write_format: The file format for the output files.
            process_by_file: A flag to indicate if processing should be done file by file.
            subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".
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
                stride_by_split=stride_by_split,
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
                continue_preprocessing=self.continue_preprocessing,
                subsequence_start_mode=subsequence_start_mode,
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
                "stride_by_split": stride_by_split,
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
                "continue_preprocessing": self.continue_preprocessing,
                "subsequence_start_mode": subsequence_start_mode,
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
        """Finalizes output files and removes temporary directories.

        If `write_format` is 'pt' and `combine_into_single_file` is False,
        this method moves the processed .pt batch files from the temporary
        `target_dir` into their final split-specific subfolders (e.g.,
        'data_name_root-split0/'). It also generates a 'metadata.json' file
        in each of these subfolders, which is required by
        `SequifierDatasetFromFolder`.

        Finally, it removes the temporary `target_dir` if it's empty or
        if `target_dir` is "temp" (implying `combine_into_single_file` was True).

        Args:
            write_format: The file format of the output files (e.g., "pt").
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
        """Exports the computed data metadata to a JSON file.

        Saves metadata such as class counts, ID mappings, split paths,
        column types, and numerical statistics to a JSON file. This file is
        saved in the `configs/ddconfigs/` directory, named after the
        `data_name_root`. This metadata is essential for initializing the
        model and data loaders during training.

        Args:
            id_maps: A dictionary containing the id maps for each
                categorical column.
            n_classes: A dictionary containing the number of classes for
                each categorical column.
            col_types: A dictionary containing the column types.
            selected_columns_statistics: A dictionary containing the
                statistics for each numerical column.
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
        """Scans a directory for .pt files, counts samples, and writes metadata.json.

                This method is used when `write_format` is 'pt' and
                `combine_into_single_file` is False. It iterates over all .pt files
                in the given `folder_path`, loads each one to count the number of
                samples (sequences), and writes a `metadata.json` file in that
                same folder. This JSON file contains the total sample count and a
                list of all batch files with their respective sample counts, which
        s        is required by the `SequifierDatasetFromFolder` data loader.

                Args:
                    folder_path: The path to the directory containing the .pt batch files
                        for a specific data split.
        """
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
                sequences_dict, _, _, _, _ = torch.load(file_path)
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
    """Applies pre-computed statistics to transform the data.

    This function performs two main transformations on the DataFrame:
    1.  **Categorical Mapping**: Replaces original values in categorical
        columns with their corresponding integer IDs using `id_maps`.
    2.  **Numerical Standardization**: Standardizes numerical columns
        (those in `selected_columns_statistics`) using the Z-score
        formula: (value - mean) / (std + 1e-9).

    It also computes `n_classes` and `col_types` if they are not provided.

    Args:
        data: The input Polars DataFrame to transform.
        data_columns: A list of column names to process.
        id_maps: A nested dictionary mapping column names to their
            value-to-integer-ID maps.
        selected_columns_statistics: A nested dictionary mapping
            column names to their 'mean' and 'std' statistics.
        n_classes: An optional dictionary mapping column names to
            their total number of unique classes. If `None`, it's computed
            from `id_maps`.
        col_types: An optional dictionary mapping column names to
            their string data types. If `None`, it's inferred from the
            `data` schema.

    Returns:
        A tuple `(data, n_classes, col_types)` where:
            - `data`: The transformed Polars DataFrame.
            - `n_classes`: The (potentially computed) class count dictionary.
            - `col_types`: The (potentially computed) column type dictionary.
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
    """Computes or updates column statistics from a data chunk.

    This function iterates over the `data_columns` and updates the
    `id_maps` and `selected_columns_statistics` dictionaries based on the
    provided `data` chunk.

    - For string/integer columns: It creates an ID map of unique values and
      merges it with any existing map in `id_maps`.
    - For float columns: It computes the mean and standard deviation of the
      chunk and combines them with the existing statistics in
      `selected_columns_statistics` using a numerically stable parallel
      algorithm, weighted by `n_rows_running_count`.

    Args:
        data: The Polars DataFrame chunk to process.
        data_columns: A list of column names in `data` to analyze.
        id_maps: The dictionary of existing ID maps to be updated.
        selected_columns_statistics: The dictionary of existing numerical
            statistics to be updated.
        n_rows_running_count: The total number of rows processed *before*
            this chunk, used for weighting statistics.

    Returns:
        A tuple `(id_maps, selected_columns_statistics)` containing the
        updated dictionaries.

    Raises:
        ValueError: If a column has an unsupported data type (neither
            string, integer, nor float).
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
    """Loads data from a file and performs initial preparation.

    This function reads a data file using the specified `read_format`.
    It then performs the following steps:
    1.  Asserts that the data contains no null or NaN values.
    2.  Selects only the `selected_columns` if provided.
    3.  Slices the DataFrame to `max_rows` if provided.

    Args:
        data_path: The path to the data file.
        read_format: The file format to read (e.g., "csv", "parquet").
        selected_columns: A list of columns to load. If `None`, all
            columns are loaded.
        max_rows: The maximum number of rows to load. If `None`, all
            rows are loaded.

    Returns:
        A Polars DataFrame containing the loaded and initially
        prepared data.
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


def _check_file_has_been_processed(
    project_path: str,
    data_name_root: str,
    process_id: int,
    group_proportions: list[float],
    write_format: str,
    target_dir: str,
    combine_into_single_file: bool,
    file_index_str: str,
):
    file_prefix_str = f"{data_name_root}-{process_id}-{file_index_str}"

    if combine_into_single_file:
        # Case 1: Combining into a single file. Check for the intermediate
        # combined file in the target_dir.
        expected_file_path = ""
        for split_index in range(len(group_proportions)):
            expected_file_path = create_split_file_path(
                project_path,
                data_name_root,
                split_index,
                write_format,
                in_target_dir=True,  # Intermediate files are in target_dir
                target_dir=target_dir,
                pre_split_str=file_prefix_str,  # This file's unique ID
                post_split_str=None,
            )
            if not os.path.exists(expected_file_path):
                # If any split's intermediate file is missing, we must re-process
                return False
        print(
            f"[INFO] Files: {expected_file_path.split('split')[0] + 'splitX'} found, skipping"
        )
        return True
    else:
        temp_dir_path = os.path.join(project_path, "data", target_dir)

        if not os.path.isdir(temp_dir_path):
            return False

        for file_name in os.listdir(temp_dir_path):
            if file_name.startswith(file_prefix_str) and file_name.endswith(
                f".{write_format}"
            ):
                print(f"[INFO] Found {file_name}, skipping corresponding input file...")
                return True

        return False


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
    stride_by_split: list[int],
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
    continue_preprocessing: bool,
    subsequence_start_mode: str,
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
        stride_by_split: A list of step sizes for creating subsequences.
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
        continue_preprocessing: Continue preprocessing job that was interrupted while writing to temp folder.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".
    """
    n_files = len(file_paths)
    assert n_files > 0
    pad_width = len(str(n_files - 1))
    n_rows_running_count = 0
    for file_index, path in enumerate(file_paths):
        max_rows_inner = None if max_rows is None else max_rows - n_rows_running_count
        if max_rows_inner is None or max_rows_inner > 0:
            file_index_str = str(file_index).zfill(pad_width)

            adjusted_split_paths = [
                path.replace(
                    data_name_root, f"{data_name_root}-{process_id}-{file_index_str}"
                )
                for path in split_paths
            ]
            if continue_preprocessing:
                file_has_been_processed = _check_file_has_been_processed(
                    project_path,
                    data_name_root,
                    process_id,
                    group_proportions,
                    write_format,
                    target_dir,
                    combine_into_single_file,
                    file_index_str,
                )

                if file_has_been_processed:
                    print(f"[INFO] Skipping already processed file: {path}")
                    if max_rows is not None:
                        data = _load_and_preprocess_data(
                            path, read_format, selected_columns, max_rows_inner
                        )
                        n_rows_running_count += data.shape[0]
                    continue

            data = _load_and_preprocess_data(
                path, read_format, selected_columns, max_rows_inner
            )
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

            data_name_root_inner = f"{data_name_root}-{process_id}-{file_index_str}"

            n_batches = _process_batches_single_file(
                project_path,
                data_name_root_inner,
                data,
                schema,
                n_cores,
                seq_length,
                stride_by_split,
                data_columns,
                col_types,
                group_proportions,
                write_format,
                adjusted_split_paths,
                target_dir,
                batches_per_file,
                subsequence_start_mode,
            )

            if combine_into_single_file:
                input_files = create_file_paths_for_multiple_files1(
                    project_path,
                    target_dir,
                    len(group_proportions),
                    n_batches,
                    process_id,
                    file_index_str,
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
                    pre_split_str=f"{process_id}-{file_index_str}",
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
    stride_by_split: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    group_proportions: list[float],
    write_format: str,
    split_paths: list[str],
    target_dir: str,
    batches_per_file: int,
    subsequence_start_mode: str,
) -> int:
    """Processes batches of data from a single file.

    Args:
        project_path: The path to the sequifier project directory.
        data_name_root: The root name of the data file.
        data: The data to process.
        schema: The schema for the preprocessed data.
        n_cores: The number of cores to use for parallel processing.
        seq_length: The sequence length for the model inputs.
        stride_by_split: A list of step sizes for creating subsequences.
        data_columns: A list of data columns.
        col_types: A dictionary containing the column types.
        group_proportions: A list of floats that define the relative sizes of data splits.
        write_format: The file format for the output files.
        split_paths: The paths to the output split files.
        target_dir: The target directory for temporary files.
        batches_per_file: The number of batches to process per file.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".

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
            stride_by_split,
            data_columns,
            col_types,
            group_proportions,
            target_dir,
            write_format,
            batches_per_file,
            subsequence_start_mode,
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
    """Calculates the combined mean and standard deviation of two data subsets.

    Uses a stable parallel algorithm (related to Welford's algorithm) to
    combine statistics from two subsets without needing the original data.

    Args:
        n1: Number of samples in subset 1.
        mean1: Mean of subset 1.
        std1: Standard deviation of subset 1.
        n2: Number of samples in subset 2.
        mean2: Mean of subset 2.
        std2: Standard deviation of subset 2.

    Returns:
        A tuple `(combined_mean, combined_std)` containing the combined
        mean and standard deviation of the two subsets.
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
    """Creates a map from unique values in a column to integer indices.

    Finds all unique values in the specified `column` of the `data`
    DataFrame, sorts them, and creates a dictionary mapping each unique
    value to a 1-based integer index.

    Args:
        data: The Polars DataFrame containing the column.
        column: The name of the column to map.

    Returns:
        A dictionary mapping unique values (str or int) to an integer
        index (starting from 1).
    """
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )  # type: ignore
    id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
    return dict(id_map)


@beartype
def get_batch_limits(data: pl.DataFrame, n_batches: int) -> list[tuple[int, int]]:
    """Calculates row indices to split a DataFrame into batches.

    This function divides the DataFrame into `n_batches` roughly equal
    chunks. Crucially, it ensures that no `sequenceId` is split across
    two different batches. It does this by finding the ideal split points
    and then adjusting them to the nearest `sequenceId` boundary.

    Args:
        data: The DataFrame to split. Must be sorted by "sequenceId".
        n_batches: The desired number of batches.

    Returns:
        A list of `(start_index, end_index)` tuples, where each tuple
        defines the row indices for a batch.
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
    """Combines two ID maps into a new, consolidated map.

    Takes all unique keys from both `map1` and `map2`, sorts them,
    and creates a new, single map where keys are mapped to 1-based
    indices based on the sorted order. This ensures a consistent
    mapping across different data chunks.

    Args:
        map1: The first ID map.
        map2: The second ID map.

    Returns:
        A new, combined, and re-indexed ID map.
    """
    combined_keys = sorted(list(set(list(map1.keys())).union(list(set(map2.keys())))))
    id_map = {id_: i + 1 for i, id_ in enumerate(combined_keys)}
    return id_map


@beartype
def get_group_bounds(data_subset: pl.DataFrame, group_proportions: list[float]):
    """Calculates row indices for splitting a sequence into groups.

    This function takes a DataFrame `data_subset` (which typically
    contains all items for a single `sequenceId`) and calculates the
    row indices to split it into multiple groups (e.g., train, val, test)
    based on the provided `group_proportions`.

    Args:
        data_subset: The DataFrame (for a single sequence) to split.
        group_proportions: A list of floats (e.g., [0.8, 0.1, 0.1]) that
            sum to 1.0, defining the relative sizes of the splits.

    Returns:
        A list of `(start_index, end_index)` tuples, one for each
        proportion, defining the row slices for each group.
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
    """Processes the sequence DataFrame and writes it to a .pt file.

    This function takes the long-format sequence DataFrame (`data`),
    aggregates it by `sequenceId` and `subsequenceId`, and pivots it
    so that each `inputCol` becomes its own column containing a list
    of sequence items. It also extracts the `startItemPosition`.

    It then converts these lists into NumPy arrays, splits them into
    `sequences` (all but last item) and `targets` (all but first item),
    and converts them to PyTorch tensors along with sequence/subsequence IDs
    and start positions. The final data tuple
    `(sequences_dict, targets_dict, sequence_ids_tensor, subsequence_ids_tensor, start_item_positions_tensor)`
    is saved to a .pt file using `torch.save`.

    Args:
        data: The long-format Polars DataFrame of extracted sequences.
        seq_length: The total sequence length (N). The resulting tensors
            will have sequence length N-1.
        path: The output file path (e.g., "data/batch_0.pt").
        column_types: A dictionary mapping column names to their
            string data types, used to determine the correct torch dtype.
    """
    if data.is_empty():
        return

    sequence_cols = [str(c) for c in range(seq_length, -1, -1)]

    all_feature_cols = data.get_column("inputCol").unique().to_list()

    aggs = [
        pl.concat_list(sequence_cols)
        .filter(pl.col("inputCol") == col_name)
        .flatten()
        .alias(f"seq_{col_name}")
        for col_name in all_feature_cols
    ] + [pl.col("startItemPosition").first().alias("startItemPosition")]

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
    subsequence_ids_tensor = torch.tensor(
        aggregated_data.get_column("subsequenceId").to_numpy(), dtype=torch.int64
    )
    start_item_positions_tensor = torch.tensor(
        aggregated_data.get_column("startItemPosition").to_numpy(), dtype=torch.int64
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
    data_to_save = (
        sequences_dict,
        targets_dict,
        sequence_ids_tensor,
        subsequence_ids_tensor,
        start_item_positions_tensor,
    )
    torch.save(data_to_save, path)


@beartype
def _write_accumulated_sequences(
    sequences_to_write: list[pl.DataFrame],
    split_path: str,
    write_format: str,
    process_id: int,
    file_index_str: str,
    target_dir: str,
    seq_length: int,
    col_types: dict[str, str],
):
    """Helper to write a batch of accumulated sequences to a single .pt file.

    This function concatenates a list of sequence DataFrames and writes
    the combined DataFrame to a single .pt file using
    `process_and_write_data_pt`. This is used to batch multiple sequences
    into fewer, larger files when `write_format` is 'pt'.

    Args:
        sequences_to_write: A list of Polars DataFrames to combine and write.
        split_path: The base path for the split (e.g., "data/split0.pt").
        write_format: The file format (e.g., "pt").
        process_id: The ID of the parent multiprocessing process.
        file_index: The index of this file batch for this split and process.
        target_dir: The temporary directory to write the file into.
        seq_length: The total sequence length.
        col_types: A dictionary mapping column names to their string types.
    """
    if not sequences_to_write:
        return

    combined_df = pl.concat(sequences_to_write)

    # Construct a unique filename for the batched file
    split_path_batch_seq = split_path.replace(
        f".{write_format}", f"-{process_id}-{file_index_str}.{write_format}"
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
    stride_by_split: list[int],
    data_columns: list[str],
    col_types: dict[str, str],
    group_proportions: list[float],
    target_dir: str,
    write_format: str,
    batches_per_file: int,
    subsequence_start_mode: str,
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
        stride_by_split: A list of step sizes for creating subsequences.
        data_columns: A list of data columns.
        col_types: A dictionary containing the column types.
        group_proportions: A list of floats that define the relative sizes of data splits.
        target_dir: The target directory for temporary files.
        write_format: The file format for the output files.
        batches_per_file: The number of batches to process per file.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".
    """
    sequence_ids = sorted(batch.get_column("sequenceId").unique().to_list())

    if write_format == "pt":
        # New logic for batching sequences into files for .pt format
        sequences_by_split = {i: [] for i in range(len(split_paths))}
        file_indices = {i: 0 for i in range(len(split_paths))}

        pad_width = len(str(math.ceil(len(sequence_ids) / batches_per_file) + 1))
        for i, sequence_id in enumerate(sequence_ids):
            data_subset = batch.filter(pl.col("sequenceId") == sequence_id)
            group_bounds = get_group_bounds(data_subset, group_proportions)
            sequences = {
                i: cast_columns_to_string(
                    extract_sequences(
                        data_subset.slice(lb, ub - lb),
                        schema,
                        seq_length,
                        stride_by_split[i],
                        data_columns,
                        subsequence_start_mode,
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
                        str(file_indices[group]).zfill(pad_width),
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
                str(file_indices[group]).zfill(pad_width),
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
                        stride_by_split[j],
                        data_columns,
                        subsequence_start_mode,
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
    stride_for_split: int,
    columns: list[str],
    subsequence_start_mode: str,
) -> pl.DataFrame:
    """Extracts subsequences from a DataFrame of full sequences.

    This function takes a DataFrame where each row contains all items
    for a single `sequenceId`. It iterates through each `sequenceId`,
    extracts all possible subsequences of `seq_length` using the
    specified `stride_for_split`, calculates the starting position of each
    subsequence within the original sequence, and formats them into a new,
    long-format DataFrame that conforms to the provided `schema`.

    Args:
        data: The input Polars DataFrame, grouped by "sequenceId".
        schema: The schema for the output long-format DataFrame.
        seq_length: The length of the subsequences to extract.
        stride_for_split: The step size to use when sliding the window
            to create subsequences.
        columns: A list of the data column names (features) to extract.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".

    Returns:
        A new, long-format Polars DataFrame containing the extracted
        subsequences, matching the provided `schema`. Includes columns
        for `sequenceId`, `subsequenceId`, `startItemPosition`, `inputCol`,
        and the sequence items ('0', '1', ...).
    """
    if data.is_empty():
        return pl.DataFrame(schema=schema)

    print(f"[INFO] {data.shape = }")

    raw_sequences = data.group_by("sequenceId", maintain_order=True).agg(
        [pl.col(c) for c in columns]
    )

    rows = []
    for in_row in raw_sequences.iter_rows(named=True):
        in_seq_lists_only = {col: in_row[col] for col in columns}

        subsequences = extract_subsequences(
            in_seq_lists_only,
            seq_length,
            stride_for_split,
            columns,
            subsequence_start_mode,
        )

        for subsequence_id in range(len(subsequences[columns[0]])):
            for col, subseqs in subsequences.items():
                row = [
                    in_row["sequenceId"],
                    subsequence_id,
                    subsequence_id * stride_for_split,
                    col,
                ] + subseqs[subsequence_id]
                assert len(row) == (seq_length + 5), f"{row = }"
                rows.append(row)
    print(f"[INFO] {len(rows) = }")

    sequences = pl.DataFrame(
        rows,
        schema=schema,
        orient="row",
    )
    return sequences


@beartype
def get_subsequence_starts(
    in_seq_length: int,
    seq_length: int,
    stride_for_split: int,
    subsequence_start_mode: str,
) -> np.ndarray:
    """Calculates the start indices for extracting subsequences.

    This function determines the starting indices for sliding a window of
    `seq_length` over an input sequence of `in_seq_length`. It aims to
    use `stride_for_split`, but adjusts the step size slightly to ensure
    that the windows are distributed as evenly as possible and cover the
    full sequence from the beginning to the end.

    Args:
        in_seq_length: The length of the original input sequence.
        seq_length: The length of the subsequences to extract.
        stride_for_split: The *desired* step size between subsequences.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".

    Returns:
        A numpy array of integer start indices for each subsequence.
    """
    assert subsequence_start_mode in [
        "distribute",
        "exact",
    ], f"{subsequence_start_mode = } not in ['distribute', 'exact']"
    if subsequence_start_mode == "distribute":
        last_available_start = in_seq_length - (seq_length + 1)
        starts = np.arange(0, last_available_start + stride_for_split, stride_for_split)
        starts[-1] = last_available_start
        if len(starts) > 2:
            while True:
                starts_delta = starts[1:] - starts[:-1]
                if np.max(starts_delta) - np.min(starts_delta) > 1:
                    starts[np.argmin(starts_delta) + 1] -= 1
                else:
                    return starts
        return starts

    if subsequence_start_mode == "exact":
        assert (
            ((in_seq_length - 1) - seq_length) % stride_for_split == 0
        ), f"'exact' can only be used if: ((in_seq_length - 1) - seq_length) % stride_for_split == 0, {(in_seq_length -1) = }, {seq_length = }, {stride_for_split = }"
        last_possible_start = (
            in_seq_length - (seq_length - 1) - 1
        )  # the latter '-1' is to translate to index
        return np.arange(
            0, last_possible_start + 1, stride_for_split
        )  # the '+1' is to make it inclusive
    return np.array([])


@beartype
def extract_subsequences(
    in_seq: dict[str, list],
    seq_length: int,
    stride_for_split: int,
    columns: list[str],
    subsequence_start_mode: str,
) -> dict[str, list[list[Union[float, int]]]]:
    """Extracts subsequences from a dictionary of sequence lists.

    This function takes a dictionary `in_seq` where keys are column
    names and values are lists of items for a single full sequence.
    It first pads the sequences with 0s at the beginning if they are
    shorter than `seq_length`. Then, it calculates the subsequence
    start indices using `get_subsequence_starts` and extracts all
    subsequences.

    Args:
        in_seq: A dictionary mapping column names to lists of items
            (e.g., `{'col_A': [1, 2, 3, 4, 5], 'col_B': [6, 7, 8, 9, 10]}`).
        seq_length: The length of the subsequences to extract.
        stride_for_split: The desired step size between subsequences.
        columns: A list of the column names (keys in `in_seq`) to process.
        subsequence_start_mode: "distribute" to minimize max subsequence overlap, or "exact".

    Returns:
        A dictionary mapping column names to a *list of lists*, where
        each inner list is a subsequence.
    """
    if not in_seq[columns[0]]:
        return {col: [] for col in columns}

    in_seq_len = len(in_seq[columns[0]])
    if in_seq_len < (seq_length + 1):
        pad_len = (seq_length + 1) - in_seq_len
        in_seq = {col: ([0] * pad_len) + in_seq[col] for col in columns}
    in_seq_length = len(in_seq[columns[0]])

    subsequence_starts = get_subsequence_starts(
        in_seq_length, seq_length, stride_for_split, subsequence_start_mode
    )
    subsequence_starts_diff = subsequence_starts[1:] - subsequence_starts[:-1]
    assert np.all(
        subsequence_starts_diff <= stride_for_split
    ), f"Diff of {subsequence_starts = }, {subsequence_starts_diff = } larger than {stride_for_split = }"

    return {
        col: [list(in_seq[col][i : i + seq_length + 1]) for i in subsequence_starts]
        for col in columns
    }


@beartype
def insert_top_folder(path: str, folder_name: str) -> str:
    """Inserts a directory name into a file path, just before the filename.

    Example:
        `insert_top_folder("a/b/c.txt", "temp")` returns `"a/b/temp/c.txt"`

    Args:
        path: The original file path.
        folder_name: The name of the folder to insert.

    Returns:
        The new path string with the folder inserted.
    """
    components = os.path.split(path)
    new_components = list(components[:-1]) + [folder_name] + [components[-1]]
    return os.path.join(*new_components)


@beartype
def cast_columns_to_string(data: pl.DataFrame) -> pl.DataFrame:
    """Casts the column names of a Polars DataFrame to strings.

    This is often necessary because Polars schemas may use integers as
    column names (e.g., '0', '1', '2'...) which need to be strings for
    some operations.

    Args:
        data: The Polars DataFrame.

    Returns:
        The same DataFrame with its `columns` attribute modified.
    """
    data.columns = [str(col) for col in data.columns]
    return data


@beartype
def delete_files(files: Union[list[str], dict[int, list[str]]]) -> None:
    """Deletes a list of files from the filesystem.

    Args:
        files: A list of file paths to delete, or a dictionary
            whose values are lists of file paths to delete.
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
    file_index_str: str,
    dataset_name: str,
    write_format: str,
) -> dict[int, list[str]]:
    """Creates a dictionary of temporary file paths for a specific data file.

    This is used in the multi-file, `combine_into_single_file=True`
    workflow. It generates file path names for intermediate batches
    *before* they are combined.

    The naming pattern is:
    `{dataset_name}-{process_id}-{file_index_str}-split{split}-{batch_id}.{write_format}`

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The temporary directory to place files in.
        n_splits: The number of data splits.
        n_batches: The number of batches created by the process.
        process_id: The ID of the multiprocessing worker.
        file_index_str: The index of the file being processed by this worker.
        dataset_name: The root name of the dataset.
        write_format: The file extension.

    Returns:
        A dictionary mapping a split index (int) to a list of file paths
        (str) for all batches in that split.
    """
    files = {}
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_path,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{file_index_str}-split{split}-{batch_id}.{write_format}",
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
    """Creates a dictionary of temporary file paths for a single-file run.

    This is used in the single-file, `combine_into_single_file=True`
    workflow. It generates file path names for intermediate batches
    created by different processes *before* they are combined.

    The naming pattern is:
    `{dataset_name}-split{split}-{core_id}.{write_format}`

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The temporary directory to place files in.
        n_splits: The number of data splits.
        n_batches: The number of processes (batches) running in parallel.
        dataset_name: The root name of the dataset.
        write_format: The file extension.

    Returns:
        A dictionary mapping a split index (int) to a list of file paths
        (str) for all batches in that split.
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
    """Creates a dictionary of intermediate file paths for a multi-file run.

    This is used in the multi-file, `combine_into_single_file=True`
    workflow. It generates the file paths for the *combined* files
    from each process, which are the *inputs* to the final combination step.

    The naming pattern is:
    `{dataset_name}-{process_id}-{file_index}-split{split}.{write_format}`

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The temporary directory where files are located.
        n_splits: The number of data splits.
        n_processes: The total number of multiprocessing workers.
        n_files: A dictionary mapping `process_id` to the number of
            files that process handled.
        dataset_name: The root name of the dataset.
        write_format: The file extension.

    Returns:
        A dictionary mapping a split index (int) to a list of all
        intermediate combined file paths (str) for that split.
    """
    files = {}
    n_files_max = max(n_files.values()) if n_files else 1
    pad_width = len(str(n_files_max - 1))
    for split in range(n_splits):
        files_for_split = [
            os.path.join(
                project_path,
                "data",
                target_dir,
                f"{dataset_name}-{process_id}-{str(file_index).zfill(pad_width)}-split{split}.{write_format}",
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
    """Combines multiple intermediate batch files into final split files.

    This function iterates through each split and combines all the
    intermediate files listed in `input_files[split]` into a single
    final output file for that split.

    - For "csv" format, it uses the `csvstack` command-line utility.
    - For "parquet" format, it uses `pyarrow.parquet.ParquetWriter`
      to concatenate the files efficiently.

    Args:
        project_path: The path to the sequifier project directory.
        target_dir: The temporary directory containing intermediate files.
        n_splits: The number of data splits.
        input_files: A dictionary mapping split index (int) to a list
            of input file paths (str) for that split.
        dataset_name: The root name for the final output files.
        write_format: The file format ("csv" or "parquet").
        in_target_dir: If True, the final combined file is written
            inside `target_dir`. If False, it's written to `data/`.
        pre_split_str: An optional string to insert into the filename
            before the "-split{i}" part.
        post_split_str: An optional string to insert into the filename
            after the "-split{i}" part.
    """
    for split in range(n_splits):
        split_file_path = create_split_file_path(
            project_path,
            dataset_name,
            split,
            write_format,
            in_target_dir,
            target_dir,
            pre_split_str,
            post_split_str,
        )

        print(f"[INFO] writing to: {split_file_path}")
        if write_format == "csv":
            command = " ".join(
                ["csvstack"] + input_files[split] + [f"> {split_file_path}"]
            )
            result = os.system(command)
            assert result == 0, f"command '{command}' failes: {result = }"
        elif write_format == "parquet":
            combine_parquet_files(input_files[split], split_file_path)


@beartype
def create_split_file_path(
    project_path: str,
    dataset_name: str,
    split: int,
    write_format: str,
    in_target_dir: bool,
    target_dir: str,
    pre_split_str: Optional[str],
    post_split_str: Optional[str],
) -> str:
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

    return out_path


@beartype
def combine_parquet_files(files: list[str], out_path: str) -> None:
    """Combines multiple Parquet files into a single Parquet file.

    This function reads the schema from the first file and uses it to
    initialize a `ParquetWriter`. It then iterates through all files in
    the list, reading each one as a table and writing it to the new
    combined file. This is more memory-efficient than reading all files
    into one large table first.

    Args:
        files: A list of paths to the Parquet files to combine.
        out_path: The path for the combined output Parquet file.
    """
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))
