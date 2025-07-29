import json
import math
import multiprocessing
import os
import shutil
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from beartype import beartype

from sequifier.config.preprocess_config import load_preprocessor_config
from sequifier.helpers import read_data, write_data


@beartype
def preprocess(args: Any, args_config: dict[str, Any]) -> None:
    """
    Main preprocessing function.

    Args:
        args: Command line arguments.
        args_config: Configuration dictionary.
    """
    config_path = args.config_path or "configs/preprocess.yaml"
    config = load_preprocessor_config(config_path, args_config)
    Preprocessor(**config.dict())
    print("Preprocessing complete")


class Preprocessor:
    @beartype
    def __init__(
        self,
        project_path: str,
        data_path: str,
        read_format: str,
        write_format: str,
        selected_columns: Optional[list[str]],
        group_proportions: list[float],
        seq_length: int,
        seq_step_sizes: list[int],
        max_rows: Optional[int],
        seed: int,
        n_cores: Optional[int],
    ):
        self.project_path = project_path
        self.seed = seed
        np.random.seed(seed)

        self._setup_directories()

        if selected_columns is not None:
            selected_columns = ["sequenceId", "itemPosition"] + selected_columns

        data = self._load_and_preprocess_data(
            data_path, read_format, selected_columns, max_rows
        )
        self._setup_split_paths(write_format, len(group_proportions))

        data_columns = [
            col for col in data.columns if col not in ["sequenceId", "itemPosition"]
        ]
        data, n_classes, id_maps, selected_columns_statistics, col_types = (
            self._process_columns(data, data_columns)
        )
        self._export_metadata(
            id_maps, n_classes, col_types, selected_columns_statistics
        )

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

        data = data.sort(["sequenceId", "itemPosition"])
        self._process_batches(
            data,
            schema,
            n_cores,
            seq_length,
            seq_step_sizes,
            data_columns,
            group_proportions,
            write_format,
        )
        self._cleanup()

    @beartype
    def _setup_directories(self) -> None:
        os.makedirs(os.path.join(self.project_path, "data"), exist_ok=True)
        temp_path = os.path.join(self.project_path, "data", "temp")
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path)

    @beartype
    def _load_and_preprocess_data(
        self,
        data_path: str,
        read_format: str,
        selected_columns: Optional[list[str]],
        max_rows: Optional[int],
    ) -> pl.DataFrame:
        data = read_data(data_path, read_format, columns=selected_columns)
        assert (
            data.null_count().sum().sum_horizontal().item() == 0
        ), f"NaN or null values not accepted: {data.null_count()}"
        self.data_name_root = os.path.splitext(os.path.basename(data_path))[0]

        if selected_columns:
            selected_columns_filtered = [
                col
                for col in selected_columns
                if col not in ["sequenceId", "itemPosition"]
            ]
            data = data.select(
                ["sequenceId", "itemPosition"] + selected_columns_filtered
            )

        if max_rows:
            data = data.slice(0, int(max_rows))

        return data

    @beartype
    def _setup_split_paths(self, write_format: str, n_splits: int) -> None:
        self.split_paths = [
            os.path.join(
                self.project_path,
                "data",
                f"{self.data_name_root}-split{i}.{write_format}",
            )
            for i in range(n_splits)
        ]

    @beartype
    def _process_columns(
        self, data: pl.DataFrame, data_columns: list[str]
    ) -> tuple[
        pl.DataFrame,
        dict[str, int],
        dict[str, dict[Union[str, int], int]],
        dict[str, dict[str, float]],
        dict[str, str],
    ]:
        n_classes, id_maps = {}, {}
        selected_columns_statistics = {}
        float_data_columns = []

        for data_col in data_columns:
            dtype = data.schema[data_col]
            if isinstance(dtype, (pl.String, pl.Utf8)) or isinstance(
                dtype, (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            ):
                data, sup_id_map = replace_ids(data, column=data_col)
                id_maps[data_col] = dict(sup_id_map)
                n_classes[data_col] = data.get_column(data_col).n_unique() + 1
            elif isinstance(dtype, (pl.Float32, pl.Float64)):
                std = data.get_column(data_col).std()
                mean = data.get_column(data_col).mean()
                data = data.with_columns(
                    ((pl.col(data_col) - mean) / (std + 1e-9)).alias(data_col)
                )
                selected_columns_statistics[data_col] = {"std": std, "mean": mean}
                float_data_columns.append(data_col)
            else:
                raise ValueError(f"Column {data_col} has unsupported dtype: {dtype}")

        col_types = {col: str(data.schema[col]) for col in data_columns}
        return data, n_classes, id_maps, selected_columns_statistics, col_types

    @beartype
    def _process_batches(
        self,
        data: pl.DataFrame,
        schema: Any,
        n_cores: Optional[int],
        seq_length: int,
        seq_step_sizes: list[int],
        data_columns: list[str],
        group_proportions: list[float],
        write_format: str,
    ) -> None:
        n_cores = n_cores or multiprocessing.cpu_count()
        batch_limits = get_batch_limits(data, n_cores)
        batches = [
            (
                i,
                data.slice(start, end - start),
                schema,
                self.split_paths,
                seq_length,
                seq_step_sizes,
                data_columns,
                group_proportions,
                write_format,
            )
            for i, (start, end) in enumerate(batch_limits)
            if (end - start) > 0
        ]

        with multiprocessing.Pool(processes=len(batches)) as pool:
            pool.starmap(preprocess_batch, batches)

        combine_multiprocessing_outputs(
            self.project_path,
            len(group_proportions),
            len(batches),
            self.data_name_root,
            write_format,
        )

    @beartype
    def _cleanup(self) -> None:
        delete_path = os.path.join(self.project_path, "data", "temp")
        assert len(delete_path) > 9
        os.system(f"rm -rf {delete_path}*")

    @beartype
    def _export_metadata(
        self,
        id_maps: dict[str, dict[Union[str, int], int]],
        n_classes: dict[str, int],
        col_types: dict[str, str],
        selected_columns_statistics: dict[str, dict[str, float]],
    ) -> None:
        data_driven_config = {
            "n_classes": n_classes,
            "id_maps": id_maps,
            "split_paths": self.split_paths,
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
def replace_ids(
    data: pl.DataFrame, column: str
) -> tuple[pl.DataFrame, dict[Union[str, int], int]]:
    ids = sorted(
        [int(x) if not isinstance(x, str) else x for x in np.unique(data[column])]
    )  # type: ignore
    id_map = {id_: i + 1 for i, id_ in enumerate(ids)}
    data = data.with_columns(pl.col(column).replace(id_map))
    return data, id_map


@beartype
def get_batch_limits(data: pl.DataFrame, n_batches: int) -> list[tuple[int, int]]:
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
def get_group_bounds(data_subset: pl.DataFrame, group_proportions: list[float]):
    n = data_subset.shape[0]
    upper_bounds = list((np.cumsum(group_proportions) * n).astype(int))
    lower_bounds = [0] + list(upper_bounds[:-1])
    group_bounds = list(zip(lower_bounds, upper_bounds))
    return group_bounds


@beartype
def preprocess_batch(
    process_id: int,
    batch: pl.DataFrame,
    schema: Any,
    split_paths: list[str],
    seq_length: int,
    seq_step_sizes: list[int],
    data_columns: list[str],
    group_proportions: list[float],
    write_format: str,
) -> None:
    sequence_ids = sorted(batch.get_column("sequenceId").unique().to_list())
    written_files: dict[int, list[str]] = {i: [] for i in range(len(split_paths))}
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

        for split_path, (group, split) in zip(split_paths, sequences.items()):
            split_path_batch_seq = split_path.replace(
                f".{write_format}", f"-{process_id}-{i}.{write_format}"
            )
            split_path_batch_seq = insert_top_folder(split_path_batch_seq, "temp")

            if write_format == "csv":
                write_data(split, split_path_batch_seq, "csv")
            elif write_format == "parquet":
                write_data(split, split_path_batch_seq, "parquet")

            written_files[group].append(split_path_batch_seq)

    for j, split_path in enumerate(split_paths):
        out_path = split_path.replace(
            f".{write_format}", f"-{process_id}.{write_format}"
        )
        out_path = insert_top_folder(out_path, "temp")

        if write_format == "csv":
            command = " ".join(["csvstack"] + written_files[j] + [f"> {out_path}"])
            os.system(command)
        elif write_format == "parquet":
            combine_parquet_files(written_files[j], out_path)


@beartype
def extract_sequences(
    data: pl.DataFrame,
    schema: Any,
    seq_length: int,
    seq_step_size: int,
    columns: list[str],
) -> pl.DataFrame:
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
    components = os.path.split(path)
    new_components = list(components[:-1]) + [folder_name] + [components[-1]]
    return os.path.join(*new_components)


@beartype
def cast_columns_to_string(data: pl.DataFrame) -> pl.DataFrame:
    data.columns = [str(col) for col in data.columns]
    return data


@beartype
def combine_multiprocessing_outputs(
    project_path: str,
    n_splits: int,
    n_batches: int,
    dataset_name: str,
    write_format: str,
) -> None:
    for split in range(n_splits):
        out_path = os.path.join(
            project_path, "data", f"{dataset_name}-split{split}.{write_format}"
        )

        files = [
            os.path.join(
                project_path,
                "data",
                "temp",
                f"{dataset_name}-split{split}-{batch}.{write_format}",
            )
            for batch in range(n_batches)
        ]
        if write_format == "csv":
            command = " ".join(["csvstack"] + files + [f"> {out_path}"])
            os.system(command)
        elif write_format == "parquet":
            combine_parquet_files(files, out_path)


@beartype
def combine_parquet_files(files: list[str], out_path: str) -> None:
    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(out_path, schema=schema, compression="snappy") as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))
