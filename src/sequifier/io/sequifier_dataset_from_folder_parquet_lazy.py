import os
from collections import Counter
from collections.abc import Iterator
from typing import Any

import polars as pl
import torch
from loguru import logger
from torch.utils.data import get_worker_info

from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    columns_from_slice,
    get_left_pad_lengths_from_preprocessed_data,
)
from sequifier.io.batch import SequifierBatch
from sequifier.io.folder_dataset import LazyFolderChunk, LazyFolderDataset
from sequifier.io.sample_order import curriculum_positions_from_parquet
from sequifier.io.window_sampling import (
    target_valid_from_offsets,
    validate_split_bounds_available,
)
from sequifier.typechecking import beartype


class SequifierDatasetFromFolderParquetLazy(LazyFolderDataset):
    """Streams long-format Parquet chunks into rank/worker-aligned batches."""

    chunk_format = "Parquet"

    @beartype
    def __init__(self, data_path: str, config: Any, shuffle: bool = True):
        super().__init__(data_path, config, shuffle, "merge_output: False")
        metadata = self.metadata
        self._initialize_window_layout()

        self.batch_files_info = []
        raw_file_infos = list(metadata["batch_files"])
        if self.file_order == "name":
            raw_file_infos.sort(key=lambda item: item["path"])
        for raw_file_info in raw_file_infos:
            file_info = dict(raw_file_info)
            file_info["stored_samples"] = int(raw_file_info["samples"])
            histogram = raw_file_info.get("target_valid_from_histogram")
            if histogram is None and not self.sampling_plan.legacy_single_window:
                file_path = os.path.join(self.data_dir, file_info["path"])
                position_rows = (
                    pl.scan_parquet(file_path)
                    .group_by(["sequenceId", "subsequenceId"])
                    .agg(
                        pl.col("leftPadLength").first(),
                        pl.col("startItemPosition").first(),
                        pl.col("splitStartItemPosition").first(),
                    )
                    .select(
                        pl.max_horizontal(
                            "leftPadLength",
                            pl.col("splitStartItemPosition")
                            - pl.col("startItemPosition"),
                        ).alias("targetValidFrom")
                    )
                    .collect()
                    .get_column("targetValidFrom")
                    .to_list()
                )
                histogram = {
                    str(value): count for value, count in Counter(position_rows).items()
                }
            if self.sampling_plan.legacy_single_window:
                file_info["samples"] = file_info["stored_samples"]
            else:
                assert histogram is not None
                file_info["samples"] = (
                    self.sampling_plan.sample_count_from_target_valid_from_histogram(
                        histogram
                    )
                )
            if file_info["samples"] > 0:
                self.batch_files_info.append(file_info)
        self.total_samples = sum(info["samples"] for info in self.batch_files_info)
        if self.total_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")

        self.column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_data_types[col]]
            for col in config.column_data_types
        }

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)
        logger.info(
            f"Lazy Parquet Dataset mapped with {self.target_samples} samples and {self.total_batches} batches."
        )

    @beartype
    def __iter__(self) -> Iterator[SequifierBatch]:
        yield from self._iter_batches(get_worker_info())

    def _read_chunk(self, file_path: str) -> LazyFolderChunk:
        df = pl.read_parquet(file_path)
        left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(df)
        position_rows = (
            df.group_by(["sequenceId", "subsequenceId"])
            .agg(
                pl.col("startItemPosition").first(),
                pl.col("splitStartItemPosition").first(),
            )
            .sort(["sequenceId", "subsequenceId"])
        )
        sample_index = self.sampling_plan.build_index(
            left_pad_lengths,
            target_valid_from_offsets(
                left_pad_lengths,
                torch.tensor(
                    position_rows["startItemPosition"].to_numpy(),
                    dtype=torch.int64,
                ),
                torch.tensor(
                    position_rows["splitStartItemPosition"].to_numpy(),
                    dtype=torch.int64,
                ),
            ),
        )
        sample_positions = curriculum_positions_from_parquet(self.config, df, file_path)

        start_item_positions = None
        split_start_item_positions = None
        split_end_item_positions = None
        if {"splitStartItemPosition", "splitEndItemPosition"} <= set(df.columns):
            positions = (
                df.group_by(["sequenceId", "subsequenceId"])
                .agg(
                    pl.col("startItemPosition").first(),
                    pl.col("splitStartItemPosition").first(),
                    pl.col("splitEndItemPosition").first(),
                )
                .sort(["sequenceId", "subsequenceId"])
            )
            start_item_positions = torch.tensor(
                positions["startItemPosition"].to_numpy(), dtype=torch.int64
            )
            split_start_item_positions = torch.tensor(
                positions["splitStartItemPosition"].to_numpy(), dtype=torch.int64
            )
            split_end_item_positions = torch.tensor(
                positions["splitEndItemPosition"].to_numpy(), dtype=torch.int64
            )
        validate_split_bounds_available(
            start_item_positions,
            split_start_item_positions,
            split_end_item_positions,
            file_path,
        )
        assert start_item_positions is not None
        assert split_start_item_positions is not None
        assert split_end_item_positions is not None

        sequence_columns = columns_from_slice(
            slice(0, self.folder_layout.window_length),
            self.folder_layout.window_length,
        )
        feature_partitions = {
            frame.item(0, "inputCol"): frame for frame in df.partition_by("inputCol")
        }
        stored_sequences = {}
        for col_name in set(self.config.input_columns + self.config.target_columns):
            if col_name in feature_partitions:
                stored_sequences[col_name] = torch.tensor(
                    feature_partitions[col_name]
                    .sort(["sequenceId", "subsequenceId"])
                    .select(sequence_columns)
                    .to_numpy(),
                    dtype=self.column_torch_types[col_name],
                )
            else:
                raise ValueError(f"Column not found in input data: {col_name}")

        return LazyFolderChunk(
            sequences=stored_sequences,
            sample_index=sample_index,
            sample_positions=sample_positions,
            start_item_positions=start_item_positions,
            split_start_item_positions=split_start_item_positions,
            split_end_item_positions=split_end_item_positions,
        )
