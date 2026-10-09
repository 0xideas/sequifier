import os
from collections.abc import Iterator
from typing import Any, Dict

import polars as pl
import torch
from loguru import logger

from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    columns_from_slice,
    get_left_pad_lengths_from_preprocessed_data,
)
from sequifier.io.batch import SequifierBatch
from sequifier.io.folder_dataset import EagerFolderDataset
from sequifier.io.sample_order import (
    SampleOrderPlan,
    curriculum_positions_from_parquet,
    logical_sample_positions,
)
from sequifier.io.window_sampling import (
    build_window_batch,
    target_valid_from_offsets,
    validate_split_bounds_available,
)
from sequifier.typechecking import beartype


class SequifierDatasetFromFolderParquet(EagerFolderDataset):
    """Eager Parquet-folder dataset yielding rank/worker-aligned batches."""

    @beartype
    def __init__(self, data_path: str, config: Any, shuffle: bool = True):
        super().__init__(data_path, config, shuffle, "merge_output: False")
        metadata = self.metadata
        self._initialize_window_layout()

        logger.info(
            f"Loading Parquet folder dataset into memory from '{self.data_dir}'..."
        )

        column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_data_types[col]]
            for col in config.column_data_types
        }

        # Sequence formatting structures matching long-format schema boundaries
        sequence_columns = columns_from_slice(
            slice(0, self.folder_layout.window_length),
            self.folder_layout.window_length,
        )
        all_sequences: Dict[str, list[torch.Tensor]] = {
            col: [] for col in set(config.input_columns + config.target_columns)
        }
        all_left_pad_lengths: list[torch.Tensor] = []
        all_start_item_positions: list[torch.Tensor] = []
        all_split_start_item_positions: list[torch.Tensor] = []
        all_split_end_item_positions: list[torch.Tensor] = []
        self.file_sample_orders: list[tuple[int, SampleOrderPlan]] = []
        logical_offset = 0

        # Step 1: Eager I/O reduction pass over all chunk allocations
        file_infos = list(metadata["batch_files"])
        if self.file_order == "name":
            file_infos.sort(key=lambda item: item["path"])
        for file_info in file_infos:
            file_path = os.path.join(self.data_dir, file_info["path"])
            df = pl.read_parquet(file_path)

            left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(df)
            if left_pad_lengths is not None:
                all_left_pad_lengths.append(left_pad_lengths)
            required_position_columns = {
                "startItemPosition",
                "splitStartItemPosition",
                "splitEndItemPosition",
            }
            if not required_position_columns <= set(df.columns):
                raise ValueError(
                    f"Stored windows in {file_path!r} are missing required position "
                    "or split-boundary columns; re-run preprocessing with the "
                    "current format."
                )
            positions = (
                df.group_by(["sequenceId", "subsequenceId"])
                .agg(
                    pl.col("startItemPosition").first(),
                    pl.col("splitStartItemPosition").first(),
                    pl.col("splitEndItemPosition").first(),
                )
                .sort(["sequenceId", "subsequenceId"])
            )
            all_start_item_positions.append(
                torch.tensor(
                    positions["startItemPosition"].to_numpy(), dtype=torch.int64
                )
            )
            all_split_start_item_positions.append(
                torch.tensor(
                    positions["splitStartItemPosition"].to_numpy(),
                    dtype=torch.int64,
                )
            )
            all_split_end_item_positions.append(
                torch.tensor(
                    positions["splitEndItemPosition"].to_numpy(),
                    dtype=torch.int64,
                )
            )
            local_sample_index = self.sampling_plan.build_index(
                left_pad_lengths,
                target_valid_from_offsets(
                    left_pad_lengths,
                    all_start_item_positions[-1],
                    all_split_start_item_positions[-1],
                ),
            )
            local_sample_count = len(local_sample_index)
            self.file_sample_orders.append(
                (
                    logical_offset,
                    SampleOrderPlan.build(
                        local_sample_count,
                        logical_sample_positions(
                            curriculum_positions_from_parquet(config, df, file_path),
                            local_sample_index,
                        ),
                    ),
                )
            )
            logical_offset += local_sample_count

            for col in all_sequences.keys():
                feature_df = df.filter(pl.col("inputCol") == col)
                if not feature_df.is_empty():
                    tensor_seq = torch.tensor(
                        feature_df.sort(["sequenceId", "subsequenceId"])
                        .select(sequence_columns)
                        .to_numpy(),
                        dtype=column_torch_types[col],
                    )
                    all_sequences[col].append(tensor_seq)
            del df

        # Step 2: Consolidate data lists into contiguous blocks
        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors, dim=0)
            for col, tensors in all_sequences.items()
            if tensors
        }
        self.left_pad_lengths = torch.cat(all_left_pad_lengths)
        self.start_item_positions = torch.cat(all_start_item_positions)
        self.split_start_item_positions = torch.cat(all_split_start_item_positions)
        self.split_end_item_positions = torch.cat(all_split_end_item_positions)
        validate_split_bounds_available(
            self.start_item_positions,
            self.split_start_item_positions,
            self.split_end_item_positions,
            self.data_dir,
        )
        self.sample_index = self.sampling_plan.build_index(
            self.left_pad_lengths,
            target_valid_from_offsets(
                self.left_pad_lengths,
                self.start_item_positions,
                self.split_start_item_positions,
            ),
        )
        self.n_samples = len(self.sample_index)
        if self.n_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")

        # Step 3: Prevent serialization duplications across worker forks via shared memory flags
        for tensor in self.sequences.values():
            tensor.share_memory_()
        self.sample_index.share_memory_()

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

        logger.info(
            f"Parquet Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
        )

    @beartype
    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        indices_for_worker, sample_is_real_for_worker = self._worker_indices()

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]
            batch_sample_is_real = sample_is_real_for_worker[i : i + self.batch_size]

            yield build_window_batch(
                self.sequences,
                self.config.input_columns,
                self.config.target_columns,
                self.sample_index,
                batch_indices,
                batch_sample_is_real,
                start_item_positions=self.start_item_positions,
                split_start_item_positions=self.split_start_item_positions,
                split_end_item_positions=self.split_end_item_positions,
            )
