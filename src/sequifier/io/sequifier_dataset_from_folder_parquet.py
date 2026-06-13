import json
import math
import os
from typing import Dict, Iterator

import polars as pl
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    generate_padding_masks,
    get_left_pad_lengths_from_preprocessed_data,
    normalize_path,
    sequence_column_names,
    sequence_layout_from_metadata,
)
from sequifier.io.batch import SequifierBatch


class SequifierDatasetFromFolderParquet(IterableDataset):
    """
    An efficient PyTorch IterableDataset that pre-loads a folder of chunked
    Parquet files entirely into CPU RAM at initialization.

    Yields full, pre-collated batches natively. Fully supports DDP/FSDP distributed
    environments using customizable sampling strategies.
    """

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.data_dir = normalize_path(data_path, config.project_root)
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.sampling_strategy = config.training_spec.sampling_strategy

        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Ensure data is pre-processed with merge_output: False."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        folder_layout = sequence_layout_from_metadata(metadata)
        if folder_layout.sample_length != config.layout.sample_length:
            raise ValueError(
                f"Preprocessed folder sample_length={folder_layout.sample_length} "
                f"does not match config sample_length={config.layout.sample_length}."
            )

        self.n_samples = metadata["total_samples"]

        logger.info(
            f"[INFO] Loading Parquet folder dataset into memory from '{self.data_dir}'..."
        )

        column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        # Sequence formatting structures matching long-format schema boundaries
        train_seq_len = self.config.layout.context_length
        input_seq_cols = sequence_column_names(
            train_seq_len, self.config.layout.input_offset
        )
        target_seq_cols = sequence_column_names(
            train_seq_len,
            self.config.layout.get_target_offset(
                self.config.training_spec.training_objective
            ),
        )
        all_sequences: Dict[str, list[torch.Tensor]] = {
            col: [] for col in config.input_columns
        }
        all_targets: Dict[str, list[torch.Tensor]] = {
            col: [] for col in config.target_columns
        }
        all_left_pad_lengths: list[torch.Tensor] = []

        # Step 1: Eager I/O reduction pass over all chunk allocations
        for file_info in metadata["batch_files"]:
            file_path = os.path.join(self.data_dir, file_info["path"])
            df = pl.read_parquet(file_path)

            left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(df)
            if left_pad_lengths is not None:
                all_left_pad_lengths.append(left_pad_lengths)

            for col in all_sequences.keys():
                feature_df = df.filter(pl.col("inputCol") == col)
                if not feature_df.is_empty():
                    tensor_seq = torch.tensor(
                        feature_df.select(input_seq_cols).to_numpy(),
                        dtype=column_torch_types[col],
                    )
                    all_sequences[col].append(tensor_seq)

            for col in all_targets.keys():
                feature_df = df.filter(pl.col("inputCol") == col)
                if not feature_df.is_empty():
                    tensor_tgt = torch.tensor(
                        feature_df.select(target_seq_cols).to_numpy(),
                        dtype=column_torch_types[col],
                    )
                    all_targets[col].append(tensor_tgt)
            del df

        # Step 2: Consolidate data lists into contiguous blocks
        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors, dim=0)
            for col, tensors in all_sequences.items()
            if tensors
        }
        self.targets: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors, dim=0)
            for col, tensors in all_targets.items()
            if tensors
        }
        self.left_pad_lengths = torch.cat(all_left_pad_lengths)

        # Step 3: Prevent serialization duplications across worker forks via shared memory flags
        for tensor in self.sequences.values():
            tensor.share_memory_()
        for tensor in self.targets.values():
            tensor.share_memory_()
        if self.left_pad_lengths is not None:
            self.left_pad_lengths.share_memory_()

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

        logger.info(
            f"[INFO] Parquet Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
        )

    def _calculate_total_batches(self, target_samples: int) -> int:
        num_workers = self.config.training_spec.num_workers
        num_workers_to_use = num_workers if num_workers > 0 else 1

        total_batches = 0
        for worker_id in range(num_workers_to_use):
            worker_samples = target_samples // num_workers_to_use + (
                1 if worker_id < target_samples % num_workers_to_use else 0
            )
            total_batches += math.ceil(worker_samples / self.batch_size)
        return total_batches

    def set_epoch(self, epoch: int):
        """Allows the training loop to synchronize seed steps for shuffling."""
        self.epoch = epoch

    def _get_target_samples(self) -> int:
        """Calculates precise sample counts per rank to manage FSDP layer allocations."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        samples_per_rank = [
            len(range(r, self.n_samples, world_size)) for r in range(world_size)
        ]

        if self.sampling_strategy == "exact":
            return samples_per_rank[rank]
        elif self.sampling_strategy == "oversampling":
            return max(samples_per_rank)
        elif self.sampling_strategy == "undersampling":
            return min(samples_per_rank)
        return samples_per_rank[rank]

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # 1. Coordinate global shuffling masks
        indices = torch.arange(self.n_samples)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.config.seed + self.epoch)
            indices = indices[torch.randperm(self.n_samples, generator=g)]

        # 2. Slice metrics based on GPU distribution metrics
        indices_for_rank = indices[rank::world_size].tolist()

        # 3. Synchronize cross-device oversampling/undersampling rules
        if self.sampling_strategy == "oversampling":
            while len(indices_for_rank) < self.target_samples:
                indices_for_rank.extend(
                    indices_for_rank[: self.target_samples - len(indices_for_rank)]
                )
        elif self.sampling_strategy == "undersampling":
            indices_for_rank = indices_for_rank[: self.target_samples]

        # 4. Map worker task splits
        indices_for_worker = indices_for_rank[worker_id::num_workers]

        # 5. Extract and pass unified data frames
        train_seq_len = self.config.layout.context_length
        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]

            data_batch = {
                key: tensor[batch_indices, -train_seq_len:]
                for key, tensor in self.sequences.items()
            }
            targets_batch = {
                key: tensor[batch_indices, -train_seq_len:]
                for key, tensor in self.targets.items()
            }

            metadata_batch = {}
            if self.left_pad_lengths is not None:
                metadata_batch = generate_padding_masks(
                    self.left_pad_lengths[batch_indices],
                    train_seq_len,
                    self.config.layout.sample_length,
                    self.config.layout.input_offset,
                    self.config.layout.get_target_offset(
                        self.config.training_spec.training_objective
                    ),
                )

            yield SequifierBatch(
                inputs=data_batch,
                targets=targets_batch,
                metadata=metadata_batch,
            )
