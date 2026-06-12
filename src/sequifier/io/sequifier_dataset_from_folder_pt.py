import json
import math
import os
from typing import Dict, Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import generate_padding_masks, normalize_path
from sequifier.io.batch import SequifierBatch


class SequifierDatasetFromFolderPt(IterableDataset):
    """
    An efficient PyTorch IterableDataset that pre-loads all data into RAM.

    This strategy pays a one-time I/O cost at initialization, after which
    all data access during training is extremely fast. It yields full,
    pre-collated batches natively.
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
                "Ensure data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.n_samples = metadata["total_samples"]

        logger.info(
            f"[INFO] Loading training dataset into memory from '{self.data_dir}'..."
        )

        all_sequences: Dict[str, list[torch.Tensor]] = {
            col: [] for col in set(config.input_columns + config.target_columns)
        }
        all_left_pad_lengths: list[torch.Tensor] = []

        # Load all data files into RAM
        for file_info in metadata["batch_files"]:
            file_path = os.path.join(self.data_dir, file_info["path"])
            (
                sequences_batch,
                _,
                _,
                _,
                left_pad_lengths_batch,
            ) = torch.load(file_path, map_location="cpu", weights_only=False)
            for col in all_sequences.keys():
                if col in sequences_batch:
                    all_sequences[col].append(sequences_batch[col])
            if left_pad_lengths_batch is not None:
                all_left_pad_lengths.append(left_pad_lengths_batch)

        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors) for col, tensors in all_sequences.items() if tensors
        }
        self.left_pad_lengths = (
            torch.cat(all_left_pad_lengths)
            if all_left_pad_lengths
            and len(all_left_pad_lengths) == len(metadata["batch_files"])
            else None
        )
        for tensor in self.sequences.values():
            tensor.share_memory_()
        if self.left_pad_lengths is not None:
            self.left_pad_lengths.share_memory_()

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

        logger.info(
            f"[INFO] Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
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
        """Allows the training loop to set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def _get_target_samples(self) -> int:
        """Calculates exact sample count per rank to ensure FSDP syncs properly."""
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

        # 1. Global Shuffling
        indices = torch.arange(self.n_samples)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.config.seed + self.epoch)
            indices = indices[torch.randperm(self.n_samples, generator=g)]

        # 2. Distribute across GPU ranks
        indices_for_rank = indices[rank::world_size].tolist()

        # 3. Handle FSDP oversampling/undersampling sync requirements
        if self.sampling_strategy == "oversampling":
            # Loop the indices to pad out the shorter ranks
            while len(indices_for_rank) < self.target_samples:
                indices_for_rank.extend(
                    indices_for_rank[: self.target_samples - len(indices_for_rank)]
                )
        elif self.sampling_strategy == "undersampling":
            indices_for_rank = indices_for_rank[: self.target_samples]

        # 4. Distribute among CPU workers for this GPU
        indices_for_worker = indices_for_rank[worker_id::num_workers]

        # 5. Yield full batches
        train_seq_len = self.config.seq_length
        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]

            data_offset = self.config.training_spec.data_offset
            target_offset = self.config.training_spec.target_offset
            data_batch = {
                key: tensor[
                    batch_indices,
                    -(train_seq_len + data_offset) : (
                        -data_offset if data_offset > 0 else None
                    ),
                ]
                for key, tensor in self.sequences.items()
                if key in self.config.input_columns
            }
            targets_batch = {
                key: tensor[
                    batch_indices,
                    -(train_seq_len + target_offset) : (
                        -target_offset if target_offset > 0 else None
                    ),
                ]
                for key, tensor in self.sequences.items()
                if key in self.config.target_columns
            }

            metadata_batch = {}
            if self.left_pad_lengths is not None:
                metadata_batch = generate_padding_masks(
                    self.left_pad_lengths[batch_indices],
                    train_seq_len,
                    data_offset,
                    target_offset,
                )

            yield SequifierBatch(
                inputs=data_batch,
                targets=targets_batch,
                metadata=metadata_batch,
            )
