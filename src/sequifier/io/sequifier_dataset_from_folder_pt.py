import json
import math
import os
from typing import Dict, Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import (
    normalize_path,
    resolve_window_view,
    stored_window_layout_from_metadata,
    validate_stored_window_width,
)
from sequifier.io.batch import SequifierBatch


class SequifierDatasetFromFolderPt(IterableDataset):
    """Eager PT-folder dataset yielding rank/worker-aligned batches."""

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

        self.folder_layout = stored_window_layout_from_metadata(metadata)
        self.resolved_view = resolve_window_view(self.folder_layout, config.window_view)

        self.n_samples = metadata["total_samples"]

        logger.info(
            f"[INFO] Loading training dataset into memory from '{self.data_dir}'..."
        )

        all_sequences: Dict[str, list[torch.Tensor]] = {
            col: [] for col in set(config.input_columns + config.target_columns)
        }
        all_left_pad_lengths: list[torch.Tensor] = []

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
                    validate_stored_window_width(
                        sequences_batch[col], self.folder_layout.stored_context_width
                    )
                    all_sequences[col].append(sequences_batch[col])
            all_left_pad_lengths.append(left_pad_lengths_batch)

        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors) for col, tensors in all_sequences.items() if tensors
        }
        self.left_pad_lengths = torch.cat(all_left_pad_lengths)
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
        """Set the shuffle epoch."""
        self.epoch = epoch

    def _get_target_samples(self) -> int:
        """Return per-rank sample count under the configured sampling strategy."""
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

        indices = torch.arange(self.n_samples)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.config.seed + self.epoch)
            indices = indices[torch.randperm(self.n_samples, generator=g)]

        indices_for_rank = indices[rank::world_size].tolist()
        sample_is_real = [True] * len(indices_for_rank)

        if self.sampling_strategy == "oversampling":
            real_count = len(indices_for_rank)
            if real_count == 0:
                fallback_indices = indices.tolist()
                while len(indices_for_rank) < self.target_samples and fallback_indices:
                    n = min(
                        len(fallback_indices),
                        self.target_samples - len(indices_for_rank),
                    )
                    indices_for_rank.extend(fallback_indices[:n])
                    sample_is_real.extend([False] * n)
            while len(indices_for_rank) < self.target_samples:
                n = min(real_count, self.target_samples - len(indices_for_rank))
                indices_for_rank.extend(indices_for_rank[:n])
                sample_is_real.extend([False] * n)
        elif self.sampling_strategy == "undersampling":
            indices_for_rank = indices_for_rank[: self.target_samples]
            sample_is_real = sample_is_real[: self.target_samples]

        indices_for_worker = indices_for_rank[worker_id::num_workers]
        sample_is_real_for_worker = sample_is_real[worker_id::num_workers]

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]
            batch_sample_is_real = sample_is_real_for_worker[i : i + self.batch_size]

            data_batch = {
                key: tensor[batch_indices, self.resolved_view.input_slice]
                for key, tensor in self.sequences.items()
                if key in self.config.input_columns
            }
            targets_batch = {
                key: tensor[batch_indices, self.resolved_view.target_slice]
                for key, tensor in self.sequences.items()
                if key in self.config.target_columns
            }

            metadata_batch = {}
            if self.left_pad_lengths is not None:
                metadata_batch = self.resolved_view.build_masks(
                    self.left_pad_lengths[batch_indices]
                )
            metadata_batch["sample_valid_mask"] = torch.tensor(
                batch_sample_is_real, dtype=torch.bool
            )

            yield SequifierBatch(
                inputs=data_batch,
                targets=targets_batch,
                metadata=metadata_batch,
            )
