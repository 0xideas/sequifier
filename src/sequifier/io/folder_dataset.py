"""Shared folder-dataset setup and distributed batch bookkeeping."""

import json
import math
import os
from typing import Any

import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.helpers import (
    configured_window_stride,
    normalize_path,
    resolve_window_sampling_plan,
    stored_window_layout_from_metadata,
)
from sequifier.io.config import global_training
from sequifier.io.iteration_state import (
    read_shared_int,
    resolve_resume_worker,
    shared_int,
    skip_samples_for_batches,
    write_shared_int,
)
from sequifier.io.sample_order import (
    concatenate_file_orders,
    configured_file_order,
    validate_folder_curriculum,
)
from sequifier.typechecking import beartype


class FolderDataset(IterableDataset):
    """Initialize the metadata and state shared by every folder loader."""

    def __init__(self, data_path: str, config: Any, shuffle: bool, format_hint: str):
        super().__init__()
        # DataLoader spawn workers do not inherit the process group.
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.data_dir = normalize_path(data_path, config.project_root)
        self.config = config
        self.batch_size = global_training(config).batch_size
        self.shuffle = shuffle
        self._epoch_state = shared_int(0)
        self._start_batch_state = shared_int(0)

        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                f"Ensure data is pre-processed with {format_hint}."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        validate_folder_curriculum(config, metadata, self.data_dir)

        self.metadata = metadata

    def _initialize_window_layout(self) -> None:
        self.folder_layout = stored_window_layout_from_metadata(self.metadata)
        self.sampling_plan = resolve_window_sampling_plan(
            self.folder_layout,
            self.config.window_view,
            configured_window_stride(self.config),
        )
        self.file_order = configured_file_order(self.config)

    @beartype
    def _calculate_total_batches(self, target_samples: int) -> int:
        num_workers = global_training(self.config).num_workers
        num_workers_to_use = num_workers if num_workers > 0 else 1

        total_batches = 0
        for worker_id in range(num_workers_to_use):
            worker_samples = target_samples // num_workers_to_use + (
                1 if worker_id < target_samples % num_workers_to_use else 0
            )
            total_batches += math.ceil(worker_samples / self.batch_size)
        return total_batches

    @beartype
    def set_epoch(self, epoch: int):
        """Set the shuffle epoch."""
        write_shared_int(self._epoch_state, epoch)

    @beartype
    def set_start_batch(self, start_batch: int):
        """Set the first global batch to yield on the next iteration."""
        write_shared_int(self._start_batch_state, start_batch)

    @beartype
    def __len__(self) -> int:
        return self.total_batches


class EagerFolderDataset(FolderDataset):
    """Shared sample sharding for fully materialized folders."""

    @beartype
    def _get_target_samples(self) -> int:
        world_size = self.world_size
        samples_per_rank = [
            len(range(r, self.n_samples, world_size)) for r in range(world_size)
        ]
        return max(samples_per_rank)

    def _worker_indices(self) -> tuple[list[int], list[bool]]:
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = read_shared_int(self._epoch_state)
        start_batch = read_shared_int(self._start_batch_state)

        indices = concatenate_file_orders(
            self.file_sample_orders,
            seed=self.config.seed,
            epoch=epoch,
            shuffle=self.shuffle,
            file_order=self.file_order,
        )
        indices_for_rank = indices[self.rank :: self.world_size].tolist()
        sample_is_real = [True] * len(indices_for_rank)

        real_count = len(indices_for_rank)
        if real_count == 0:
            fallback_indices = indices.tolist()
            n = min(len(fallback_indices), self.target_samples)
            indices_for_rank.extend(fallback_indices[:n])
            sample_is_real.extend([False] * n)
        else:
            while len(indices_for_rank) < self.target_samples:
                n = min(real_count, self.target_samples - len(indices_for_rank))
                indices_for_rank.extend(indices_for_rank[:n])
                sample_is_real.extend([False] * n)

        worker_batch_counts = [
            math.ceil(len(indices_for_rank[i::num_workers]) / self.batch_size)
            for i in range(num_workers)
        ]
        worker_id, skip_batches = resolve_resume_worker(
            start_batch,
            physical_worker_id,
            num_workers,
            worker_batch_counts,
        )
        indices_for_worker = indices_for_rank[worker_id::num_workers]
        sample_is_real_for_worker = sample_is_real[worker_id::num_workers]
        skipped_samples = skip_samples_for_batches(
            skip_batches, self.batch_size, len(indices_for_worker)
        )
        return (
            indices_for_worker[skipped_samples:],
            sample_is_real_for_worker[skipped_samples:],
        )


class LazyFolderDataset(FolderDataset):
    """Shared per-file sample accounting for streamed folders."""

    @beartype
    def _get_target_samples(self) -> int:
        world_size = self.world_size
        num_files = len(self.batch_files_info)
        samples_per_rank = []
        for r in range(world_size):
            f_r = list(range(r, num_files, world_size))
            samples_per_rank.append(
                sum(self.batch_files_info[i]["samples"] for i in f_r) if f_r else 0
            )
        return max(samples_per_rank)
