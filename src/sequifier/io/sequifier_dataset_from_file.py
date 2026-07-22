import math
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    configured_model_window_stride,
    numpy_storage_to_pytorch,
    read_data,
    resolve_window_sampling_plan,
)
from sequifier.io.batch import SequifierBatch
from sequifier.io.iteration_state import (
    read_shared_int,
    resolve_resume_worker,
    shared_int,
    skip_samples_for_batches,
    write_shared_int,
)
from sequifier.io.window_sampling import build_window_batch


class SequifierDatasetFromFile(IterableDataset):
    """Eager single-file dataset yielding pre-collated batches."""

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self._epoch_state = shared_int(0)
        self._start_batch_state = shared_int(0)

        all_columns = sorted(list(set(config.input_columns + config.target_columns)))

        logger.info(
            f"[INFO] Loading training dataset into memory from '{data_path}'..."
        )
        data_df = read_data(data_path, config.read_format)

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        sampling_plan = resolve_window_sampling_plan(
            config.storage_layout,
            config.window_view,
            configured_model_window_stride(config),
        )
        all_tensors, left_pad_lengths = numpy_storage_to_pytorch(
            data=data_df,
            column_types=column_types,
            all_columns=all_columns,
            stored_context_width=config.storage_layout.stored_context_width,
        )
        self.sample_index = sampling_plan.build_index(left_pad_lengths)
        self.n_samples = len(self.sample_index)
        if self.n_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")

        del data_df

        self.sequences = all_tensors

        if config.training_spec.device.startswith("cuda"):
            for key in self.sequences:
                self.sequences[key] = self.sequences[key].pin_memory()

        logger.info(f"[INFO] Dataset loaded with {self.n_samples} samples.")

    def set_epoch(self, epoch: int):
        """Set the shuffle epoch."""
        write_shared_int(self._epoch_state, epoch)

    def set_start_batch(self, start_batch: int):
        """Set the first global batch to yield on the next iteration."""
        write_shared_int(self._start_batch_state, start_batch)

    def __len__(self) -> int:
        num_workers = max(1, self.config.training_spec.num_workers)
        total_batches = 0
        for worker_id in range(num_workers):
            worker_samples = self.n_samples // num_workers + (
                1 if worker_id < self.n_samples % num_workers else 0
            )
            total_batches += math.ceil(worker_samples / self.batch_size)
        return total_batches

    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        worker_info = torch.utils.data.get_worker_info()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if worker_info is None:
            # Single-process data loading
            physical_worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            physical_worker_id = worker_info.id
            num_workers = worker_info.num_workers
        epoch = read_shared_int(self._epoch_state)
        start_batch = read_shared_int(self._start_batch_state)

        indices = torch.arange(self.n_samples)
        if self.shuffle:
            g = torch.Generator()
            # Use epoch and seed for a different but deterministic shuffle each epoch
            g.manual_seed(self.config.seed + epoch)
            indices = indices[torch.randperm(self.n_samples, generator=g)]

        indices_for_rank = indices[rank::world_size]
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
        skipped_samples = skip_samples_for_batches(
            skip_batches, self.batch_size, len(indices_for_worker)
        )
        indices_for_worker = indices_for_worker[skipped_samples:]

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_end = i + self.batch_size
            batch_indices = indices_for_worker[i:batch_end]

            yield build_window_batch(
                self.sequences,
                self.config.input_columns,
                self.config.target_columns,
                self.sample_index,
                batch_indices,
            )
