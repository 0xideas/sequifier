import math
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    numpy_to_pytorch,
    read_data,
    resolve_window_view,
)
from sequifier.io.batch import SequifierBatch


class SequifierDatasetFromFile(IterableDataset):
    """Eager single-file dataset yielding pre-collated batches."""

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self.epoch = 0

        all_columns = sorted(list(set(config.input_columns + config.target_columns)))

        logger.info(
            f"[INFO] Loading training dataset into memory from '{data_path}'..."
        )
        data_df = read_data(data_path, config.read_format)

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        resolved_view = resolve_window_view(config.storage_layout, config.window_view)
        all_tensors, metadata_tensors = numpy_to_pytorch(
            data=data_df,
            column_types=column_types,
            all_columns=all_columns,
            resolved_view=resolved_view,
        )
        self.n_samples = all_tensors[all_columns[0]].shape[0]

        del data_df

        self.sequence_tensors = {
            key: all_tensors[key] for key in self.config.input_columns
        }
        self.target_tensors = {
            key: all_tensors[f"{key}_target"] for key in self.config.target_columns
        }
        self.metadata_tensors = metadata_tensors
        del all_tensors

        if config.training_spec.device.startswith("cuda"):
            for key in self.sequence_tensors:
                self.sequence_tensors[key] = self.sequence_tensors[key].pin_memory()
            for key in self.target_tensors:
                self.target_tensors[key] = self.target_tensors[key].pin_memory()
            for key in self.metadata_tensors:
                self.metadata_tensors[key] = self.metadata_tensors[key].pin_memory()

        logger.info(f"[INFO] Dataset loaded with {self.n_samples} samples.")

    def set_epoch(self, epoch: int):
        """Set the shuffle epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        worker_info = torch.utils.data.get_worker_info()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        indices = torch.arange(self.n_samples)
        if self.shuffle:
            g = torch.Generator()
            # Use epoch and seed for a different but deterministic shuffle each epoch
            g.manual_seed(self.config.seed + self.epoch)
            indices = indices[torch.randperm(self.n_samples, generator=g)]

        indices_for_rank = indices[rank::world_size]
        indices_for_worker = indices_for_rank[worker_id::num_workers]

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_end = i + self.batch_size
            if batch_end > len(indices_for_worker):
                continue

            batch_indices = indices_for_worker[i:batch_end]

            data_batch = {
                key: tensor[batch_indices]
                for key, tensor in self.sequence_tensors.items()
            }
            targets_batch = {
                key: tensor[batch_indices]
                for key, tensor in self.target_tensors.items()
            }
            metadata_batch = {
                key: tensor[batch_indices]
                for key, tensor in self.metadata_tensors.items()
            }

            yield SequifierBatch(
                inputs=data_batch,
                targets=targets_batch,
                metadata=metadata_batch,
            )
