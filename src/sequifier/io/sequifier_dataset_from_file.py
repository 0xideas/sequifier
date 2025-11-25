from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, numpy_to_pytorch, read_data


class SequifierDatasetFromFile(IterableDataset):
    """
    An iterable-style dataset that pre-loads all data into CPU RAM and yields
    pre-collated batches.

    This is the idiomatic PyTorch solution for implementing custom 'en block'
    batching. The __iter__ method handles shuffling and batch slicing, ensuring
    maximum performance.
    """

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self.epoch = 0

        # Create a unified list of all columns the model might need
        all_columns = sorted(list(set(config.input_columns + config.target_columns)))

        logger.info(
            f"[INFO] Loading training dataset into memory from '{data_path}'..."
        )
        data_df = read_data(data_path, config.read_format)

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        # self.all_tensors now holds both inputs and targets
        all_tensors = numpy_to_pytorch(
            data=data_df,
            column_types=column_types,
            all_columns=all_columns,
            seq_length=config.seq_length,
        )
        self.n_samples = all_tensors[all_columns[0]].shape[0]

        del data_df

        self.sequence_tensors = {
            key: all_tensors[key] for key in self.config.input_columns
        }
        self.target_tensors = {
            key: all_tensors[f"{key}_target"] for key in self.config.target_columns
        }
        del all_tensors

        if config.training_spec.device.startswith("cuda"):
            for key in self.sequence_tensors:
                self.sequence_tensors[key] = self.sequence_tensors[key].pin_memory()
            for key in self.target_tensors:
                self.target_tensors[key] = self.target_tensors[key].pin_memory()

        logger.info(f"[INFO] Dataset loaded with {self.n_samples} samples.")

    def set_epoch(self, epoch: int):
        """Allows the training loop to set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], None, None, None]
    ]:
        """Yields batches of data.

        Handles shuffling (if enabled) and slicing data based on distributed
        rank and worker ID.

        Yields:
            Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], None, None, None]]:
            An iterator where each item is a tuple containing:
                - data_batch (dict): Dictionary of feature tensors for the batch.
                - targets_batch (dict): Dictionary of target tensors for the batch.
                - None: Placeholder for sequence_id (not used in this dataset type).
                - None: Placeholder for subsequence_id (not used in this dataset type).
                - None: Placeholder for start_position (not used in this dataset type).
        """
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

            yield data_batch, targets_batch, None, None, None
