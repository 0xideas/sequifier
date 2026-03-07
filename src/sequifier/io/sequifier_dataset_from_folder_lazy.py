import json
import math
import os
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import normalize_path


class SequifierDatasetFromFolderLazy(IterableDataset):
    """
    An efficient PyTorch IterableDataset for datasets that do not fit into RAM.

    Instead of loading a file to cache and slicing individual rows, it reads
    whole files sequentially, shuffles the indices, and yields full batches.
    This completely eliminates the CPU cloning bottleneck.
    """

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.data_dir = normalize_path(data_path, config.project_root)
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self.epoch = 0

        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Ensure data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.batch_files_info = metadata["batch_files"]
        self.sampling_strategy = config.training_spec.sampling_strategy

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)
        logger.info(
            f"[INFO] Lazy Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
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
        """Allows the training loop to set the epoch for deterministic file shuffling."""
        self.epoch = epoch

    def _get_target_samples(self) -> int:
        """Calculates exact sample count per rank to ensure FSDP syncs properly."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        num_files = len(self.batch_files_info)

        samples_per_rank = []
        for r in range(world_size):
            f_r = list(range(r, num_files, world_size))
            samples_per_rank.append(
                sum(self.batch_files_info[i]["samples"] for i in f_r) if f_r else 0
            )

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
    ) -> Iterator[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], None, None, None]
    ]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # 1. Distribute files among ranks
        num_files = len(self.batch_files_info)
        files_for_this_rank = list(range(rank, num_files, world_size))

        if not files_for_this_rank:
            if self.sampling_strategy == "oversampling":
                files_for_this_rank = [rank % num_files]
            else:
                raise Exception(f"No file found for GPU rank {rank}.")

        # 2. Assign exact sample quotas to this specific DataLoader worker thread
        base_samples_per_worker = self.target_samples // num_workers
        remainder = self.target_samples % num_workers
        worker_target_samples = base_samples_per_worker + (
            1 if worker_id < remainder else 0
        )

        # 3. Shuffle files deterministically
        g = torch.Generator()
        g.manual_seed(self.config.seed + self.epoch)

        if self.shuffle:
            file_order = torch.randperm(len(files_for_this_rank), generator=g).tolist()
            ordered_files = [files_for_this_rank[i] for i in file_order]
        else:
            ordered_files = files_for_this_rank.copy()

        # 4. Extend files if using oversampling
        extended_files = []
        current_samples = 0
        file_idx = 0
        while current_samples < self.target_samples:
            f_id = ordered_files[file_idx % len(ordered_files)]
            extended_files.append(f_id)
            current_samples += self.batch_files_info[f_id]["samples"]
            file_idx += 1

        # 5. Distribute assigned files among workers
        worker_files = extended_files[worker_id::num_workers]

        yielded_samples = 0
        train_seq_len = self.config.seq_length

        # 6. Stream data
        for f_id in worker_files:
            file_path = os.path.join(self.data_dir, self.batch_files_info[f_id]["path"])

            # Load file to CPU
            (sequences_batch, targets_batch, _, _, _) = torch.load(
                file_path, map_location="cpu", weights_only=False
            )

            file_samples = sequences_batch[list(sequences_batch.keys())[0]].shape[0]
            indices = torch.arange(file_samples)

            if self.shuffle:
                g_file = torch.Generator()
                g_file.manual_seed(self.config.seed + self.epoch + f_id + rank)
                indices = indices[torch.randperm(file_samples, generator=g_file)]

            # Slice and yield full batches natively
            for i in range(0, file_samples, self.batch_size):
                if yielded_samples >= worker_target_samples:
                    break

                batch_indices = indices[i : i + self.batch_size]

                # Trim batch if it pushes us over the exact target quota
                if yielded_samples + len(batch_indices) > worker_target_samples:
                    batch_indices = batch_indices[
                        : worker_target_samples - yielded_samples
                    ]

                # FAST NATIVE SLICING
                seq_dict = {
                    k: v[batch_indices, -train_seq_len:]
                    for k, v in sequences_batch.items()
                }
                tgt_dict = {
                    k: v[batch_indices, -train_seq_len:]
                    for k, v in targets_batch.items()
                }

                yield seq_dict, tgt_dict, None, None, None
                yielded_samples += len(batch_indices)

            # Drop the dictionaries so Python's Garbage Collector frees the memory immediately
            del sequences_batch, targets_batch

            if yielded_samples >= worker_target_samples:
                break
