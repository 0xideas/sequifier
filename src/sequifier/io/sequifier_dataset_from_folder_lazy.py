import json
import math
import os
from typing import Dict, Iterator, Tuple

import numpy as np
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
        self.total_samples = metadata["total_samples"]
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

        num_files = len(self.batch_files_info)

        samples_per_rank = []
        for r in range(world_size):
            f_r = list(range(r, num_files, world_size))
            samples_per_rank.append(
                sum(self.batch_files_info[i]["samples"] for i in f_r) if f_r else 0
            )

        if self.sampling_strategy == "exact":
            samples_per_rank = np.array(samples_per_rank)
            unique_samples_per_rank, counts = np.unique(
                samples_per_rank, return_counts=True
            )
            if len(unique_samples_per_rank) > 1:
                if np.max(counts) / np.sum(counts) > 0.8:
                    most_frequent_unique_samples_val = unique_samples_per_rank[
                        np.argmax(counts)
                    ]
                    non_max_idx = np.where(
                        samples_per_rank != most_frequent_unique_samples_val
                    )[0]
                    files_strings = []
                    for i in non_max_idx:
                        f_r = list(range(i, num_files, world_size))
                        files_strings.append(
                            "\n\t".join(
                                [
                                    f'{self.batch_files_info[j]["path"].split(os.sep)[-1]}: {self.batch_files_info[j]["samples"]}'
                                    for j in f_r
                                ]
                            )
                        )
                    rank_details = [
                        f"Rank {i}: {samples_per_rank[i]} samples, files:\n\t{files_strings[i]}"
                        for i in non_max_idx
                    ]
                    exception_detail = f":\nMost frequent sample value: {most_frequent_unique_samples_val}\n{'\n'.join(rank_details)}"
                else:
                    exception_detail = ""

                raise Exception(
                    f"Found {len(unique_samples_per_rank)} different number of samples per rank/GPU: {unique_samples_per_rank}{exception_detail}"
                )
            return int(unique_samples_per_rank[0])

        elif self.sampling_strategy == "oversampling":
            return max(samples_per_rank)
        else:
            assert self.sampling_strategy == "undersampling"
            return min(samples_per_rank)

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

        # 2. Assign exact sample quotas and boundaries to this specific worker thread
        base_samples_per_worker = self.target_samples // num_workers
        remainder = self.target_samples % num_workers

        # Calculate exactly where this worker's data starts and ends in the global stream
        worker_start_sample = 0
        for i in range(worker_id):
            worker_start_sample += base_samples_per_worker + (1 if i < remainder else 0)

        worker_target_samples = base_samples_per_worker + (
            1 if worker_id < remainder else 0
        )
        worker_end_sample = worker_start_sample + worker_target_samples

        # 3. Shuffle files deterministically
        g = torch.Generator()
        g.manual_seed(self.config.seed + self.epoch)

        if self.shuffle:
            file_order = torch.randperm(len(files_for_this_rank), generator=g).tolist()
            ordered_files = [files_for_this_rank[i] for i in file_order]
        else:
            ordered_files = files_for_this_rank.copy()

        # 4. Extend files based on exact target requirements
        # This naturally only loops if target_samples demands it (e.g., oversampling)
        extended_files = []
        current_samples = 0
        file_idx = 0
        while current_samples < self.target_samples:
            f_id = ordered_files[file_idx % len(ordered_files)]
            extended_files.append(f_id)
            current_samples += self.batch_files_info[f_id]["samples"]
            file_idx += 1

        # 5. Stream data using precise global boundaries
        yielded_samples = 0
        train_seq_len = self.config.seq_length
        global_file_start_sample = 0

        for f_id in extended_files:
            if yielded_samples >= worker_target_samples:
                break

            file_samples = self.batch_files_info[f_id]["samples"]
            file_start = global_file_start_sample
            file_end = global_file_start_sample + file_samples
            global_file_start_sample += file_samples

            # Skip this file if it belongs entirely to other workers
            if file_end <= worker_start_sample or file_start >= worker_end_sample:
                continue

            # This file overlaps with our worker's assigned boundary. Load it.
            file_path = os.path.join(self.data_dir, self.batch_files_info[f_id]["path"])
            (sequences_batch, targets_batch, _, _, _) = torch.load(
                file_path, map_location="cpu", weights_only=False
            )

            # Generate indices for the whole file
            indices = torch.arange(file_samples)
            if self.shuffle:
                g_file = torch.Generator()
                # Use identical seed across workers so the file is shuffled exactly the same way
                g_file.manual_seed(self.config.seed + self.epoch + f_id + rank)
                indices = indices[torch.randperm(file_samples, generator=g_file)]

            # Slice the indices to extract ONLY the portion belonging to this worker
            worker_file_start_idx = max(0, worker_start_sample - file_start)
            worker_file_end_idx = min(file_samples, worker_end_sample - file_start)
            worker_indices = indices[worker_file_start_idx:worker_file_end_idx]

            # Yield batches from the worker's specific slice
            for i in range(0, len(worker_indices), self.batch_size):
                batch_indices = worker_indices[i : i + self.batch_size]

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

            del sequences_batch, targets_batch
