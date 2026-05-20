import json
import math
import os
from typing import Dict, Iterator, Tuple

import polars as pl
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, normalize_path


class SequifierDatasetFromFolderParquetLazy(IterableDataset):
    """
    An efficient, memory-safe PyTorch IterableDataset for out-of-core training
    that streams chunked Parquet files from a directory using metadata.json boundaries.
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
            raise FileNotFoundError(f"metadata.json not found in '{self.data_dir}'.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.batch_files_info = metadata["batch_files"]
        self.total_samples = metadata["total_samples"]
        self.sampling_strategy = config.training_spec.sampling_strategy

        # Re-use your cross-GPU sync arithmetic
        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

        self.column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

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
        self.epoch = epoch

    def _get_target_samples(self) -> int:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_files = len(self.batch_files_info)
        samples_per_rank = []
        for r in range(world_size):
            f_r = list(range(r, num_files, world_size))
            samples_per_rank.append(
                sum(self.batch_files_info[i]["samples"] for i in f_r) if f_r else 0
            )

        if self.sampling_strategy == "exact":
            return int(samples_per_rank[0])
        elif self.sampling_strategy == "oversampling":
            return max(samples_per_rank)
        else:
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

        num_files = len(self.batch_files_info)
        files_for_this_rank = list(range(rank, num_files, world_size))

        if not files_for_this_rank and self.sampling_strategy == "oversampling":
            files_for_this_rank = [rank % num_files]

        base_samples_per_worker = self.target_samples // num_workers
        remainder = self.target_samples % num_workers

        worker_start_sample = sum(
            base_samples_per_worker + (1 if i < remainder else 0)
            for i in range(worker_id)
        )
        worker_target_samples = base_samples_per_worker + (
            1 if worker_id < remainder else 0
        )
        worker_end_sample = worker_start_sample + worker_target_samples

        g = torch.Generator()
        g.manual_seed(self.config.seed + self.epoch)
        if self.shuffle:
            file_order = torch.randperm(len(files_for_this_rank), generator=g).tolist()
            ordered_files = [files_for_this_rank[i] for i in file_order]
        else:
            ordered_files = files_for_this_rank.copy()

        extended_files = []
        current_samples = 0
        file_idx = 0
        while current_samples < self.target_samples:
            f_id = ordered_files[file_idx % len(ordered_files)]
            extended_files.append(f_id)
            current_samples += self.batch_files_info[f_id]["samples"]
            file_idx += 1

        yielded_samples = 0
        train_seq_len = self.config.seq_length
        global_file_start_sample = 0

        seq_buffer, tgt_buffer = {}, {}
        buffer_len = 0

        # Sequence formatting configurations mimicking numpy_to_pytorch logic
        input_seq_cols = [str(c) for c in range(train_seq_len, 0, -1)]
        target_seq_cols = [str(c) for c in range(train_seq_len - 1, -1, -1)]

        for f_id in extended_files:
            if yielded_samples >= worker_target_samples:
                break

            file_samples = self.batch_files_info[f_id]["samples"]
            file_start = global_file_start_sample
            file_end = global_file_start_sample + file_samples
            global_file_start_sample += file_samples

            if file_end <= worker_start_sample or file_start >= worker_end_sample:
                continue

            file_path = os.path.join(self.data_dir, self.batch_files_info[f_id]["path"])

            # Read Long format Parquet into Polars
            df = pl.read_parquet(file_path)
            feature_names = df["inputCol"].unique().to_list()

            # Slice the sequence IDs matching this worker's chunk boundaries
            worker_file_start_idx = max(0, worker_start_sample - file_start)
            worker_file_end_idx = min(file_samples, worker_end_sample - file_start)
            num_new_samples = worker_file_end_idx - worker_file_start_idx

            if num_new_samples <= 0:
                continue

            # Process Long format data structures into PyTorch Tensors
            new_seq, new_tgt = {}, {}
            for col_name in feature_names:
                feature_df = df.filter(pl.col("inputCol") == col_name)

                # Extract chunk rows matching worker constraints
                feature_chunk = feature_df.slice(worker_file_start_idx, num_new_samples)

                torch_type = self.column_torch_types[col_name]

                new_seq[col_name] = torch.tensor(
                    feature_chunk.select(input_seq_cols).to_numpy(), dtype=torch_type
                )
                new_tgt[col_name] = torch.tensor(
                    feature_chunk.select(target_seq_cols).to_numpy(), dtype=torch_type
                )

            del df

            if buffer_len == 0:
                seq_buffer, tgt_buffer = new_seq, new_tgt
            else:
                seq_buffer = {
                    k: torch.cat([seq_buffer[k], new_seq[k]], dim=0) for k in seq_buffer
                }
                tgt_buffer = {
                    k: torch.cat([tgt_buffer[k], new_tgt[k]], dim=0) for k in tgt_buffer
                }

            buffer_len += num_new_samples

            while buffer_len >= self.batch_size:
                if yielded_samples >= worker_target_samples:
                    break

                batch_seq = {k: v[: self.batch_size] for k, v in seq_buffer.items()}
                batch_tgt = {k: v[: self.batch_size] for k, v in tgt_buffer.items()}

                yield batch_seq, batch_tgt, None, None, None
                yielded_samples += self.batch_size

                seq_buffer = {k: v[self.batch_size :] for k, v in seq_buffer.items()}
                tgt_buffer = {k: v[self.batch_size :] for k, v in tgt_buffer.items()}
                buffer_len -= self.batch_size

        if buffer_len > 0 and yielded_samples < worker_target_samples:
            remaining_needed = worker_target_samples - yielded_samples
            final_yield_size = min(buffer_len, remaining_needed)

            batch_seq = {k: v[:final_yield_size] for k, v in seq_buffer.items()}
            batch_tgt = {k: v[:final_yield_size] for k, v in tgt_buffer.items()}

            yield batch_seq, batch_tgt, None, None, None
