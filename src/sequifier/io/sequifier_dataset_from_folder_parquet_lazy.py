import json
import math
import os
from collections import Counter
from typing import Dict, Iterator

import polars as pl
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.train_config import TrainModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    columns_from_slice,
    configured_model_window_stride,
    get_left_pad_lengths_from_preprocessed_data,
    normalize_path,
    resolve_window_sampling_plan,
    stored_window_layout_from_metadata,
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


class SequifierDatasetFromFolderParquetLazy(IterableDataset):
    """Streams long-format Parquet chunks into rank/worker-aligned batches."""

    def __init__(self, data_path: str, config: TrainModel, shuffle: bool = True):
        super().__init__()
        self.data_dir = normalize_path(data_path, config.project_root)
        self.config = config
        self.batch_size = config.training_spec.batch_size
        self.shuffle = shuffle
        self._epoch_state = shared_int(0)
        self._start_batch_state = shared_int(0)

        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Ensure data is pre-processed with merge_output: False."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.folder_layout = stored_window_layout_from_metadata(metadata)
        self.sampling_plan = resolve_window_sampling_plan(
            self.folder_layout,
            config.window_view,
            configured_model_window_stride(config),
        )

        self.batch_files_info = []
        for raw_file_info in metadata["batch_files"]:
            file_info = dict(raw_file_info)
            file_info["stored_samples"] = int(raw_file_info["samples"])
            histogram = raw_file_info.get("left_pad_length_histogram")
            if histogram is None and not self.sampling_plan.legacy_single_window:
                file_path = os.path.join(self.data_dir, file_info["path"])
                padding_rows = (
                    pl.scan_parquet(file_path)
                    .group_by(["sequenceId", "subsequenceId"])
                    .agg(pl.col("leftPadLength").first())
                    .select("leftPadLength")
                    .collect()
                    .get_column("leftPadLength")
                    .to_list()
                )
                histogram = {
                    str(value): count for value, count in Counter(padding_rows).items()
                }
            if self.sampling_plan.legacy_single_window:
                file_info["samples"] = file_info["stored_samples"]
            else:
                assert histogram is not None
                file_info["samples"] = self.sampling_plan.sample_count_from_histogram(
                    histogram
                )
            if file_info["samples"] > 0:
                self.batch_files_info.append(file_info)
        self.total_samples = sum(info["samples"] for info in self.batch_files_info)
        if self.total_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")

        self.column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)
        logger.info(
            f"[INFO] Lazy Parquet Dataset mapped with {self.target_samples} samples and {self.total_batches} batches."
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
        write_shared_int(self._epoch_state, epoch)

    def set_start_batch(self, start_batch: int):
        """Set the first global batch to yield on the next iteration."""
        write_shared_int(self._start_batch_state, start_batch)

    def _get_target_samples(self) -> int:
        """Return the padded per-rank sample count for aligned distributed steps."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        num_files = len(self.batch_files_info)

        samples_per_rank = []
        for r in range(world_size):
            f_r = list(range(r, num_files, world_size))
            samples_per_rank.append(
                sum(self.batch_files_info[i]["samples"] for i in f_r) if f_r else 0
            )

        return max(samples_per_rank)

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = read_shared_int(self._epoch_state)
        start_batch = read_shared_int(self._start_batch_state)

        num_files = len(self.batch_files_info)
        original_files_for_this_rank = list(range(rank, num_files, world_size))
        rank_real_samples = sum(
            self.batch_files_info[i]["samples"] for i in original_files_for_this_rank
        )
        files_for_this_rank = original_files_for_this_rank.copy()

        if not files_for_this_rank:
            if self.target_samples == 0:
                return
            files_for_this_rank = [rank % num_files]

        base_samples_per_worker = self.target_samples // num_workers
        remainder = self.target_samples % num_workers
        worker_sample_counts = [
            base_samples_per_worker + (1 if i < remainder else 0)
            for i in range(num_workers)
        ]
        worker_batch_counts = [
            math.ceil(sample_count / self.batch_size)
            for sample_count in worker_sample_counts
        ]
        worker_id, skip_batches = resolve_resume_worker(
            start_batch,
            physical_worker_id,
            num_workers,
            worker_batch_counts,
        )

        worker_start_sample = 0
        for i in range(worker_id):
            worker_start_sample += worker_sample_counts[i]

        worker_target_samples = worker_sample_counts[worker_id]
        worker_end_sample = worker_start_sample + worker_target_samples
        skipped_samples = skip_samples_for_batches(
            skip_batches, self.batch_size, worker_target_samples
        )
        worker_start_sample += skipped_samples
        worker_target_samples -= skipped_samples
        if worker_target_samples <= 0:
            return

        g = torch.Generator()
        g.manual_seed(self.config.seed + epoch)

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
        global_file_start_sample = 0

        sequence_columns = columns_from_slice(
            slice(0, self.folder_layout.stored_context_width),
            self.folder_layout.stored_context_width,
        )

        seq_buffer: Dict[str, torch.Tensor] = {}
        tgt_buffer: Dict[str, torch.Tensor] = {}
        meta_buffer: Dict[str, torch.Tensor] = {}
        buffer_len = 0

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
            df = pl.read_parquet(file_path)
            left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(df)
            sample_index = self.sampling_plan.build_index(left_pad_lengths)
            if len(sample_index) != file_samples:
                raise RuntimeError(
                    f"Expanded sample count mismatch for {file_path}: "
                    f"metadata={file_samples}, loaded={len(sample_index)}."
                )

            indices = torch.arange(file_samples)
            if self.shuffle:
                g_file = torch.Generator()
                g_file.manual_seed(self.config.seed + epoch + f_id + rank)
                indices = indices[torch.randperm(file_samples, generator=g_file)]

            worker_file_start_idx = max(0, worker_start_sample - file_start)
            worker_file_end_idx = min(file_samples, worker_end_sample - file_start)

            worker_indices = indices[worker_file_start_idx:worker_file_end_idx]
            logical_positions = torch.arange(
                file_start + worker_file_start_idx,
                file_start + worker_file_end_idx,
                dtype=torch.int64,
            )
            sample_is_real = logical_positions < rank_real_samples

            num_new_samples = len(worker_indices)

            if num_new_samples == 0:
                del df
                continue

            feature_partitions = {
                frame.item(0, "inputCol"): frame
                for frame in df.partition_by("inputCol")
            }

            stored_sequences = {}
            for col_name in set(self.config.input_columns + self.config.target_columns):
                if col_name in feature_partitions:
                    stored_sequences[col_name] = torch.tensor(
                        feature_partitions[col_name]
                        .sort(["sequenceId", "subsequenceId"])
                        .select(sequence_columns)
                        .to_numpy(),
                        dtype=self.column_torch_types[col_name],
                    )
                else:
                    raise ValueError(f"Column not found in input data: {col_name}")

            new_batch = build_window_batch(
                stored_sequences,
                self.config.input_columns,
                self.config.target_columns,
                sample_index,
                worker_indices,
                sample_is_real,
            )
            new_seq = new_batch.inputs
            new_tgt = new_batch.targets
            new_meta = new_batch.metadata

            del df

            if buffer_len == 0:
                seq_buffer = new_seq
                tgt_buffer = new_tgt
                meta_buffer = new_meta
            else:
                seq_buffer = {
                    k: torch.cat([seq_buffer[k], new_seq[k]], dim=0) for k in seq_buffer
                }
                tgt_buffer = {
                    k: torch.cat([tgt_buffer[k], new_tgt[k]], dim=0) for k in tgt_buffer
                }
                if set(meta_buffer) != set(new_meta):
                    raise RuntimeError(
                        "Inconsistent leftPadLength metadata across Parquet chunks."
                    )
                meta_buffer = {
                    k: torch.cat([meta_buffer[k], new_meta[k]], dim=0)
                    for k in meta_buffer
                }

            buffer_len += num_new_samples

            while buffer_len >= self.batch_size:
                if yielded_samples >= worker_target_samples:
                    break

                batch_seq = {k: v[: self.batch_size] for k, v in seq_buffer.items()}
                batch_tgt = {k: v[: self.batch_size] for k, v in tgt_buffer.items()}
                batch_meta = {k: v[: self.batch_size] for k, v in meta_buffer.items()}

                yield SequifierBatch(
                    inputs=batch_seq,
                    targets=batch_tgt,
                    metadata=batch_meta,
                )
                yielded_samples += self.batch_size

                seq_buffer = {k: v[self.batch_size :] for k, v in seq_buffer.items()}
                tgt_buffer = {k: v[self.batch_size :] for k, v in tgt_buffer.items()}
                meta_buffer = {k: v[self.batch_size :] for k, v in meta_buffer.items()}
                buffer_len -= self.batch_size

        if buffer_len > 0 and yielded_samples < worker_target_samples:
            remaining_needed = worker_target_samples - yielded_samples
            final_yield_size = min(buffer_len, remaining_needed)

            batch_seq = {k: v[:final_yield_size] for k, v in seq_buffer.items()}
            batch_tgt = {k: v[:final_yield_size] for k, v in tgt_buffer.items()}
            batch_meta = {k: v[:final_yield_size] for k, v in meta_buffer.items()}

            yield SequifierBatch(
                inputs=batch_seq,
                targets=batch_tgt,
                metadata=batch_meta,
            )
