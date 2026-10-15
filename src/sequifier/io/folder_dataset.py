"""Shared folder-dataset setup and distributed batch bookkeeping."""

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.helpers import (
    WindowSampleIndex,
    configured_window_stride,
    normalize_path,
    resolve_window_sampling_plan,
    stored_window_layout_from_metadata,
)
from sequifier.io.batch import SequifierBatch
from sequifier.io.config import global_training
from sequifier.io.iteration_state import (
    read_shared_int,
    resolve_resume_worker,
    shared_int,
    skip_samples_for_batches,
    write_shared_int,
)
from sequifier.io.sample_order import (
    SampleOrderPlan,
    concatenate_file_orders,
    configured_file_order,
    epoch_file_order,
    logical_sample_positions,
    validate_folder_curriculum,
)
from sequifier.io.window_sampling import build_window_batch
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


@dataclass
class LazyFolderChunk:
    """Format-independent contents of one stored folder chunk."""

    sequences: dict[str, Tensor]
    sample_index: WindowSampleIndex
    sample_positions: Tensor | None
    start_item_positions: Tensor
    split_start_item_positions: Tensor
    split_end_item_positions: Tensor
    depth_valid_masks: dict[str, Tensor] | None = None


class LazyFolderDataset(FolderDataset):
    """Schedule, order, and batch streamed folder chunks."""

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

    def _read_chunk(self, file_path: str) -> LazyFolderChunk:
        """Read one format-specific chunk into stored tensors and window indices."""
        raise NotImplementedError

    def _iter_batches(self, worker_info: Any) -> Iterator[SequifierBatch]:
        rank = self.rank
        physical_worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = read_shared_int(self._epoch_state)
        start_batch = read_shared_int(self._start_batch_state)

        num_files = len(self.batch_files_info)
        original_files_for_this_rank = list(range(rank, num_files, self.world_size))
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
        worker_start_sample = sum(worker_sample_counts[:worker_id])
        worker_target_samples = worker_sample_counts[worker_id]
        worker_end_sample = worker_start_sample + worker_target_samples
        skipped_samples = skip_samples_for_batches(
            skip_batches, self.batch_size, worker_target_samples
        )
        worker_start_sample += skipped_samples
        worker_target_samples -= skipped_samples
        if worker_target_samples <= 0:
            return

        file_order = epoch_file_order(
            len(files_for_this_rank),
            seed=self.config.seed,
            epoch=epoch,
            shuffle=self.shuffle,
            policy=self.file_order,
        )
        ordered_files = [files_for_this_rank[i] for i in file_order]
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
        seq_buffer: dict[str, Tensor] = {}
        tgt_buffer: dict[str, Tensor] = {}
        meta_buffer: dict[str, Tensor] = {}
        buffer_len = 0

        for f_id in extended_files:
            if yielded_samples >= worker_target_samples:
                break

            file_samples = self.batch_files_info[f_id]["samples"]
            file_start = global_file_start_sample
            file_end = file_start + file_samples
            global_file_start_sample = file_end
            if file_end <= worker_start_sample or file_start >= worker_end_sample:
                continue

            file_path = os.path.join(self.data_dir, self.batch_files_info[f_id]["path"])
            chunk = self._read_chunk(file_path)
            if len(chunk.sample_index) != file_samples:
                raise RuntimeError(
                    f"Expanded sample count mismatch for {file_path}: "
                    f"metadata={file_samples}, loaded={len(chunk.sample_index)}."
                )
            indices = SampleOrderPlan.build(
                file_samples,
                logical_sample_positions(chunk.sample_positions, chunk.sample_index),
            ).indices_for_epoch(
                seed=self.config.seed + f_id + rank,
                epoch=epoch,
                shuffle=self.shuffle,
            )
            worker_file_start_idx = max(0, worker_start_sample - file_start)
            worker_file_end_idx = min(file_samples, worker_end_sample - file_start)
            worker_indices = indices[worker_file_start_idx:worker_file_end_idx]
            logical_positions = torch.arange(
                file_start + worker_file_start_idx,
                file_start + worker_file_end_idx,
                dtype=torch.int64,
            )
            # Repeated file cycles are padding even when their source rows are real.
            sample_is_real = logical_positions < rank_real_samples
            if len(worker_indices) == 0:
                continue

            new_batch = build_window_batch(
                chunk.sequences,
                self.config.input_columns,
                self.config.target_columns,
                chunk.sample_index,
                worker_indices,
                sample_is_real,
                depth_valid_masks=chunk.depth_valid_masks,
                start_item_positions=chunk.start_item_positions,
                split_start_item_positions=chunk.split_start_item_positions,
                split_end_item_positions=chunk.split_end_item_positions,
            )
            new_seq = new_batch.inputs
            new_tgt = new_batch.targets
            new_meta = new_batch.metadata
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
                        "Inconsistent leftPadLength metadata across "
                        f"{self.chunk_format} chunks."
                    )
                meta_buffer = {
                    k: torch.cat([meta_buffer[k], new_meta[k]], dim=0)
                    for k in meta_buffer
                }
            buffer_len += len(worker_indices)

            while buffer_len >= self.batch_size:
                if yielded_samples >= worker_target_samples:
                    break
                yield SequifierBatch(
                    inputs={k: v[: self.batch_size] for k, v in seq_buffer.items()},
                    targets={k: v[: self.batch_size] for k, v in tgt_buffer.items()},
                    metadata={k: v[: self.batch_size] for k, v in meta_buffer.items()},
                )
                yielded_samples += self.batch_size
                seq_buffer = {k: v[self.batch_size :] for k, v in seq_buffer.items()}
                tgt_buffer = {k: v[self.batch_size :] for k, v in tgt_buffer.items()}
                meta_buffer = {k: v[self.batch_size :] for k, v in meta_buffer.items()}
                buffer_len -= self.batch_size

        if buffer_len > 0 and yielded_samples < worker_target_samples:
            final_yield_size = min(buffer_len, worker_target_samples - yielded_samples)
            yield SequifierBatch(
                inputs={k: v[:final_yield_size] for k, v in seq_buffer.items()},
                targets={k: v[:final_yield_size] for k, v in tgt_buffer.items()},
                metadata={k: v[:final_yield_size] for k, v in meta_buffer.items()},
            )
