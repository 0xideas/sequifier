"""Shared folder-dataset setup and distributed batch bookkeeping."""

import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.config.metadata import validate_persisted_metadata_versions
from sequifier.helpers import (
    configured_window_stride,
    normalize_path,
    resolve_window_sampling_plan,
    stored_window_layout_from_metadata,
)
from sequifier.io.batch import SequifierBatch
from sequifier.io.folder_shards import FolderChunk, folder_shard_reader
from sequifier.io.images import ImageStore
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
from sequifier.io.static_features import table_for_config
from sequifier.io.window_sampling import build_window_batch, target_valid_from_offsets
from sequifier.typechecking import beartype


class FolderDataset(IterableDataset):
    """Initialize the metadata and state shared by every folder loader."""

    total_batches: int
    target_samples: int

    def __init__(
        self,
        data_path: str,
        config: Any,
        shuffle: bool,
        format_hint: str,
        *,
        metadata=None,
        indexed_view=None,
    ):
        super().__init__()
        # DataLoader spawn workers do not inherit the process group.
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.data_dir = normalize_path(data_path, config.project_root)
        self.config = config
        self.static_table = table_for_config(config)
        self.image_store = None
        if indexed_view is not None:
            self.image_store = indexed_view.image_store(
                config.image_sources,
                config.preconditioning_image,
                project_root=config.project_root,
            )
        elif config.image_sources or config.preconditioning_image is not None:
            self.image_store = ImageStore(
                config.image_sources, config.preconditioning_image, config.project_root
            )
        self.batch_size = config.batch_size
        self.shuffle = shuffle
        self._epoch_state = shared_int(0)
        self._start_batch_state = shared_int(0)

        if metadata is None:
            metadata_path = os.path.join(self.data_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"metadata.json not found in '{self.data_dir}'. "
                    f"Ensure data is pre-processed with {format_hint}."
                )
            with open(metadata_path, "r") as stream:
                metadata = json.load(stream)
        validate_persisted_metadata_versions(metadata)
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

    def _shard_files(self):
        entries = self.metadata["batch_files"]
        if self.file_order == "name":
            entries = sorted(entries, key=lambda entry: entry["path"])
        for entry in entries:
            yield os.path.join(self.data_dir, entry["path"]), entry

    @beartype
    def _calculate_total_batches(self, target_samples: int) -> int:
        num_workers = self.config.runtime_settings.num_workers
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

    n_samples: int
    file_sample_orders: list[tuple[int, SampleOrderPlan]]

    depth_valid_masks: dict[str, Tensor] | None = None
    sequences: dict[str, Tensor]

    def __init__(self, path, config, shuffle=True):
        super().__init__(path, config, shuffle, config.read_format)
        self._initialize_window_layout()
        reader = folder_shard_reader(config, self.metadata, self.sampling_plan)
        columns = set(config.input_columns + config.target_columns)
        sequences = {name: [] for name in columns}
        depth_masks = {}
        sequence_ids, starts, lows, highs, pads = [], [], [], [], []
        self.file_sample_orders = []
        offset = 0
        for path, _ in self._shard_files():
            chunk = reader.read(path)
            index = chunk.sample_index
            positions = logical_sample_positions(chunk.sample_positions, index)
            self.file_sample_orders.append(
                (offset, SampleOrderPlan.build(len(index), positions))
            )
            offset += len(index)
            for name in columns:
                sequences[name].append(chunk.sequences[name])
            for name, mask in (chunk.depth_valid_masks or {}).items():
                depth_masks.setdefault(name, []).append(mask)
            sequence_ids.append(chunk.sequence_ids)
            starts.append(chunk.start_item_positions)
            lows.append(chunk.split_start_item_positions)
            highs.append(chunk.split_end_item_positions)
            pads.append(index.left_pad_lengths)
        if not offset:
            raise ValueError("No usable model windows were found in the dataset.")
        self.sequences = {name: torch.cat(parts) for name, parts in sequences.items()}
        self.depth_valid_masks = {
            name: torch.cat(parts) for name, parts in depth_masks.items()
        }
        self.sequence_ids = torch.cat(sequence_ids)
        self.start_item_positions = torch.cat(starts)
        self.split_start_item_positions = torch.cat(lows)
        self.split_end_item_positions = torch.cat(highs)
        self.left_pad_lengths = torch.cat(pads)
        self.sample_index = self.sampling_plan.build_index(
            self.left_pad_lengths,
            target_valid_from_offsets(
                self.left_pad_lengths,
                self.start_item_positions,
                self.split_start_item_positions,
            ),
        )
        self.n_samples = len(self.sample_index)
        for tensor in self.sequences.values():
            tensor.share_memory_()
        for mask in (self.depth_valid_masks or {}).values():
            mask.share_memory_()
        self.sample_index.share_memory_()

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

    @beartype
    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        indices_for_worker, sample_is_real_for_worker = self._worker_indices()

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]
            batch_sample_is_real = sample_is_real_for_worker[i : i + self.batch_size]

            yield build_window_batch(
                self.sequences,
                self.config.input_columns,
                self.config.target_columns,
                self.sample_index,
                batch_indices,
                batch_sample_is_real,
                depth_valid_masks=self.depth_valid_masks,
                start_item_positions=self.start_item_positions,
                split_start_item_positions=self.split_start_item_positions,
                split_end_item_positions=self.split_end_item_positions,
                sequence_ids=self.sequence_ids,
                static_table=self.static_table,
                image_store=self.image_store,
            )

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


ChunkKey = TypeVar("ChunkKey")


@dataclass(frozen=True)
class ChunkReference(Generic[ChunkKey]):
    key: ChunkKey
    samples: int


class LazyFolderDataset(FolderDataset, Generic[ChunkKey]):
    """Schedule, order, and batch streamed folder chunks."""

    chunks: list[ChunkReference[ChunkKey]]
    chunk_format: str
    ram_chunk_limit: int | None = None

    def _set_chunks(self, chunks: Iterable[ChunkReference[ChunkKey]]) -> None:
        self.chunks = [chunk for chunk in chunks if chunk.samples > 0]
        self.total_samples = sum(chunk.samples for chunk in self.chunks)
        if not self.total_samples:
            raise ValueError("No usable model windows were found in the dataset.")
        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

    def __iter__(self) -> Iterator[SequifierBatch]:
        yield from self._iter_batches(get_worker_info())

    @beartype
    def _get_target_samples(self) -> int:
        return max(
            sum(chunk.samples for chunk in self.chunks[rank :: self.world_size])
            for rank in range(self.world_size)
        )

    def _read_chunk(self, key: ChunkKey, /) -> FolderChunk:
        """Read one format-specific chunk into stored tensors and window indices."""
        raise NotImplementedError

    def _iter_batches(self, worker_info: Any) -> Iterator[SequifierBatch]:
        rank = self.rank
        physical_worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = read_shared_int(self._epoch_state)
        start_batch = read_shared_int(self._start_batch_state)

        num_files = len(self.chunks)
        original_files_for_this_rank = list(range(rank, num_files, self.world_size))
        rank_real_samples = sum(
            self.chunks[i].samples for i in original_files_for_this_rank
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
            current_samples += self.chunks[f_id].samples
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

            file_samples = self.chunks[f_id].samples
            file_start = global_file_start_sample
            file_end = file_start + file_samples
            global_file_start_sample = file_end
            if file_end <= worker_start_sample or file_start >= worker_end_sample:
                continue

            key = self.chunks[f_id].key
            chunk = self._read_chunk(key)
            if self.ram_chunk_limit is not None:
                from sequifier.io.memory_budget import resident_bytes

                actual = resident_bytes(chunk)
                if actual > self.ram_chunk_limit:
                    raise MemoryError(
                        f"Decoded shard {key!r} uses {actual / 1e9:.3f} GB, "
                        f"exceeding its planned {self.ram_chunk_limit / 1e9:.3f} GB "
                        "working allowance under max_ram_gb. Use smaller shards "
                        "or increase the data-loading RAM budget."
                    )
            if len(chunk.sample_index) != file_samples:
                raise RuntimeError(
                    f"Expanded sample count mismatch for {key}: "
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
            if len(worker_indices) == 0:
                del chunk
                continue
            offset = 0
            while offset < len(worker_indices):
                if yielded_samples >= worker_target_samples:
                    break
                # Expanding an entire shard here can dwarf the stored payload,
                # particularly with images, depth or multiple views per row.
                count = min(self.batch_size - buffer_len, len(worker_indices) - offset)
                positions = torch.arange(
                    file_start + worker_file_start_idx + offset,
                    file_start + worker_file_start_idx + offset + count,
                    dtype=torch.int64,
                )
                new_batch = build_window_batch(
                    chunk.sequences,
                    self.config.input_columns,
                    self.config.target_columns,
                    chunk.sample_index,
                    worker_indices[offset : offset + count],
                    positions < rank_real_samples,
                    depth_valid_masks=chunk.depth_valid_masks,
                    start_item_positions=chunk.start_item_positions,
                    split_start_item_positions=chunk.split_start_item_positions,
                    split_end_item_positions=chunk.split_end_item_positions,
                    sequence_ids=chunk.sequence_ids,
                    static_table=self.static_table,
                    image_store=self.image_store,
                )
                if buffer_len == 0:
                    seq_buffer = new_batch.inputs
                    tgt_buffer = new_batch.targets
                    meta_buffer = new_batch.metadata
                else:
                    if set(meta_buffer) != set(new_batch.metadata):
                        raise RuntimeError(
                            f"Inconsistent metadata across {self.chunk_format} chunks."
                        )
                    seq_buffer = {
                        k: torch.cat([seq_buffer[k], new_batch.inputs[k]], dim=0)
                        for k in seq_buffer
                    }
                    tgt_buffer = {
                        k: torch.cat([tgt_buffer[k], new_batch.targets[k]], dim=0)
                        for k in tgt_buffer
                    }
                    meta_buffer = {
                        k: torch.cat([meta_buffer[k], new_batch.metadata[k]], dim=0)
                        for k in meta_buffer
                    }
                offset += count
                buffer_len += count
                if buffer_len == self.batch_size:
                    yield SequifierBatch(seq_buffer, tgt_buffer, meta_buffer)
                    yielded_samples += buffer_len
                    seq_buffer, tgt_buffer, meta_buffer = {}, {}, {}
                    buffer_len = 0
                del new_batch
            # Do not retain the previous chunk while decoding its successor.
            del chunk

        if buffer_len > 0 and yielded_samples < worker_target_samples:
            final_yield_size = min(buffer_len, worker_target_samples - yielded_samples)
            yield SequifierBatch(
                inputs={k: v[:final_yield_size] for k, v in seq_buffer.items()},
                targets={k: v[:final_yield_size] for k, v in tgt_buffer.items()},
                metadata={k: v[:final_yield_size] for k, v in meta_buffer.items()},
            )


class LazyShardDataset(LazyFolderDataset[str]):
    """Resolve file-backed chunk references from a folder manifest."""

    def __init__(self, path, config, shuffle=True):
        super().__init__(path, config, shuffle, config.read_format)
        self._initialize_window_layout()
        self.reader = folder_shard_reader(config, self.metadata, self.sampling_plan)
        self.chunk_format = config.read_format
        self._set_chunks(self._shard_chunks())

    def _read_chunk(self, path: str, /) -> FolderChunk:
        return self.reader.read(path)

    def _shard_chunks(self) -> Iterator[ChunkReference[str]]:
        for path, entry in self._shard_files():
            count = int(entry["samples"])
            if not self.sampling_plan.legacy_single_window:
                histogram = entry.get("target_valid_from_histogram")
                if histogram is None:
                    histogram = {
                        str(k): n
                        for k, n in Counter(self.reader.read_valid_from(path)).items()
                    }
                count = (
                    self.sampling_plan.sample_count_from_target_valid_from_histogram(
                        histogram
                    )
                )
            yield ChunkReference(path, count)
