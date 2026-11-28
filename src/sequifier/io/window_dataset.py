"""Stored windows and worker-aware batching shared by file and folder datasets."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Any, Generic, Protocol, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.helpers import WindowSampleIndex, validate_stored_window_width
from sequifier.io.batch import SequifierBatch
from sequifier.io.images import ImageStore
from sequifier.io.indexed_sequences import IndexedImageStore
from sequifier.io.iteration_state import (
    read_shared_int,
    resolve_resume_worker,
    shared_int,
    skip_samples_for_batches,
    write_shared_int,
)
from sequifier.io.sample_order import (
    SampleOrderPlan,
    configured_file_order,
    curriculum_sample_positions,
    epoch_file_order,
    logical_sample_positions,
)
from sequifier.io.static_features import table_for_config
from sequifier.io.window_sampling import (
    build_window_batch,
    target_valid_from_offsets,
    temporal_columns,
    validate_split_bounds_available,
)


@dataclass
class WindowChunk:
    """Format-independent contents of one or more stored windows."""

    sequences: dict[str, Tensor]
    sequence_ids: Tensor
    sample_index: WindowSampleIndex
    sample_positions: Tensor | None
    start_item_positions: Tensor
    split_start_item_positions: Tensor
    split_end_item_positions: Tensor
    depth_valid_masks: dict[str, Tensor]

    @classmethod
    def from_payload(cls, payload, config, sampling_plan, file_path):
        validate_split_bounds_available(
            payload.start_item_positions,
            payload.split_start_item_positions,
            payload.split_end_item_positions,
            file_path,
        )
        for tensor in payload.sequences.values():
            validate_stored_window_width(
                tensor, sampling_plan.resolved_view.storage.window_length
            )
        sample_index = sampling_plan.build_index(
            payload.left_pad_lengths,
            target_valid_from_offsets(
                payload.left_pad_lengths,
                payload.start_item_positions,
                payload.split_start_item_positions,
            ),
        )
        return cls(
            sequences={
                name: payload.sequences[name] for name in temporal_columns(config)
            },
            sequence_ids=payload.sequence_ids,
            sample_index=sample_index,
            sample_positions=curriculum_sample_positions(
                config,
                payload.sample_positions,
                file_path,
                payload.curriculum_columns,
            ),
            depth_valid_masks={
                name: payload.depth_valid_masks[name]
                for name in config.depth_layouts.root
            },
            start_item_positions=payload.start_item_positions,
            split_start_item_positions=payload.split_start_item_positions,
            split_end_item_positions=payload.split_end_item_positions,
        )

    def sample_order(self):
        return SampleOrderPlan.build(
            len(self.sample_index),
            logical_sample_positions(self.sample_positions, self.sample_index),
        )


class WindowDataset(IterableDataset):
    target_samples: int

    def __init__(
        self,
        config,
        shuffle,
        *,
        image_store_factory: Callable[
            ..., ImageStore | IndexedImageStore | None
        ] = ImageStore,
    ):
        super().__init__()
        # DataLoader spawn workers do not inherit the process group.
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.config = config
        self.static_table = table_for_config(config)
        self.image_store = None
        if config.image_sources or config.preconditioning_image is not None:
            self.image_store = image_store_factory(
                config.image_sources,
                config.preconditioning_image,
                project_root=config.project_root,
            )
        self.batch_size = config.batch_size
        self.shuffle = shuffle
        self._epoch_state = shared_int(0)
        self._start_batch_state = shared_int(0)

    def set_epoch(self, epoch):
        write_shared_int(self._epoch_state, epoch)

    def set_start_batch(self, start_batch):
        write_shared_int(self._start_batch_state, start_batch)

    @property
    def epoch(self):
        return read_shared_int(self._epoch_state)

    def _worker_batch_counts(self, workers):
        return [
            ceil(len(range(i, self.target_samples, workers)) / self.batch_size)
            for i in range(workers)
        ]

    def _resume_worker(self, info, workers):
        return resolve_resume_worker(
            read_shared_int(self._start_batch_state),
            info.id if info else 0,
            workers,
            self._worker_batch_counts(workers),
        )

    def __len__(self):
        return sum(
            self._worker_batch_counts(max(1, self.config.data_loading.num_workers))
        )

    def _window_batch(self, chunk, indices, real=None):
        return build_window_batch(
            chunk.sequences,
            self.config.input_columns,
            self.config.target_columns,
            chunk.sample_index,
            indices,
            real,
            depth_valid_masks=chunk.depth_valid_masks,
            start_item_positions=chunk.start_item_positions,
            split_start_item_positions=chunk.split_start_item_positions,
            split_end_item_positions=chunk.split_end_item_positions,
            sequence_ids=chunk.sequence_ids,
            static_table=self.static_table,
            image_store=self.image_store,
        )

    def _eager_batches(self, chunk, indices, *, pad_ranks=False):
        ranked = indices[self.rank :: self.world_size]
        real = None
        if pad_ranks:
            real = torch.arange(self.target_samples) < len(ranked)
            if len(ranked) < self.target_samples:
                # Round-robin sharding leaves at most one missing sample on a rank.
                ranked = torch.cat((ranked, ranked[:1] if len(ranked) else indices[:1]))

        info = get_worker_info()
        workers = info.num_workers if info else 1
        worker, skipped = self._resume_worker(info, workers)
        selection = slice(worker + skipped * self.batch_size * workers, None, workers)
        indices = ranked[selection]
        if real is not None:
            real = real[selection]
        for start in range(0, len(indices), self.batch_size):
            batch = slice(start, start + self.batch_size)
            yield self._window_batch(
                chunk, indices[batch], real[batch] if real is not None else None
            )


ChunkKey = TypeVar("ChunkKey")


@dataclass(frozen=True)
class ChunkReference(Generic[ChunkKey]):
    key: ChunkKey
    samples: int


ReadKey = TypeVar("ReadKey", contravariant=True)


class ChunkReader(Protocol[ReadKey]):
    def read(self, key: ReadKey, /) -> WindowChunk: ...


class LazyWindowDataset(WindowDataset, Generic[ChunkKey]):
    """Schedule, order, and batch streamed window chunks."""

    ram_chunk_limit: int | None = None

    def __init__(
        self,
        config,
        shuffle,
        chunks: Iterable[ChunkReference[ChunkKey]],
        reader: ChunkReader[ChunkKey],
        *,
        image_store_factory: Callable[
            ..., ImageStore | IndexedImageStore | None
        ] = ImageStore,
    ):
        super().__init__(config, shuffle, image_store_factory=image_store_factory)
        self.file_order = configured_file_order(config)
        self.reader = reader
        self.chunks = [chunk for chunk in chunks if chunk.samples > 0]
        if not self.chunks:
            raise ValueError("No usable model windows were found in the dataset.")
        self.target_samples = max(
            sum(chunk.samples for chunk in self.chunks[rank :: self.world_size])
            for rank in range(self.world_size)
        )

    def __iter__(self) -> Iterator[SequifierBatch]:
        yield from self._iter_batches(get_worker_info())

    def _iter_batches(self, worker_info: Any) -> Iterator[SequifierBatch]:
        rank = self.rank
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = self.epoch

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

        worker_id, skip_batches = self._resume_worker(worker_info, num_workers)
        samples, remainder = divmod(self.target_samples, num_workers)
        worker_start_sample = worker_id * samples + min(worker_id, remainder)
        worker_target_samples = samples + int(worker_id < remainder)
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
            chunk = self.reader.read(key)
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
            indices = chunk.sample_order().indices_for_epoch(
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
                new_batch = self._window_batch(
                    chunk,
                    worker_indices[offset : offset + count],
                    positions < rank_real_samples,
                )
                if buffer_len == 0:
                    seq_buffer = new_batch.inputs
                    tgt_buffer = new_batch.targets
                    meta_buffer = new_batch.metadata
                else:
                    if set(meta_buffer) != set(new_batch.metadata):
                        raise RuntimeError(
                            "Inconsistent metadata across window chunks."
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
