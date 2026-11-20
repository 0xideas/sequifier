"""Stored windows and worker-aware batching shared by file and folder datasets."""

from dataclasses import dataclass
from math import ceil

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from sequifier.helpers import WindowSampleIndex, validate_stored_window_width
from sequifier.io.images import ImageStore
from sequifier.io.iteration_state import (
    read_shared_int,
    resolve_resume_worker,
    shared_int,
    write_shared_int,
)
from sequifier.io.sample_order import (
    SampleOrderPlan,
    curriculum_sample_positions,
    logical_sample_positions,
)
from sequifier.io.static_features import table_for_config
from sequifier.io.window_sampling import (
    build_window_batch,
    target_valid_from_offsets,
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
                name: payload.sequences[name]
                for name in dict.fromkeys(config.input_columns + config.target_columns)
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

    def __init__(self, config, shuffle, *, indexed_view=None):
        super().__init__()
        # DataLoader spawn workers do not inherit the process group.
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
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
