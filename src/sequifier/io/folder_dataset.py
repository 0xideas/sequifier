"""Shared folder-dataset setup and distributed batch bookkeeping."""

import json
import math
import os
from collections import Counter
from collections.abc import Iterator

import torch
from loguru import logger

from sequifier.config.metadata import validate_persisted_metadata_versions
from sequifier.helpers import (
    configured_window_stride,
    normalize_path,
    resolve_window_sampling_plan,
    stored_window_layout_from_metadata,
)
from sequifier.io.folder_shards import folder_shard_reader
from sequifier.io.sample_order import (
    concatenate_file_orders,
    configured_file_order,
    validate_curriculum_columns,
)
from sequifier.io.window_dataset import (
    ChunkReference,
    LazyWindowDataset,
    WindowChunk,
    WindowDataset,
)
from sequifier.io.window_sampling import target_valid_from_offsets, temporal_columns


class FolderShards:
    """Read the layout and shard order from a prepared folder manifest."""

    def __init__(self, path, config):
        self.data_dir = normalize_path(path, config.project_root)
        with open(os.path.join(self.data_dir, "metadata.json")) as stream:
            metadata = json.load(stream)
        validate_persisted_metadata_versions(metadata)
        validate_curriculum_columns(
            config, metadata.get("curriculum_column"), self.data_dir
        )
        self.entries = metadata["batch_files"]
        self.sampling_plan = resolve_window_sampling_plan(
            stored_window_layout_from_metadata(metadata),
            config.window_view,
            configured_window_stride(config),
        )
        self.file_order = configured_file_order(config)
        self.reader = folder_shard_reader(config, metadata, self.sampling_plan)

    def files(self):
        entries = self.entries
        if self.file_order == "name":
            entries = sorted(entries, key=lambda entry: entry["path"])
        for entry in entries:
            yield os.path.join(self.data_dir, entry["path"]), entry

    def chunks(self) -> Iterator[ChunkReference[str]]:
        for path, entry in self.files():
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


def load_folder_dataset(path, config, shuffle):
    shards = FolderShards(path, config)
    if config.data_loading.load_full_data_to_ram:
        dataset = EagerFolderDataset(config, shuffle, shards)
    else:
        dataset = LazyWindowDataset(config, shuffle, shards.chunks(), shards.reader)
    logger.info(
        f"Dataset loaded from {path} with {dataset.target_samples} samples and "
        f"{len(dataset)} batches."
    )
    return dataset


class EagerFolderDataset(WindowDataset):
    """Shared sample sharding for fully materialized folders."""

    def __init__(self, config, shuffle, shards):
        super().__init__(config, shuffle)
        self.file_order = shards.file_order
        columns = temporal_columns(config)
        sequences = {name: [] for name in columns}
        depth_masks = {}
        sequence_ids, starts, lows, highs, pads = [], [], [], [], []
        self.file_sample_orders = []
        offset = 0
        for path, _ in shards.files():
            chunk = shards.reader.read(path)
            index = chunk.sample_index
            self.file_sample_orders.append((offset, chunk.sample_order()))
            offset += len(index)
            for name in columns:
                sequences[name].append(chunk.sequences[name])
            for name, mask in chunk.depth_valid_masks.items():
                depth_masks.setdefault(name, []).append(mask)
            sequence_ids.append(chunk.sequence_ids)
            starts.append(chunk.start_item_positions)
            lows.append(chunk.split_start_item_positions)
            highs.append(chunk.split_end_item_positions)
            pads.append(index.left_pad_lengths)
        if not offset:
            raise ValueError("No usable model windows were found in the dataset.")
        start_positions = torch.cat(starts)
        low_positions = torch.cat(lows)
        padding = torch.cat(pads)
        self.chunk = WindowChunk(
            sequences={
                name: torch.cat(parts).share_memory_()
                for name, parts in sequences.items()
            },
            depth_valid_masks={
                name: torch.cat(parts).share_memory_()
                for name, parts in depth_masks.items()
            },
            sequence_ids=torch.cat(sequence_ids),
            start_item_positions=start_positions,
            split_start_item_positions=low_positions,
            split_end_item_positions=torch.cat(highs),
            sample_positions=None,
            sample_index=shards.sampling_plan.build_index(
                padding,
                target_valid_from_offsets(padding, start_positions, low_positions),
            ).share_memory_(),
        )
        self.target_samples = math.ceil(offset / self.world_size)

    def __iter__(self):
        indices = concatenate_file_orders(
            self.file_sample_orders,
            seed=self.config.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
            file_order=self.file_order,
        )
        yield from self._eager_batches(self.chunk, indices, pad_ranks=True)
