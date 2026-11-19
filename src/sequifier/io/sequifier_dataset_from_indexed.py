"""Load indexed sequence views using the supervised lazy batch scheduler."""

from pathlib import Path

import torch
from torch.utils.data import get_worker_info

from sequifier.io.folder_dataset import LazyFolderChunk, LazyFolderDataset
from sequifier.io.indexed_sequences import WINDOW_CHUNK_SIZE, IndexedView
from sequifier.io.sample_order import curriculum_sample_positions
from sequifier.io.window_sampling import target_valid_from_offsets


class SequifierDatasetFromIndexed(LazyFolderDataset):
    chunk_format = "indexed sequence"

    def __init__(self, data_path, config, shuffle=True):
        self.view = IndexedView.load(data_path)
        self.view_metadata = self.view.metadata()
        metadata = self.view_metadata.model_dump(mode="json")
        metadata["curriculum_column"] = (
            metadata.get("indexed_curriculum_columns") or None
        )
        layout = self.view.layout(config.window_view)
        metadata["window_length"] = layout.window_length
        metadata["max_target_offset"] = layout.max_target_offset
        super().__init__(
            str(self.view.root), config, shuffle, "indexed output", metadata=metadata
        )
        self._initialize_window_layout()
        self.columns = sorted(set(config.input_columns + config.target_columns))
        self.payload_metadata = self.view_metadata.model_copy(
            update={"depth_layouts": config.depth_layouts}
        )
        entries = self.view.entries(layout.window_length)
        self.chunks = {}
        self.batch_files_info = []
        for start in range(0, len(entries), WINDOW_CHUNK_SIZE):
            rows = entries[start : start + WINDOW_CHUNK_SIZE]
            name = str(start)
            self.chunks[name] = rows
            samples = len(self._index_for_entries(rows))
            if samples:
                self.batch_files_info.append({"path": name, "samples": samples})
        self.total_samples = sum(info["samples"] for info in self.batch_files_info)
        if not self.total_samples:
            raise ValueError("No usable model windows were found in the indexed view.")
        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

    def _index_for_entries(self, rows):
        pad = torch.tensor([row["pad"] for row in rows], dtype=torch.int64)
        starts = torch.tensor([row["start"] for row in rows], dtype=torch.int64)
        lows = torch.tensor([row["low"] for row in rows], dtype=torch.int64)
        return self.sampling_plan.build_index(
            pad, target_valid_from_offsets(pad, starts, lows)
        )

    def __iter__(self):
        yield from self._iter_batches(get_worker_info())

    def _read_chunk(self, file_path):
        rows = self.chunks[Path(file_path).name]
        payload = self.view.tensor_batch(
            rows, self.payload_metadata, self.columns, self.folder_layout.window_length
        )
        return LazyFolderChunk(
            sequences=payload.sequences,
            sequence_ids=payload.sequence_ids,
            sample_index=self._index_for_entries(rows),
            sample_positions=curriculum_sample_positions(
                self.config,
                payload.sample_positions,
                file_path,
                payload.curriculum_columns,
            ),
            depth_valid_masks=payload.depth_valid_masks,
            start_item_positions=payload.start_item_positions,
            split_start_item_positions=payload.split_start_item_positions,
            split_end_item_positions=payload.split_end_item_positions,
        )
