"""Load indexed sequence views using the supervised lazy batch scheduler."""

from dataclasses import replace

import torch

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.io.folder_dataset import ChunkReference, LazyFolderDataset
from sequifier.io.indexed_sequences import IndexedView, WindowIndex
from sequifier.io.window_dataset import WindowChunk
from sequifier.io.window_sampling import target_valid_from_offsets


class SequifierDatasetFromIndexed(LazyFolderDataset[int]):
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
            str(self.view.root),
            replace(config, static_features_path=self.view.static_path),
            shuffle,
            "indexed output",
            metadata=metadata,
            indexed_view=self.view,
        )
        self._initialize_window_layout()
        self.columns = sorted(set(config.input_columns + config.target_columns))
        self.view.validate_item_columns(
            [
                *self.columns,
                *(depth_mask_metadata_key(name) for name in config.depth_layouts.root),
            ]
        )
        self.payload_metadata = self.view_metadata.model_copy(
            update={"depth_layouts": config.depth_layouts}
        )
        self.index = WindowIndex(self.view, layout.window_length)
        self._set_chunks(
            ChunkReference(number, len(self._index_for_entries(rows)))
            for number, rows in enumerate(self.index.chunks(coordinates_only=True))
        )

    def _index_for_entries(self, rows):
        pad = torch.tensor([row["pad"] for row in rows], dtype=torch.int64)
        starts = torch.tensor([row["start"] for row in rows], dtype=torch.int64)
        lows = torch.tensor([row["low"] for row in rows], dtype=torch.int64)
        return self.sampling_plan.build_index(
            pad, target_valid_from_offsets(pad, starts, lows)
        )

    def _read_chunk(self, number: int):
        rows = self.index.read_chunk(number)
        payload = self.view.tensor_batch(
            rows, self.payload_metadata, self.columns, self.folder_layout.window_length
        )
        return WindowChunk.from_payload(
            payload, self.config, self.sampling_plan, str(self.view.path)
        )
