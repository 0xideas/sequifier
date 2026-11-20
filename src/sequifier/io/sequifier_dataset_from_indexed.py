"""Load indexed sequence views using the supervised lazy batch scheduler."""

from dataclasses import replace

import torch

from sequifier.config.depth_layout import depth_mask_metadata_key
from sequifier.helpers import configured_window_stride, resolve_window_sampling_plan
from sequifier.io.indexed_sequences import IndexedView, WindowIndex
from sequifier.io.sample_order import validate_curriculum_columns
from sequifier.io.window_dataset import ChunkReference, LazyWindowDataset, WindowChunk
from sequifier.io.window_sampling import target_valid_from_offsets


def load_indexed_dataset(path, config, shuffle):
    view = IndexedView.load(path)
    config = replace(config, static_features_path=view.static_path)
    reader = IndexedWindowReader(view, config)
    return LazyWindowDataset(
        config, shuffle, reader.chunks(), reader, image_store_factory=view.image_store
    )


class IndexedWindowReader:
    def __init__(self, view, config):
        self.view = view
        self.config = config
        metadata = view.metadata()
        validate_curriculum_columns(
            config, getattr(metadata, "indexed_curriculum_columns", ()), str(view.path)
        )
        layout = view.layout(config.window_view)
        self.sampling_plan = resolve_window_sampling_plan(
            layout, config.window_view, configured_window_stride(config)
        )
        self.columns = sorted(set(config.input_columns + config.target_columns))
        self.view.validate_item_columns(
            [
                *self.columns,
                *(depth_mask_metadata_key(name) for name in config.depth_layouts.root),
            ]
        )
        self.payload_metadata = metadata.model_copy(
            update={"depth_layouts": config.depth_layouts}
        )
        self.index = WindowIndex(self.view, layout.window_length)

    def chunks(self):
        for number, rows in enumerate(self.index.chunks(coordinates_only=True)):
            yield ChunkReference(number, len(self._index_for_entries(rows)))

    def _index_for_entries(self, rows):
        pad = torch.tensor([row["pad"] for row in rows], dtype=torch.int64)
        starts = torch.tensor([row["start"] for row in rows], dtype=torch.int64)
        lows = torch.tensor([row["low"] for row in rows], dtype=torch.int64)
        return self.sampling_plan.build_index(
            pad, target_valid_from_offsets(pad, starts, lows)
        )

    def read(self, number: int):
        rows = self.index.read_chunk(number)
        payload = self.view.tensor_batch(
            rows, self.payload_metadata, self.columns, self.index.width
        )
        return WindowChunk.from_payload(
            payload, self.config, self.sampling_plan, str(self.view.path)
        )
