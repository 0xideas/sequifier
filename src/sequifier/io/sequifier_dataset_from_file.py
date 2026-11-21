from typing import Any

from loguru import logger

from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    configured_window_stride,
    read_data,
    resolve_window_sampling_plan,
)
from sequifier.io.sample_order import curriculum_columns
from sequifier.io.window_dataset import WindowChunk, WindowDataset
from sequifier.typechecking import beartype


class SequifierDatasetFromFile(WindowDataset):
    """Eager single-file dataset yielding pre-collated batches."""

    @beartype
    def __init__(self, data_path: str, config: Any, shuffle: bool = True):
        super().__init__(config, shuffle)
        all_columns = sorted(set(config.input_columns + config.target_columns))
        logger.info(f"Loading dataset into memory from '{data_path}'...")
        sampling_plan = resolve_window_sampling_plan(
            config.storage_layout,
            config.window_view,
            configured_window_stride(config),
        )
        if config.read_format == "pt":
            from sequifier.io.pt_payload import load_pt_payload

            try:
                part = config.part
            except AttributeError:
                part = None
            metadata = part.metadata if part is not None else None
            payload = load_pt_payload(
                data_path,
                layouts=metadata.depth_layouts
                if metadata is not None
                else config.depth_layouts,
                n_classes=metadata.n_classes
                if metadata is not None
                else config.n_classes,
            )
        else:
            from sequifier.io.pt_payload import tabular_tensor_batch

            payload = tabular_tensor_batch(
                read_data(data_path, config.read_format),
                {
                    col: PANDAS_TO_TORCH_TYPES[config.column_data_types[col]]
                    for col in all_columns
                },
                all_columns,
                config.storage_layout.window_length,
                curriculum=curriculum_columns(config.curriculum_column),
            )
        self.chunk = WindowChunk.from_payload(payload, config, sampling_plan, data_path)
        self.sample_order = self.chunk.sample_order()
        samples = len(self.chunk.sample_index)
        if not samples:
            raise ValueError("No usable model windows were found in the dataset.")
        self.target_samples = len(range(self.rank, samples, self.world_size))
        for mask in self.chunk.depth_valid_masks.values():
            mask.share_memory_()
        if config.device.startswith("cuda"):
            self.chunk.sequences = {
                name: tensor.pin_memory()
                for name, tensor in self.chunk.sequences.items()
            }
        logger.info(f"Dataset loaded with {samples} samples.")

    def __iter__(self):
        indices = self.sample_order.indices_for_epoch(
            seed=self.config.seed, epoch=self.epoch, shuffle=self.shuffle
        )
        yield from self._eager_batches(self.chunk, indices)
