"""Decode prepared PT and Parquet shards for both in-memory and streaming loaders."""

from dataclasses import dataclass

import polars as pl
from torch import Tensor

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    WindowSampleIndex,
    validate_stored_window_width,
)
from sequifier.io.pt_payload import load_pt_payload, tabular_tensor_batch
from sequifier.io.sample_order import curriculum_columns, curriculum_sample_positions
from sequifier.io.window_sampling import (
    target_valid_from_offsets,
    validate_split_bounds_available,
)


@dataclass
class FolderChunk:
    """Format-independent contents of one stored folder chunk."""

    sequences: dict[str, Tensor]
    sequence_ids: Tensor
    sample_index: WindowSampleIndex
    sample_positions: Tensor | None
    start_item_positions: Tensor
    split_start_item_positions: Tensor
    split_end_item_positions: Tensor
    depth_valid_masks: dict[str, Tensor] | None = None


class PtShardReader:
    def __init__(self, config, metadata, sampling_plan):
        self.config = config
        self.sampling_plan = sampling_plan
        self.payload_n_classes = metadata.get("n_classes") or config.n_classes
        self.depth_layouts = DepthLayoutRegistryModel.model_validate(
            metadata.get("depth_layouts", {})
        )
        self.selected_layouts = config.depth_layouts or DepthLayoutRegistryModel()
        if self.depth_layouts.compatibility_signature(
            config.input_columns
        ) != self.selected_layouts.compatibility_signature(config.input_columns):
            raise ValueError(
                "PT folder depth layouts are incompatible with the selected interface"
            )

    def read_valid_from(self, file_path):
        payload = load_pt_payload(
            file_path,
            layouts=self.depth_layouts,
            n_classes=self.payload_n_classes,
        )
        return target_valid_from_offsets(
            payload.left_pad_lengths,
            payload.start_item_positions,
            payload.split_start_item_positions,
        ).tolist()

    def read(self, file_path: str) -> FolderChunk:
        payload = load_pt_payload(
            file_path,
            layouts=self.depth_layouts,
            n_classes=self.payload_n_classes,
        )
        return _folder_chunk(payload, self.config, self.sampling_plan, file_path)


class ParquetShardReader:
    def __init__(self, config, sampling_plan):
        self.config = config
        self.sampling_plan = sampling_plan
        self.column_torch_types = {
            col: PANDAS_TO_TORCH_TYPES[dtype]
            for col, dtype in config.column_data_types.items()
        }

    def read_valid_from(self, file_path):
        return (
            pl.scan_parquet(file_path)
            .group_by(["sequenceId", "subsequenceId"])
            .agg(
                pl.col("leftPadLength").first(),
                pl.col("startItemPosition").first(),
                pl.col("splitStartItemPosition").first(),
            )
            .select(
                pl.max_horizontal(
                    "leftPadLength",
                    pl.col("splitStartItemPosition") - pl.col("startItemPosition"),
                ).alias("targetValidFrom")
            )
            .collect()
            .get_column("targetValidFrom")
            .to_list()
        )

    def read(self, file_path: str) -> FolderChunk:
        payload = tabular_tensor_batch(
            pl.read_parquet(file_path),
            self.column_torch_types,
            list(dict.fromkeys(self.config.input_columns + self.config.target_columns)),
            self.sampling_plan.resolved_view.storage.window_length,
            curriculum=curriculum_columns(self.config.curriculum_column),
        )
        return _folder_chunk(payload, self.config, self.sampling_plan, file_path)


def _folder_chunk(payload, config, sampling_plan, file_path):
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
    return FolderChunk(
        sequences=payload.sequences,
        sequence_ids=payload.sequence_ids,
        sample_index=sample_index,
        sample_positions=curriculum_sample_positions(
            config,
            payload.sample_positions,
            file_path,
            payload.curriculum_columns,
        ),
        depth_valid_masks={
            name: payload.depth_valid_masks[name] for name in config.depth_layouts.root
        },
        start_item_positions=payload.start_item_positions,
        split_start_item_positions=payload.split_start_item_positions,
        split_end_item_positions=payload.split_end_item_positions,
    )


def folder_shard_reader(config, metadata, sampling_plan):
    if config.read_format == "pt":
        return PtShardReader(config, metadata, sampling_plan)
    if config.read_format == "parquet":
        return ParquetShardReader(config, sampling_plan)
    raise ValueError(f"Folder parts do not support read_format={config.read_format!r}")
