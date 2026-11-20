"""Decode prepared PT and Parquet shards for both in-memory and streaming loaders."""

from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import (
    PANDAS_TO_TORCH_TYPES,
    WindowSampleIndex,
    columns_from_slice,
    get_left_pad_lengths_from_preprocessed_data,
    validate_stored_window_width,
)
from sequifier.io.pt_payload import load_pt_payload
from sequifier.io.sample_order import (
    curriculum_positions_from_parquet,
    curriculum_sample_positions,
)
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
        validate_split_bounds_available(
            payload.start_item_positions,
            payload.split_start_item_positions,
            payload.split_end_item_positions,
            file_path,
        )
        for tensor in payload.sequences.values():
            validate_stored_window_width(
                tensor, self.sampling_plan.resolved_view.storage.window_length
            )
        sample_index = self.sampling_plan.build_index(
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
                self.config,
                payload.sample_positions,
                file_path,
                payload.curriculum_columns,
            ),
            depth_valid_masks={
                name: payload.depth_valid_masks[name]
                for name in self.selected_layouts.root
            },
            start_item_positions=payload.start_item_positions,
            split_start_item_positions=payload.split_start_item_positions,
            split_end_item_positions=payload.split_end_item_positions,
        )


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
        df = pl.read_parquet(file_path)
        left_pad_lengths = get_left_pad_lengths_from_preprocessed_data(df)
        coordinate_columns = [
            "startItemPosition",
            "splitStartItemPosition",
            "splitEndItemPosition",
        ]
        if not set(coordinate_columns).issubset(df.columns):
            raise ValueError(
                f"{file_path}: stored windows require start positions and split bounds"
            )
        positions = (
            df.group_by(["sequenceId", "subsequenceId"])
            .agg(pl.col(coordinate_columns).first())
            .sort(["sequenceId", "subsequenceId"])
        )

        def coordinate(name):
            return torch.tensor(positions[name].to_numpy(), dtype=torch.int64)

        start_item_positions = coordinate("startItemPosition")
        split_start_item_positions = coordinate("splitStartItemPosition")
        sample_index = self.sampling_plan.build_index(
            left_pad_lengths,
            target_valid_from_offsets(
                left_pad_lengths, start_item_positions, split_start_item_positions
            ),
        )
        sample_positions = curriculum_positions_from_parquet(self.config, df, file_path)

        sequence_columns = columns_from_slice(
            slice(0, self.sampling_plan.resolved_view.storage.window_length),
            self.sampling_plan.resolved_view.storage.window_length,
        )
        feature_partitions = {
            frame.item(0, "inputCol"): frame for frame in df.partition_by("inputCol")
        }
        stored_sequences = {}
        for col_name in set(self.config.input_columns + self.config.target_columns):
            if col_name in feature_partitions:
                stored_sequences[col_name] = torch.tensor(
                    feature_partitions[col_name]
                    .sort(["sequenceId", "subsequenceId"])
                    .select(sequence_columns)
                    .to_numpy(),
                    dtype=self.column_torch_types[col_name],
                )
            else:
                raise ValueError(f"Column not found in input data: {col_name}")

        return FolderChunk(
            sequences=stored_sequences,
            sequence_ids=coordinate("sequenceId"),
            sample_index=sample_index,
            sample_positions=sample_positions,
            start_item_positions=start_item_positions,
            split_start_item_positions=split_start_item_positions,
            split_end_item_positions=coordinate("splitEndItemPosition"),
        )


def folder_shard_reader(config, metadata, sampling_plan):
    if config.read_format == "pt":
        return PtShardReader(config, metadata, sampling_plan)
    if config.read_format == "parquet":
        return ParquetShardReader(config, sampling_plan)
    raise ValueError(f"Folder parts do not support read_format={config.read_format!r}")
