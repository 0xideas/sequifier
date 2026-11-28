"""Decode prepared PT and Parquet shards for both in-memory and streaming loaders."""

import polars as pl

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import PANDAS_TO_TORCH_TYPES
from sequifier.io.pt_payload import load_pt_payload, tabular_tensor_batch
from sequifier.io.sample_order import curriculum_columns
from sequifier.io.window_dataset import WindowChunk
from sequifier.io.window_sampling import target_valid_from_offsets, temporal_columns


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
            temporal_columns(config)
        ) != self.selected_layouts.compatibility_signature(temporal_columns(config)):
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

    def read(self, file_path: str) -> WindowChunk:
        payload = load_pt_payload(
            file_path,
            layouts=self.depth_layouts,
            n_classes=self.payload_n_classes,
        )
        return WindowChunk.from_payload(
            payload, self.config, self.sampling_plan, file_path
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

    def read(self, file_path: str) -> WindowChunk:
        payload = tabular_tensor_batch(
            pl.read_parquet(file_path),
            self.column_torch_types,
            temporal_columns(self.config),
            self.sampling_plan.resolved_view.storage.window_length,
            curriculum=curriculum_columns(self.config.curriculum_column),
        )
        return WindowChunk.from_payload(
            payload, self.config, self.sampling_plan, file_path
        )


def folder_shard_reader(config, metadata, sampling_plan):
    if config.read_format == "pt":
        return PtShardReader(config, metadata, sampling_plan)
    if config.read_format == "parquet":
        return ParquetShardReader(config, sampling_plan)
    raise ValueError(f"Folder parts do not support read_format={config.read_format!r}")
