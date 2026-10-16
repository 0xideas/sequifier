import os
from collections import Counter
from collections.abc import Iterator
from typing import Any

from loguru import logger
from torch.utils.data import get_worker_info

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import validate_stored_window_width
from sequifier.io.batch import SequifierBatch
from sequifier.io.folder_dataset import LazyFolderChunk, LazyFolderDataset
from sequifier.io.pt_payload import load_pt_payload
from sequifier.io.sample_order import curriculum_sample_positions
from sequifier.io.window_sampling import (
    target_valid_from_offsets,
    validate_split_bounds_available,
)
from sequifier.typechecking import beartype


class SequifierDatasetFromFolderPtLazy(LazyFolderDataset):
    """Streams PT chunks into rank/worker-aligned batches."""

    chunk_format = "PT"

    @beartype
    def __init__(self, data_path: str, config: Any, shuffle: bool = True):
        super().__init__(data_path, config, shuffle, "write_format: pt")
        metadata = self.metadata

        self.payload_n_classes = metadata.get("n_classes") or config.n_classes
        self.depth_layouts = DepthLayoutRegistryModel.model_validate(
            metadata.get("depth_layouts", {})
        )
        selected_layouts: DepthLayoutRegistryModel | None = getattr(
            config, "depth_layouts", None
        )
        if selected_layouts is None:
            selected_layouts = DepthLayoutRegistryModel()
        if self.depth_layouts.compatibility_signature(
            config.input_columns
        ) != selected_layouts.compatibility_signature(config.input_columns):
            raise ValueError(
                "PT folder depth layouts are incompatible with the selected interface"
            )
        self._initialize_window_layout()
        self.batch_files_info = []
        raw_file_infos = list(metadata["batch_files"])
        if self.file_order == "name":
            raw_file_infos.sort(key=lambda item: item["path"])
        for raw_file_info in raw_file_infos:
            file_info = dict(raw_file_info)
            file_info["stored_samples"] = int(raw_file_info["samples"])
            histogram = raw_file_info.get("target_valid_from_histogram")
            if histogram is None and not self.sampling_plan.legacy_single_window:
                file_path = os.path.join(self.data_dir, file_info["path"])
                payload = load_pt_payload(
                    file_path,
                    layouts=self.depth_layouts,
                    n_classes=self.payload_n_classes,
                )
                valid_from = target_valid_from_offsets(
                    payload.left_pad_lengths,
                    payload.start_item_positions,
                    payload.split_start_item_positions,
                )
                histogram = {
                    str(value): count
                    for value, count in Counter(valid_from.tolist()).items()
                }
            if self.sampling_plan.legacy_single_window:
                file_info["samples"] = file_info["stored_samples"]
            else:
                assert histogram is not None
                file_info["samples"] = (
                    self.sampling_plan.sample_count_from_target_valid_from_histogram(
                        histogram
                    )
                )
            if file_info["samples"] > 0:
                self.batch_files_info.append(file_info)
        self.total_samples = sum(info["samples"] for info in self.batch_files_info)
        if self.total_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)
        logger.info(
            f"Lazy Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
        )

    @beartype
    def __iter__(self) -> Iterator[SequifierBatch]:
        yield from self._iter_batches(get_worker_info())

    def _read_chunk(self, file_path: str) -> LazyFolderChunk:
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
            validate_stored_window_width(tensor, self.folder_layout.window_length)
        sample_index = self.sampling_plan.build_index(
            payload.left_pad_lengths,
            target_valid_from_offsets(
                payload.left_pad_lengths,
                payload.start_item_positions,
                payload.split_start_item_positions,
            ),
        )
        return LazyFolderChunk(
            sequences=payload.sequences,
            sample_index=sample_index,
            sample_positions=curriculum_sample_positions(
                self.config,
                payload.sample_positions,
                file_path,
                payload.curriculum_columns,
            ),
            depth_valid_masks={
                name: payload.depth_valid_masks[name]
                for name in self.config.depth_layouts.root
            },
            start_item_positions=payload.start_item_positions,
            split_start_item_positions=payload.split_start_item_positions,
            split_end_item_positions=payload.split_end_item_positions,
        )
