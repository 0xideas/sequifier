import os
from collections.abc import Iterator
from typing import Any, Dict

import torch
from loguru import logger

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.helpers import validate_stored_window_width
from sequifier.io.batch import SequifierBatch
from sequifier.io.folder_dataset import EagerFolderDataset
from sequifier.io.pt_payload import load_pt_payload
from sequifier.io.sample_order import (
    SampleOrderPlan,
    curriculum_sample_positions,
    logical_sample_positions,
)
from sequifier.io.window_sampling import (
    build_window_batch,
    target_valid_from_offsets,
    validate_split_bounds_available,
)
from sequifier.typechecking import beartype


class SequifierDatasetFromFolderPt(EagerFolderDataset):
    """Eager PT-folder dataset yielding rank/worker-aligned batches."""

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
        logger.info(f"Loading training dataset into memory from '{self.data_dir}'...")

        all_sequences: Dict[str, list[torch.Tensor]] = {
            col: [] for col in set(config.input_columns + config.target_columns)
        }
        all_left_pad_lengths: list[torch.Tensor] = []
        all_start_item_positions: list[torch.Tensor] = []
        all_split_start_item_positions: list[torch.Tensor] = []
        all_split_end_item_positions: list[torch.Tensor] = []
        all_depth_masks = {name: [] for name in selected_layouts.root}
        self.file_sample_orders: list[tuple[int, SampleOrderPlan]] = []
        logical_offset = 0

        file_infos = list(metadata["batch_files"])
        if self.file_order == "name":
            file_infos.sort(key=lambda item: item["path"])
        for file_info in file_infos:
            file_path = os.path.join(self.data_dir, file_info["path"])
            payload = load_pt_payload(
                file_path,
                layouts=self.depth_layouts,
                n_classes=self.payload_n_classes,
            )
            sequences_batch = payload.sequences
            left_pad_lengths_batch = payload.left_pad_lengths
            for col in all_sequences.keys():
                if col in sequences_batch:
                    validate_stored_window_width(
                        sequences_batch[col], self.folder_layout.window_length
                    )
                    all_sequences[col].append(sequences_batch[col])
            all_left_pad_lengths.append(left_pad_lengths_batch)
            all_start_item_positions.append(payload.start_item_positions)
            all_split_start_item_positions.append(payload.split_start_item_positions)
            all_split_end_item_positions.append(payload.split_end_item_positions)
            local_sample_index = self.sampling_plan.build_index(
                left_pad_lengths_batch,
                target_valid_from_offsets(
                    left_pad_lengths_batch,
                    payload.start_item_positions,
                    payload.split_start_item_positions,
                ),
            )
            local_sample_count = len(local_sample_index)
            self.file_sample_orders.append(
                (
                    logical_offset,
                    SampleOrderPlan.build(
                        local_sample_count,
                        logical_sample_positions(
                            curriculum_sample_positions(
                                config,
                                payload.sample_positions,
                                file_path,
                                payload.curriculum_columns,
                            ),
                            local_sample_index,
                        ),
                    ),
                )
            )
            logical_offset += local_sample_count
            for name in all_depth_masks:
                all_depth_masks[name].append(payload.depth_valid_masks[name])

        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors) for col, tensors in all_sequences.items() if tensors
        }
        self.depth_valid_masks = {
            name: torch.cat(parts) for name, parts in all_depth_masks.items()
        }
        for mask in self.depth_valid_masks.values():
            mask.share_memory_()
        self.left_pad_lengths = torch.cat(all_left_pad_lengths)
        self.start_item_positions = torch.cat(all_start_item_positions)
        self.split_start_item_positions = torch.cat(all_split_start_item_positions)
        self.split_end_item_positions = torch.cat(all_split_end_item_positions)
        validate_split_bounds_available(
            self.start_item_positions,
            self.split_start_item_positions,
            self.split_end_item_positions,
            self.data_dir,
        )
        self.sample_index = self.sampling_plan.build_index(
            self.left_pad_lengths,
            target_valid_from_offsets(
                self.left_pad_lengths,
                self.start_item_positions,
                self.split_start_item_positions,
            ),
        )
        self.n_samples = len(self.sample_index)
        if self.n_samples == 0:
            raise ValueError("No usable model windows were found in the dataset.")
        for tensor in self.sequences.values():
            tensor.share_memory_()
        self.sample_index.share_memory_()

        self.target_samples = self._get_target_samples()
        self.total_batches = self._calculate_total_batches(self.target_samples)

        logger.info(
            f"Dataset loaded into RAM with {self.target_samples} samples and {self.total_batches} batches."
        )

    @beartype
    def __iter__(
        self,
    ) -> Iterator[SequifierBatch]:
        indices_for_worker, sample_is_real_for_worker = self._worker_indices()

        for i in range(0, len(indices_for_worker), self.batch_size):
            batch_indices = indices_for_worker[i : i + self.batch_size]
            batch_sample_is_real = sample_is_real_for_worker[i : i + self.batch_size]

            yield build_window_batch(
                self.sequences,
                self.config.input_columns,
                self.config.target_columns,
                self.sample_index,
                batch_indices,
                batch_sample_is_real,
                depth_valid_masks=self.depth_valid_masks,
                start_item_positions=self.start_item_positions,
                split_start_item_positions=self.split_start_item_positions,
                split_end_item_positions=self.split_end_item_positions,
            )
