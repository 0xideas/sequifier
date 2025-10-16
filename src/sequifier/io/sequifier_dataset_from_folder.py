import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import normalize_path


class SequifierDatasetFromFolder(Dataset):
    """
    An efficient PyTorch Dataset that pre-loads all data into RAM.

    This is the ideal strategy when the entire dataset split can fit into the
    system's memory. It pays a one-time I/O cost at initialization, after which
    all data access during training is extremely fast (RAM access).
    """

    def __init__(self, data_path: str, config: TrainModel):
        """
        Initializes the dataset by loading all .pt files from the data directory
        into memory.
        """
        self.data_dir = normalize_path(data_path, config.project_path)
        self.config = config
        metadata_path = os.path.join(self.data_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Ensure data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.batch_files_info = metadata["batch_files"]
        self.n_samples = metadata["total_samples"]

        print(f"Pre-loading dataset from {self.data_dir} into RAM...")

        all_sequences: Dict[str, List[torch.Tensor]] = {
            col: [] for col in config.selected_columns
        }
        all_targets: Dict[str, List[torch.Tensor]] = {
            col: [] for col in config.target_columns
        }

        # Load all data files and collect tensors
        for file_info in metadata["batch_files"]:
            file_path = os.path.join(self.data_dir, file_info["path"])
            sequences_batch, targets_batch, _ = torch.load(
                file_path, map_location="cpu"
            )

            for col in all_sequences.keys():
                if col in sequences_batch:
                    all_sequences[col].append(sequences_batch[col])

            for col in all_targets.keys():
                if col in targets_batch:
                    all_targets[col].append(targets_batch[col])

        # Concatenate all tensors into a single large tensor for each column
        self.sequences: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors) for col, tensors in all_sequences.items() if tensors
        }
        self.targets: Dict[str, torch.Tensor] = {
            col: torch.cat(tensors) for col, tensors in all_targets.items() if tensors
        }

        for tensor in self.sequences.values():
            tensor.share_memory_()
        for tensor in self.targets.values():
            tensor.share_memory_()

        print(f"Dataset pre-loading complete. Total samples: {self.n_samples}")

        # Verify that the number of loaded samples matches the metadata
        first_key = next(iter(self.sequences.keys()))
        if self.sequences[first_key].shape[0] != self.n_samples:
            raise ValueError(
                f"Mismatch in sample count! Metadata: {self.n_samples}, Loaded: {self.sequences[first_key].shape[0]}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not 0 <= idx < self.n_samples:
            raise IndexError(
                f"Index {idx} is out of range for a dataset with {self.n_samples} samples."
            )

        # Accessing data is now just a fast slice from the pre-loaded tensors in RAM
        sequence = {key: tensor[idx] for key, tensor in self.sequences.items()}
        targets = {key: tensor[idx] for key, tensor in self.targets.items()}

        return sequence, targets
