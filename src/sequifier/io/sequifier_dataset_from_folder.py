import json
import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from sequifier.config.train_config import TrainModel

# These helpers would typically be in your 'sequifier.helpers' module
from sequifier.helpers import normalize_path


class SequifierDatasetFromFolder(Dataset):
    """
    Custom PyTorch Dataset for loading pre-processed, batched Sequifier data.

    This dataset is designed for high-performance lazy loading. It reads a
    metadata file to understand the dataset's structure (file paths, sample
    counts) and only loads the required batch file from disk when an item is
    requested. An LRU cache is used to speed up sequential access.
    """

    def __init__(self, data_path: str, config: TrainModel):
        """
        Initializes the dataset.

        Args:
            data_path (str): Path to the directory containing pre-processed batch
                             files (.pt) and a metadata.json file.
            config: The training configuration object.
        """
        # The data_path is now a directory, not a single file
        self.data_dir = normalize_path(data_path, config.project_path)
        self.config = config

        # Load metadata to get file list and sample counts
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Please ensure the data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.n_samples = metadata["total_samples"]

        # Create a list of full file paths
        self.file_paths = [
            os.path.join(self.data_dir, info["path"])
            for info in metadata["batch_files"]
        ]

        # Compute cumulative sizes for efficient index mapping
        batch_sizes = [info["samples"] for info in metadata["batch_files"]]
        self.cumulative_sizes = np.cumsum(batch_sizes)

        # Initialize the file loader with an LRU cache to keep recent batches
        # in memory, significantly speeding up mostly-sequential reads.
        self.load_file = lru_cache(maxsize=8)(self._load_file_from_disk)

    def _load_file_from_disk(self, file_path: str) -> tuple[dict, dict]:
        """Helper method to load a single .pt file from disk."""
        return torch.load(file_path)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """
        Retrieves a single training sample from the dataset.

        Args:
            idx (int): The global index of the sample to retrieve.

        Returns:
            A tuple containing two dictionaries: (sequence, targets).
        """
        if not 0 <= idx < self.n_samples:
            raise IndexError(
                f"Index {idx} is out of range for a dataset with {self.n_samples} samples."
            )

        # 1. Find which batch file the index belongs to using the cumulative sizes.
        # `np.searchsorted` efficiently finds the index of the first cumulative
        # sum that is greater than `idx`.
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")

        # 2. Calculate the index of the sample *within* that batch file.
        if file_idx > 0:
            # Subtract the total size of all previous batches
            inner_idx = idx - self.cumulative_sizes[file_idx - 1]
        else:
            # The index is in the very first file
            inner_idx = idx

        # 3. Load the corresponding batch file (this call is cached).
        file_path = self.file_paths[file_idx]
        sequences_batch, targets_batch = self.load_file(file_path)

        # 4. Index the batch of tensors to get the single sample.
        sequence = {key: tensor[inner_idx] for key, tensor in sequences_batch.items()}
        targets = {key: tensor[inner_idx] for key, tensor in targets_batch.items()}

        return sequence, targets
