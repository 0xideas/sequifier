import json
import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from sequifier.config.train_config import TrainModel

# These helpers would typically be in your 'sequifier.helpers' module
from sequifier.helpers import normalize_path


@lru_cache(maxsize=8)
def _load_file_from_disk_cached(file_path: str) -> tuple[dict, dict]:
    """Helper method to load a single .pt file from disk."""
    return torch.load(file_path)


class SequifierDatasetFromFolder(Dataset):
    """
    Custom PyTorch Dataset for loading pre-processed, batched Sequifier data.
    ...
    """

    def __init__(self, data_path: str, config: TrainModel):
        """
        Initializes the dataset.
        ...
        """
        self.data_dir = normalize_path(data_path, config.project_path)
        self.config = config

        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Please ensure the data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.n_samples = metadata["total_samples"]

        self.file_paths = [
            os.path.join(self.data_dir, info["path"])
            for info in metadata["batch_files"]
        ]

        batch_sizes = [info["samples"] for info in metadata["batch_files"]]
        self.cumulative_sizes = np.cumsum(batch_sizes)

        # 2. The self.load_file assignment and the _load_file_from_disk method are now removed.

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        if not 0 <= idx < self.n_samples:
            raise IndexError(
                f"Index {idx} is out of range for a dataset with {self.n_samples} samples."
            )

        file_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")

        if file_idx > 0:
            inner_idx = idx - self.cumulative_sizes[file_idx - 1]
        else:
            inner_idx = idx

        file_path = self.file_paths[file_idx]

        # 3. Call the new module-level cached function
        sequences_batch, targets_batch = _load_file_from_disk_cached(file_path)

        sequence = {key: tensor[inner_idx] for key, tensor in sequences_batch.items()}
        targets = {key: tensor[inner_idx] for key, tensor in targets_batch.items()}

        return sequence, targets
