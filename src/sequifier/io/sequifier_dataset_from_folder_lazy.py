import bisect
import collections
import json
import os
from typing import Dict, Tuple

import psutil  # Dependency: pip install psutil
import torch
from torch.utils.data import Dataset

from sequifier.config.train_config import TrainModel
from sequifier.helpers import normalize_path


class SequifierDatasetFromFolderLazy(Dataset):
    """
    An efficient PyTorch Dataset for datasets that do not fit into RAM.

    This class loads data from individual .pt files on-demand (lazily) when an
    item is requested via `__getitem__`. It maintains an in-memory cache of
    recently used files to speed up access. To prevent memory exhaustion,
    the cache is managed by a Least Recently Used (LRU) policy, which
    evicts the oldest data chunks when the total system RAM usage exceeds a
    configurable threshold.

    This strategy balances I/O overhead and memory usage, making it suitable
    for training on datasets larger than the available system memory.
    """

    def __init__(self, data_path: str, config: TrainModel, ram_threshold: float = 70.0):
        """
        Initializes the dataset by reading metadata and setting up the cache.

        Args:
            data_path (str): The path to the directory containing the pre-processed
                             .pt files and a metadata.json file.
            config (TrainModel): The training configuration object.
            ram_threshold (float): The system RAM usage percentage (0-100)
                                   at which to trigger cache eviction.
        """
        self.data_dir = normalize_path(data_path, config.project_path)
        self.config = config
        self.ram_threshold = ram_threshold
        metadata_path = os.path.join(self.data_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{self.data_dir}'. "
                "Ensure data is pre-processed with write_format: pt."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.n_samples = metadata["total_samples"]
        self.batch_files_info = metadata["batch_files"]

        # --- Build an index for fast sample-to-file mapping ---
        # self.cumulative_samples will store the cumulative sample count at the end
        # of each file, e.g., [1024, 2048, 3072, ...], allowing for a fast binary search.
        self.cumulative_samples = []
        current_sum = 0
        for file_info in self.batch_files_info:
            current_sum += file_info["samples"]
            self.cumulative_samples.append(current_sum)

        # --- Initialize cache and thread-safety mechanisms ---
        # An OrderedDict is used to implement the LRU logic.
        self.cache: collections.OrderedDict[
            str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = collections.OrderedDict()

        print(
            f"[INFO] Initialized lazy dataset from {self.data_dir}. "
            f"Total samples: {self.n_samples}. RAM threshold: {self.ram_threshold}%"
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def _find_file_for_index(self, idx: int) -> Tuple[int, str]:
        """
        Finds which file contains the given sample index and the local index within it.

        Args:
            idx: The global sample index across all files.

        Returns:
            A tuple containing (local_index_in_file, file_path).
        """
        # bisect_right finds the insertion point for idx, which corresponds to the
        # index of the file containing this sample.
        file_index = bisect.bisect_right(self.cumulative_samples, idx)

        # Calculate the local index within the identified file.
        # If it's the first file (index 0), the local index is just idx.
        # Otherwise, subtract the cumulative sample count of the previous file.
        previous_samples = (
            self.cumulative_samples[file_index - 1] if file_index > 0 else 0
        )
        local_index = idx - previous_samples

        file_path = os.path.join(
            self.data_dir, self.batch_files_info[file_index]["path"]
        )

        return local_index, file_path

    def _evict_lru_items(self):
        """
        Checks system memory and evicts least recently used items from the cache
        until usage is below the threshold. This method must be called from
        within a locked context.
        """
        while psutil.virtual_memory().percent > self.ram_threshold:
            if not self.cache:
                # Cache is empty, but memory is still high. Nothing to evict.
                break

            # popitem(last=False) removes and returns the (key, value) pair that
            # was first inserted, effectively implementing the LRU policy.
            evicted_path, _ = self.cache.popitem(last=False)
            print(
                f"[INFO] RAM usage {psutil.virtual_memory().percent:.1f}% > {self.ram_threshold}%. "
                f"Evicting {os.path.basename(evicted_path)} from cache."
            )

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieves a single data sample, loading from disk if not in the cache.

        This method is the core of the lazy-loading strategy. It is thread-safe
        and manages the cache automatically.
        """
        if not 0 <= idx < self.n_samples:
            raise IndexError(
                f"Index {idx} is out of range for dataset with {self.n_samples} samples."
            )

        local_index, file_path = self._find_file_for_index(idx)

        # Acquire lock to ensure atomic cache operations
        # 1. Check for a cache hit
        if file_path in self.cache:
            # Mark as recently used by moving it to the end of the OrderedDict.
            self.cache.move_to_end(file_path)
            sequences_batch, targets_batch = self.cache[file_path]

        # 2. Handle a cache miss
        else:
            # Load the data from the .pt file from disk.
            sequences_batch, targets_batch, _ = torch.load(
                file_path, map_location="cpu"
            )

            # Add the newly loaded data to the cache.
            self.cache[file_path] = (sequences_batch, targets_batch)

            # After adding, check memory and evict old items if necessary.
            self._evict_lru_items()

        # 3. Retrieve the specific sample from the (now cached) batch tensors.
        sequence = {key: tensor[local_index] for key, tensor in sequences_batch.items()}
        targets = {key: tensor[local_index] for key, tensor in targets_batch.items()}

        return sequence, targets
