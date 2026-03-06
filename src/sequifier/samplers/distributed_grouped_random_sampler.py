from logging import Logger
from typing import Iterator, Optional, Union

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import Sampler

from sequifier.io.sequifier_dataset_from_folder import SequifierDatasetFromFolder
from sequifier.io.sequifier_dataset_from_folder_lazy import (
    SequifierDatasetFromFolderLazy,
)


@beartype
def get_final_indices(
    files_for_this_rank: list[int],
    index_groups: list[list[int]],
    shuffle: bool,
    generator: Optional[torch.Generator],
):
    final_indices = []
    for file_idx in files_for_this_rank:
        # IMPORTANT: Create a copy to avoid mutating self.index_groups in-place
        group = list(index_groups[file_idx])

        if shuffle:
            assert generator is not None
            perm = torch.randperm(len(group), generator=generator).tolist()  # type: ignore
            group = [group[i] for i in perm]

        final_indices.extend(group)
    return final_indices


class DistributedGroupedRandomSampler(Sampler[int]):
    """
    A distributed sampler that groups samples by file to improve cache efficiency.

    This sampler partitions the set of data FILES across the distributed processes,
    not the individual samples. Each process then iterates through its assigned
    files in a random order. Within each file, the samples are also shuffled.

    This ensures that each process sees a unique subset of the data per epoch
    while maximizing sequential reads from the same file, which is ideal for
    lazy-loading datasets.
    """

    def __init__(
        self,
        data_source: Union[SequifierDatasetFromFolder, SequifierDatasetFromFolderLazy],
        num_replicas: int,
        rank: int,
        logger: Logger,
        shuffle: bool = True,
        seed: int = 0,
        sampling_strategy: str = "exact",
    ):
        """
        Args:
            data_source: The dataset to sample from. Must have a `batch_files_info`
                         attribute.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process.
            shuffle: If True, shuffles the order of files and samples within files.
            seed: Random seed used to create the permutation.
            sampling_strategy: str = How to distribute data between GPUs
        """
        super().__init__(None)
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.logger = logger
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.sampling_strategy = sampling_strategy

        # Pre-compute the global indices for each file, same as before
        self.index_groups = []
        start_index = 0
        for file_info in self.data_source.batch_files_info:
            num_samples_in_file = file_info["samples"]
            indices = list(range(start_index, start_index + num_samples_in_file))
            self.index_groups.append(indices)
            start_index += num_samples_in_file

        # Determine the number of files and samples this rank will process
        self.num_files = len(self.index_groups)
        self.files_for_this_rank = list(range(self.num_files))[
            self.rank :: self.num_replicas
        ]

        if len(self.files_for_this_rank) == 0:
            if self.sampling_strategy == "oversampling":
                random_viable_rank = np.random.randint(self.num_files)
                self.files_for_this_rank = list(range(self.num_files))[
                    random_viable_rank :: self.num_replicas
                ]
            else:
                raise Exception(
                    f"No file found for GPU rank {self.rank}. Total number of files found: {self.num_files}. Please adapt your data or use 'oversampling'"
                )

        if self.sampling_strategy == "exact":
            if self.num_files % self.num_replicas != 0:
                raise ValueError(
                    f"Number of input files ({self.num_files}) must be divisible by "
                    f"world_size ({self.num_replicas}) when using 'exact' sampling strategy."
                )
            self.num_samples = sum(
                len(self.index_groups[i]) for i in self.files_for_this_rank
            )

        elif self.sampling_strategy == "oversampling":
            max_samples = 0
            for r in range(self.num_replicas):
                files_for_r = list(range(self.num_files))[r :: self.num_replicas]
                samples_for_r = sum(len(self.index_groups[i]) for i in files_for_r)
                max_samples = max(max_samples, samples_for_r)
            self.num_samples = max_samples

        elif self.sampling_strategy == "undersampling":
            min_samples = float("inf")
            for r in range(self.num_replicas):
                files_for_r = list(range(self.num_files))[r :: self.num_replicas]
                samples_for_r = sum(len(self.index_groups[i]) for i in files_for_r)
                min_samples = min(min_samples, samples_for_r)
            self.num_samples = int(min_samples)

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over indices for the current rank.
        """
        # 1. Initialize generator with deterministic seed for this epoch
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            files_for_this_rank_order = torch.randperm(
                len(self.files_for_this_rank), generator=generator
            ).tolist()
        else:
            generator = None
            files_for_this_rank_order = list(range(len(self.files_for_this_rank)))

        # 2. Assign a unique, non-overlapping subset of shuffled files to this rank
        files_for_this_rank = [
            self.files_for_this_rank[i] for i in files_for_this_rank_order
        ]

        final_indices = get_final_indices(
            files_for_this_rank, self.index_groups, self.shuffle, generator
        )
        if self.sampling_strategy == "oversampling":
            while len(final_indices) < self.num_samples:
                additional_file_for_this_rank = int(
                    torch.randint(0, self.num_files, (1,), generator=generator).item()
                )
                additional_indices = get_final_indices(
                    [additional_file_for_this_rank],
                    self.index_groups,
                    self.shuffle,
                    generator,
                )
                required_additional_indices = self.num_samples - len(final_indices)
                final_indices.extend(additional_indices[:required_additional_indices])
                files_for_this_rank.append(additional_file_for_this_rank)
        elif self.sampling_strategy == "undersampling":
            final_indices = final_indices[: self.num_samples]

        self.logger.info(f"Files for rank {self.rank}: {files_for_this_rank}")

        return iter(final_indices)

    def __len__(self) -> int:
        """
        Returns the number of samples for the current rank, not the total.
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. This is used to create a different
        shuffling order for each epoch.
        """
        self.epoch = epoch
