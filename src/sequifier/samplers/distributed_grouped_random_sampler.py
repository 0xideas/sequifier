from typing import Iterator, Union

import torch
from torch.utils.data import Sampler

from sequifier.io.sequifier_dataset_from_folder import SequifierDatasetFromFolder
from sequifier.io.sequifier_dataset_from_folder_lazy import (
    SequifierDatasetFromFolderLazy,
)


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
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            data_source: The dataset to sample from. Must have a `batch_files_info`
                         attribute.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process.
            shuffle: If True, shuffles the order of files and samples within files.
            seed: Random seed used to create the permutation.
        """
        super().__init__(None)
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

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

        self.num_samples = sum(
            len(self.index_groups[i]) for i in self.files_for_this_rank
        )

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over indices for the current rank.
        """
        # 1. Initialize generator with deterministic seed for this epoch
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            all_files_order = torch.randperm(
                self.num_files, generator=generator
            ).tolist()
        else:
            all_files_order = list(range(self.num_files))

        # 2. Assign a unique, non-overlapping subset of shuffled files to this rank
        files_for_this_rank = all_files_order[self.rank :: self.num_replicas]

        # 3. Create the final list of indices for this rank
        final_indices = []
        for file_idx in files_for_this_rank:
            # IMPORTANT: Create a copy to avoid mutating self.index_groups in-place
            group = list(self.index_groups[file_idx])

            if self.shuffle:
                perm = torch.randperm(len(group), generator=generator).tolist()  # type: ignore
                group = [group[i] for i in perm]

            final_indices.extend(group)

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
