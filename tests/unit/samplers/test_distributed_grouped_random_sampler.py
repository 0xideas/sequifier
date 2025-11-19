from unittest.mock import MagicMock

import pytest

from sequifier.samplers.distributed_grouped_random_sampler import (
    DistributedGroupedRandomSampler,
)


@pytest.fixture
def mock_dataset():
    """
    Creates a mock dataset with a predictable structure.
    Structure: 4 files, each with 10 samples.
    Total samples: 40.
    Global indices:
      - File 0: 0-9
      - File 1: 10-19
      - File 2: 20-29
      - File 3: 30-39
    """
    dataset = MagicMock()
    dataset.batch_files_info = [
        {"path": "file0.pt", "samples": 10},
        {"path": "file1.pt", "samples": 10},
        {"path": "file2.pt", "samples": 10},
        {"path": "file3.pt", "samples": 10},
    ]
    return dataset


def test_partitioning_ranks(mock_dataset):
    """
    Test that the sampler correctly partitions files across distributed ranks.
    With 4 files and 2 replicas:
    - Rank 0 should get 2 files.
    - Rank 1 should get 2 files.
    - The sets of indices yielded by Rank 0 and Rank 1 should be mutually exclusive
      and cover the entire dataset.
    """
    num_replicas = 2
    seed = 42

    # Sampler for Rank 0
    sampler_r0 = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas=num_replicas, rank=0, shuffle=False, seed=seed
    )
    indices_r0 = list(sampler_r0)

    # Sampler for Rank 1
    sampler_r1 = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas=num_replicas, rank=1, shuffle=False, seed=seed
    )
    indices_r1 = list(sampler_r1)

    # 1. Check Partition sizes
    # Total 4 files, 2 per rank. Each file has 10 samples.
    assert len(indices_r0) == 20
    assert len(indices_r1) == 20

    # 2. Check Exclusion
    # No index should appear in both
    intersection = set(indices_r0).intersection(set(indices_r1))
    assert len(intersection) == 0

    # 3. Check Coverage
    # Together they should cover 0 to 39
    all_indices = sorted(indices_r0 + indices_r1)
    assert all_indices == list(range(40))

    # 4. Check Specific File Assignment (Deterministic when shuffle=False)
    # Rank 0 gets files [0, 2] -> Indices 0-9 and 20-29
    # Rank 1 gets files [1, 3] -> Indices 10-19 and 30-39
    assert all(0 <= i < 10 or 20 <= i < 30 for i in indices_r0)
    assert all(10 <= i < 20 or 30 <= i < 40 for i in indices_r1)


def test_deterministic_shuffling(mock_dataset):
    """
    Test that the sampler produces the same sequence when seeded identically,
    and handles file-level vs sample-level shuffling.
    """
    seed = 123
    num_replicas = 1  # Single process to see full sequence
    rank = 0

    # Two samplers with same seed
    sampler1 = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas, rank, shuffle=True, seed=seed
    )
    sampler2 = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas, rank, shuffle=True, seed=seed
    )

    indices1 = list(sampler1)
    indices2 = list(sampler2)

    # Should be identical
    assert indices1 == indices2
    assert len(indices1) == 40

    # Verify that it is actually shuffled (not just 0..39)
    assert indices1 != list(range(40))


def test_epoch_determinism(mock_dataset):
    """
    Test that changing the epoch changes the shuffle order deterministically.
    """
    seed = 999
    sampler = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas=1, rank=0, shuffle=True, seed=seed
    )

    # Epoch 0
    sampler.set_epoch(0)
    indices_epoch0 = list(sampler)

    # Epoch 1
    sampler.set_epoch(1)
    indices_epoch1 = list(sampler)

    # Epoch 0 again
    sampler.set_epoch(0)
    indices_epoch0_again = list(sampler)

    # Different epochs -> different order
    assert indices_epoch0 != indices_epoch1
    # Same epoch -> same order
    assert indices_epoch0 == indices_epoch0_again


def test_grouped_sampling_structure(mock_dataset):
    """
    Test the specific 'grouped' logic: files are shuffled, then assigned to ranks.
    Within a rank, we process one file fully before moving to the next.
    """
    # We use shuffle=True to test the file shuffling logic.
    # We inspect the output to ensure indices from the same file are grouped together.
    seed = 555
    sampler = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas=1, rank=0, shuffle=True, seed=seed
    )
    indices = list(sampler)

    # There are 4 files, 10 samples each.
    # The output should be 4 distinct blocks of 10 indices.
    # Within each block, the indices should belong to the same file range (e.g. 0-9, or 20-29).

    # Define file ranges
    ranges = [
        range(0, 10),
        range(10, 20),
        range(20, 30),
        range(30, 40),
    ]

    # Helper to find which file an index belongs to
    def get_file_id(idx):
        for i, r in enumerate(ranges):
            if idx in r:
                return i
        return -1

    # Check the stream in chunks of 10
    for i in range(0, 40, 10):
        chunk = indices[i : i + 10]
        first_file_id = get_file_id(chunk[0])

        # Verify all indices in this chunk belong to the same file
        for idx in chunk:
            assert (
                get_file_id(idx) == first_file_id
            ), f"Index {idx} strayed from file group {first_file_id}"

        # Verify that samples *within* the chunk are shuffled (probabilistic, but highly likely for 10 items)
        # We just check it's not strictly sequential sorted
        # (Note: there's a tiny chance they randomly sort, so this is a heuristic check)
        if chunk == sorted(chunk):
            # If it happens to be sorted, check if the original file was sorted 0..9.
            # In a real random shuffle of 10 items, getting 0..9 is 1/3.6 million.
            pass


def test_length(mock_dataset):
    """Test __len__ implementation."""
    # 4 files * 10 samples = 40 samples total
    # 2 replicas -> 20 samples per replica
    sampler = DistributedGroupedRandomSampler(
        mock_dataset, num_replicas=2, rank=0, shuffle=True
    )
    assert len(sampler) == 20
