import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from sequifier.io.sequifier_dataset_from_folder_lazy import (
    SequifierDatasetFromFolderLazy,
)


# --- Fixtures ---
@pytest.fixture
def mock_config(tmp_path):
    """Creates a mock configuration object."""
    config = MagicMock()
    config.project_root = str(tmp_path)
    # Set a small RAM limit for testing logic, though we will mock psutil check directly
    config.training_spec.max_ram_gb = 1.0
    config.seq_length = 5
    return config


@pytest.fixture
def dataset_path(tmp_path):
    """Sets up a temporary data directory with metadata.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy metadata.json
    # We define 3 files with 10 samples each = 30 total samples
    metadata = {
        "total_samples": 30,
        "batch_files": [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 10},
        ],
    }

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # We don't need to create actual .pt files because we will mock torch.load
    return str(data_dir)


@pytest.fixture
def mock_torch_load():
    """Mocks torch.load to return dummy tensors."""
    with patch("torch.load") as mock_load:
        # Define a side effect to return different dummy data based on filename
        def side_effect(path, map_location):
            # Create dummy return tuple:
            # (sequences, targets, seq_ids, subseq_ids, positions)
            # Tensors size: (10 samples, sequence length 5)
            dummy_seq = {"col1": torch.zeros((10, 5))}
            dummy_tgt = {"tgt1": torch.zeros((10, 5))}
            dummy_ids = torch.arange(10)
            return (
                dummy_seq,
                dummy_tgt,
                dummy_ids,
                dummy_ids,
                dummy_ids,
            )

        mock_load.side_effect = side_effect
        yield mock_load


@pytest.fixture
def mock_memory():
    """Mocks psutil.virtual_memory to control reported memory usage."""
    with patch("psutil.virtual_memory") as mock_mem:
        # Default: plenty of memory available
        mock_mem.return_value.used = 0
        yield mock_mem


# --- Tests ---


def test_initialization(mock_config, dataset_path):
    """Tests that metadata is read correctly on initialization."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config)
    assert len(dataset) == 30
    assert len(dataset.cumulative_samples) == 3
    assert dataset.cumulative_samples == [10, 20, 30]


def test_lazy_loading_and_caching(
    mock_config, dataset_path, mock_torch_load, mock_memory
):
    """Tests that files are loaded on demand and cached."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config)

    # 1. Access index 5 (belongs to file1.pt)
    _ = dataset[5]

    # Verify torch.load was called for file1.pt
    args, _ = mock_torch_load.call_args
    assert "file1.pt" in args[0]
    # Verify cache status
    assert any("file1.pt" in k for k in dataset.cache.keys())
    assert len(dataset.cache) == 1

    # 2. Access index 5 again (should hit cache)
    mock_torch_load.reset_mock()
    _ = dataset[5]
    mock_torch_load.assert_not_called()  # Should not load from disk again


def test_lru_eviction(mock_config, dataset_path, mock_torch_load, mock_memory):
    """Tests that the oldest accessed file is evicted when memory is full."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config)
    dataset.max_ram_bytes = 1000

    # 1. Load File 1 (Index 0) - Memory OK
    mock_memory.return_value.used = 500
    _ = dataset[0]
    assert len(dataset.cache) == 1
    file1_key = [k for k in dataset.cache.keys() if "file1.pt" in k][0]

    # 2. Load File 2 (Index 10) - Memory OK
    _ = dataset[10]
    assert len(dataset.cache) == 2
    file2_key = [k for k in dataset.cache.keys() if "file2.pt" in k][0]

    assert list(dataset.cache.keys()) == [file1_key, file2_key]

    # 3. Simulate Memory Pressure
    # We need psutil.virtual_memory() to return:
    #   1. High memory (1500) -> Triggers 1st eviction
    #   2. Lower memory (900) -> Stops the while loop

    # Create two mock objects representing the memory state
    high_mem = MagicMock(used=1500)
    low_mem = MagicMock(used=900)

    # Apply side_effect to the mock function
    mock_memory.side_effect = [high_mem, low_mem]

    _ = dataset[20]

    # Reset side_effect to avoid affecting other tests/teardown if necessary
    mock_memory.side_effect = None

    file3_key = [k for k in dataset.cache.keys() if "file3.pt" in k][0]

    # Assertions
    assert len(dataset.cache) == 2
    assert file1_key not in dataset.cache  # Oldest (F1) evicted
    assert file2_key in dataset.cache  # F2 kept
    assert file3_key in dataset.cache  # Newest (F3) kept


def test_cache_access_updates_lru_order(
    mock_config, dataset_path, mock_torch_load, mock_memory
):
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config)
    dataset.max_ram_bytes = 1000

    # 1. Load F1, F2
    mock_memory.return_value.used = 500
    _ = dataset[0]
    _ = dataset[10]

    file1_key = [k for k in dataset.cache.keys() if "file1.pt" in k][0]
    file2_key = [k for k in dataset.cache.keys() if "file2.pt" in k][0]
    assert list(dataset.cache.keys()) == [file1_key, file2_key]

    # 2. Re-access F1 (Updates LRU)
    _ = dataset[5]
    assert list(dataset.cache.keys()) == [file2_key, file1_key]

    # 3. Load F3 with Memory Pressure
    # Simulate: High Mem -> Evict 1 item -> Low Mem -> Stop
    high_mem = MagicMock(used=1500)
    low_mem = MagicMock(used=900)
    mock_memory.side_effect = [high_mem, low_mem]

    _ = dataset[20]

    mock_memory.side_effect = None

    # F2 was oldest, F1 was accessed recently
    assert file2_key not in dataset.cache
    assert file1_key in dataset.cache
    assert any("file3.pt" in k for k in dataset.cache.keys())
