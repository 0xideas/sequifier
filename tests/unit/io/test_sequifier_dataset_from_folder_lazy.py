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
    config.training_spec.batch_size = 5
    config.training_spec.sampling_strategy = "exact"
    config.training_spec.num_workers = 0
    config.seed = 42
    config.seq_length = 5
    return config


@pytest.fixture
def dataset_path(tmp_path):
    """Sets up a temporary data directory with metadata.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy metadata.json
    # We define 4 files with 10 samples each = 40 total samples
    metadata = {
        "total_samples": 40,
        "batch_files": [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 10},
            {"path": "file4.pt", "samples": 10},
        ],
    }

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(data_dir)


@pytest.fixture
def mock_torch_load():
    """Mocks torch.load to return dummy tensors matching the preprocessed format."""
    with patch("torch.load") as mock_load:

        def side_effect(path, map_location, weights_only):
            # Create dummy return tuple: (sequences, targets, ids, ids, pos)
            # Tensors size: (10 samples per file, sequence length 10 to allow slicing)
            dummy_seq = {"col1": torch.ones((10, 10))}
            dummy_tgt = {"tgt1": torch.zeros((10, 10))}

            return (
                dummy_seq,
                dummy_tgt,
                None,
                None,
                None,
            )

        mock_load.side_effect = side_effect
        yield mock_load


# --- Tests ---


def test_initialization(mock_config, dataset_path):
    """Tests that metadata is read correctly and __len__ calculates batches."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config)

    # 40 total samples / batch size of 5 = 8 batches
    assert len(dataset) == 8
    assert len(dataset.batch_files_info) == 4
    assert dataset.total_samples == 40


def test_iteration_yields_correct_batches(mock_config, dataset_path, mock_torch_load):
    """Tests that the dataset iterates over files and yields correct tensor slices."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config, shuffle=False)

    # Consume the generator
    batches = list(dataset)

    # 4 files * 10 samples = 40 total samples. 40 / 5 batch_size = 8 batches
    assert len(batches) == 8

    # Each file has 10 samples, so torch.load should be called 4 times
    assert mock_torch_load.call_count == 4

    # Verify the structure of a yielded batch
    seq_dict, tgt_dict, _, _, _ = batches[0]

    assert "col1" in seq_dict
    assert "tgt1" in tgt_dict

    # Check that batch size and sequence length truncation works properly
    assert seq_dict["col1"].shape == (5, 5)
    assert tgt_dict["tgt1"].shape == (5, 5)


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=0)
def test_distributed_sharding(
    mock_rank, mock_ws, mock_init, mock_config, dataset_path, mock_torch_load
):
    """Tests that the dataset correctly shards files across distributed GPUs."""
    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config, shuffle=False)

    # World size = 2, Total files = 4
    # Rank 0 gets file index 0 and 2 (file1.pt, file3.pt) -> Total 20 samples
    # 20 samples / 5 batch_size = 4 expected batches
    assert len(dataset) == 4

    batches = list(dataset)

    assert len(batches) == 4
    assert mock_torch_load.call_count == 2

    # Verify it loaded the correct specific files
    loaded_files = [call.args[0] for call in mock_torch_load.call_args_list]
    assert any("file1.pt" in f for f in loaded_files)
    assert any("file3.pt" in f for f in loaded_files)
    assert not any("file2.pt" in f for f in loaded_files)


@patch("sequifier.io.sequifier_dataset_from_folder_lazy.get_worker_info")
def test_dataloader_worker_sharding_continuous_boundaries(
    mock_worker_info, mock_config, dataset_path, mock_torch_load
):
    """Tests that continuous sample streaming assigns correct file boundaries to workers."""

    # Simulate being DataLoader worker ID 1 out of 2 total workers
    mock_info = MagicMock()
    mock_info.id = 1
    mock_info.num_workers = 2
    mock_worker_info.return_value = mock_info

    mock_config.training_spec.num_workers = 2

    dataset = SequifierDatasetFromFolderLazy(dataset_path, mock_config, shuffle=False)

    # Consume the generator for THIS specific worker
    batches = list(dataset)

    # NEW LOGIC: Total samples = 40.
    # Worker 0 boundary: samples 0 to 20 (overlaps file1.pt, file2.pt)
    # Worker 1 boundary: samples 20 to 40 (overlaps file3.pt, file4.pt)
    # Since we are testing Worker 1, it should yield 20 samples -> 4 batches
    assert len(batches) == 4
    assert mock_torch_load.call_count == 2

    loaded_files = [call.args[0] for call in mock_torch_load.call_args_list]
    assert any("file3.pt" in f for f in loaded_files)
    assert any("file4.pt" in f for f in loaded_files)
    assert not any("file1.pt" in f for f in loaded_files)


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=0)
def test_exact_strategy_uneven_files_exception(
    mock_rank, mock_ws, mock_init, mock_config, tmp_path
):
    """Tests that 'exact' strategy raises the detailed exception on uneven rank samples."""

    data_dir = tmp_path / "data_uneven"
    data_dir.mkdir()

    # Rank 0 will get file1 (10) + file3 (10) = 20 samples
    # Rank 1 will get file2 (10) + file4 (5) = 15 samples
    metadata = {
        "total_samples": 35,
        "batch_files": [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 10},
            {"path": "file4.pt", "samples": 5},
        ],
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "exact"

    # The dataset initialization calls _get_target_samples(), which should raise the Exception
    with pytest.raises(Exception) as exc_info:
        SequifierDatasetFromFolderLazy(str(data_dir), mock_config)

    error_msg = str(exc_info.value)

    # Assert the core error text is present
    assert "Found 2 different number of samples per rank/GPU" in error_msg
