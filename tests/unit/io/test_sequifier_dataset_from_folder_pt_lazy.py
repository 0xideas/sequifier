import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.io.sequifier_dataset_from_folder_pt_lazy import (
    SequifierDatasetFromFolderPtLazy,
)

CONTEXT_LENGTH = 5
FUTURE_CAPACITY = 1
STORED_WIDTH = CONTEXT_LENGTH + FUTURE_CAPACITY


def _folder_metadata(total_samples, batch_files):
    return {
        "total_samples": total_samples,
        "batch_files": batch_files,
        "stored_context_width": STORED_WIDTH,
        "max_target_offset": FUTURE_CAPACITY,
        "stored_window_layout_version": 2,
    }


# --- Fixtures ---
@pytest.fixture
def mock_config(tmp_path):
    """Mock dataset config."""
    config = MagicMock()
    config.project_root = str(tmp_path)
    config.training_spec.batch_size = 5
    config.training_spec.sampling_strategy = "exact"
    config.training_spec.num_workers = 0
    config.seed = 42
    config.storage_layout = StoredWindowLayout(
        stored_context_width=STORED_WIDTH, max_target_offset=FUTURE_CAPACITY, version=2
    )
    config.window_view = ModelWindowView(
        context_length=CONTEXT_LENGTH, objective="causal", target_offset=1
    )
    config.input_columns = ["col1"]
    config.target_columns = ["col1", "tgt1"]
    config.training_spec.training_objective = "causal"
    return config


@pytest.fixture
def dataset_path(tmp_path):
    """Sets up a temporary data directory with metadata.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy metadata.json
    # We define 4 files with 10 samples each = 40 total samples
    metadata = _folder_metadata(
        40,
        [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 10},
            {"path": "file4.pt", "samples": 10},
        ],
    )

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(data_dir)


@pytest.fixture
def mock_torch_load():
    """Mocks torch.load to return dummy tensors matching the preprocessed format."""
    with patch("torch.load") as mock_load:

        def side_effect(path, map_location, weights_only):
            dummy_seq = {"col1": torch.ones((10, 6)), "tgt1": torch.zeros((10, 6))}
            left_pad_lengths = torch.zeros(10, dtype=torch.int64)

            return (
                dummy_seq,
                None,
                None,
                None,
                left_pad_lengths,
            )

        mock_load.side_effect = side_effect
        yield mock_load


# --- Tests ---


def test_initialization(mock_config, dataset_path):
    """Metadata-backed batch length."""
    dataset = SequifierDatasetFromFolderPtLazy(dataset_path, mock_config)

    # 40 total samples / batch size of 5 = 8 batches
    assert len(dataset) == 8
    assert len(dataset.batch_files_info) == 4
    assert dataset.total_samples == 40


def test_iteration_yields_correct_batches(mock_config, dataset_path, mock_torch_load):
    """Tensor slices from file iteration."""
    dataset = SequifierDatasetFromFolderPtLazy(dataset_path, mock_config, shuffle=False)

    # Consume the generator
    batches = list(dataset)

    # 4 files * 10 samples = 40 total samples. 40 / 5 batch_size = 8 batches
    assert len(batches) == 8

    # Each file has 10 samples, so torch.load should be called 4 times
    assert mock_torch_load.call_count == 4

    # Verify the structure of a yielded batch
    batch = batches[0]
    seq_dict = batch.inputs
    tgt_dict = batch.targets

    assert "col1" in seq_dict, f"{seq_dict = }"

    assert "tgt1" in tgt_dict, f"{tgt_dict = }"
    # Check that batch size and sequence length truncation works properly
    assert seq_dict["col1"].shape == (5, 5)
    assert tgt_dict["tgt1"].shape == (5, 5)


def test_iteration_attaches_explicit_padding_masks(mock_config, dataset_path):
    with patch("torch.load") as mock_load:

        def side_effect(path, map_location, weights_only):
            dummy_seq = {
                "col1": torch.ones((10, 6)),
                "tgt1": torch.ones((10, 6)),
            }
            left_pad_lengths = torch.tensor([0, 1, 2, 3, 4, 5, 0, 0, 0, 0])
            return (
                dummy_seq,
                torch.arange(10),
                torch.zeros(10, dtype=torch.int64),
                torch.zeros(10, dtype=torch.int64),
                left_pad_lengths,
            )

        mock_load.side_effect = side_effect
        dataset = SequifierDatasetFromFolderPtLazy(
            dataset_path, mock_config, shuffle=False
        )
        batch = next(iter(dataset))
        metadata_dict = batch.metadata

    assert torch.equal(
        metadata_dict["attention_valid_mask"],
        torch.tensor(
            [
                [True, True, True, True, True],
                [False, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ]
        ),
    )
    assert torch.equal(
        metadata_dict["target_valid_mask"],
        torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, True, True, True, True],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        ),
    )


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=0)
def test_distributed_sharding(
    mock_rank, mock_ws, mock_init, mock_config, dataset_path, mock_torch_load
):
    """Distributed file sharding."""
    dataset = SequifierDatasetFromFolderPtLazy(dataset_path, mock_config, shuffle=False)

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


@patch("sequifier.io.sequifier_dataset_from_folder_pt_lazy.get_worker_info")
def test_dataloader_worker_sharding_continuous_boundaries(
    mock_worker_info, mock_config, tmp_path
):
    """Worker boundaries can start and stop inside PT files."""
    data_dir = tmp_path / "data_uneven_worker"
    data_dir.mkdir()
    file_specs = [
        ("file1.pt", 7, 0),
        ("file2.pt", 11, 100),
        ("file3.pt", 13, 200),
        ("file4.pt", 9, 300),
    ]
    metadata = _folder_metadata(
        sum(samples for _, samples, _ in file_specs),
        [{"path": path, "samples": samples} for path, samples, _ in file_specs],
    )
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_info = MagicMock()
    mock_info.id = 1
    mock_info.num_workers = 2
    mock_worker_info.return_value = mock_info

    mock_config.training_spec.num_workers = 2

    file_offsets = {path: (samples, offset) for path, samples, offset in file_specs}

    def load_uneven_file(path, map_location, weights_only):
        file_name = path.split("/")[-1]
        samples, offset = file_offsets[file_name]
        values = torch.arange(offset, offset + samples * STORED_WIDTH).reshape(
            samples, STORED_WIDTH
        )
        return (
            {"col1": values, "tgt1": values.clone()},
            None,
            None,
            None,
            torch.zeros(samples, dtype=torch.int64),
        )

    with patch("torch.load", side_effect=load_uneven_file) as mock_load:
        dataset = SequifierDatasetFromFolderPtLazy(
            str(data_dir), mock_config, shuffle=False
        )
        batches = list(dataset)

    assert len(batches) == 4
    assert mock_load.call_count == 2

    loaded_files = [call.args[0] for call in mock_load.call_args_list]
    assert any("file3.pt" in f for f in loaded_files)
    assert any("file4.pt" in f for f in loaded_files)
    assert not any("file1.pt" in f for f in loaded_files)
    assert not any("file2.pt" in f for f in loaded_files)

    sample_ids = []
    for batch in batches:
        sample_ids.extend(batch.inputs["col1"][:, 0].tolist())

    assert sample_ids == [
        200 + 2 * STORED_WIDTH,
        200 + 3 * STORED_WIDTH,
        200 + 4 * STORED_WIDTH,
        200 + 5 * STORED_WIDTH,
        200 + 6 * STORED_WIDTH,
        200 + 7 * STORED_WIDTH,
        200 + 8 * STORED_WIDTH,
        200 + 9 * STORED_WIDTH,
        200 + 10 * STORED_WIDTH,
        200 + 11 * STORED_WIDTH,
        200 + 12 * STORED_WIDTH,
        300,
        300 + STORED_WIDTH,
        300 + 2 * STORED_WIDTH,
        300 + 3 * STORED_WIDTH,
        300 + 4 * STORED_WIDTH,
        300 + 5 * STORED_WIDTH,
        300 + 6 * STORED_WIDTH,
        300 + 7 * STORED_WIDTH,
        300 + 8 * STORED_WIDTH,
    ]


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=0)
def test_exact_strategy_uneven_files_exception(
    mock_rank, mock_ws, mock_init, mock_config, tmp_path
):
    """Exact mode rejects uneven rank samples."""

    data_dir = tmp_path / "data_uneven"
    data_dir.mkdir()

    # Rank 0 will get file1 (10) + file3 (10) = 20 samples
    # Rank 1 will get file2 (10) + file4 (5) = 15 samples
    metadata = _folder_metadata(
        35,
        [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 10},
            {"path": "file4.pt", "samples": 5},
        ],
    )
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "exact"

    # The dataset initialization calls _get_target_samples(), which should raise the Exception
    with pytest.raises(Exception) as exc_info:
        SequifierDatasetFromFolderPtLazy(str(data_dir), mock_config)

    error_msg = str(exc_info.value)

    # Assert the core error text is present
    assert "Found 2 different number of samples per rank/GPU" in error_msg


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=1)
def test_oversampling_strategy(
    mock_rank, mock_ws, mock_init, mock_config, tmp_path, mock_torch_load
):
    """Oversampling loops short ranks."""
    data_dir = tmp_path / "data_oversample"
    data_dir.mkdir()

    # Rank 0 gets file1 (10) + file3 (5) = 15 samples
    # Rank 1 gets file2 (10) = 10 samples
    metadata = _folder_metadata(
        25,
        [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 5},
        ],
    )
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "oversampling"
    mock_config.training_spec.batch_size = 5
    mock_config.training_spec.num_workers = 0

    dataset = SequifierDatasetFromFolderPtLazy(
        str(data_dir), mock_config, shuffle=False
    )

    # Max samples across ranks is 15. Rank 1 must pad its 10 samples up to 15.
    assert dataset.target_samples == 15
    assert len(dataset) == 3  # 15 total samples / batch_size 5

    batches = list(dataset)
    assert len(batches) == 3

    # Rank 1 only has file2 assigned. To get 15 samples, it must load file2,
    # run out of data, loop back, and load file2 again.
    assert mock_torch_load.call_count == 2
    loaded_files = [call.args[0] for call in mock_torch_load.call_args_list]
    assert "file2.pt" in loaded_files[0]
    assert "file2.pt" in loaded_files[1]


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=2)
@patch("torch.distributed.get_rank", return_value=0)
def test_undersampling_strategy(
    mock_rank, mock_ws, mock_init, mock_config, tmp_path, mock_torch_load
):
    """Undersampling truncates heavy ranks."""
    data_dir = tmp_path / "data_undersample"
    data_dir.mkdir()

    # Rank 0 gets file1 (10) + file3 (5) = 15 samples
    # Rank 1 gets file2 (10) = 10 samples
    metadata = _folder_metadata(
        25,
        [
            {"path": "file1.pt", "samples": 10},
            {"path": "file2.pt", "samples": 10},
            {"path": "file3.pt", "samples": 5},
        ],
    )
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "undersampling"
    mock_config.training_spec.batch_size = 5
    mock_config.training_spec.num_workers = 0

    dataset = SequifierDatasetFromFolderPtLazy(
        str(data_dir), mock_config, shuffle=False
    )

    # Min samples across ranks is 10. Rank 0 must truncate its 15 samples down to 10.
    assert dataset.target_samples == 10
    assert len(dataset) == 2  # 10 total samples / batch_size 5

    batches = list(dataset)
    assert len(batches) == 2

    # Rank 0 should only need file1 (10 samples) to meet its truncated quota of 10.
    # It should never attempt to load file3.
    assert mock_torch_load.call_count == 1
    loaded_files = [call.args[0] for call in mock_torch_load.call_args_list]
    assert "file1.pt" in loaded_files[0]
    assert not any("file3.pt" in f for f in loaded_files)
