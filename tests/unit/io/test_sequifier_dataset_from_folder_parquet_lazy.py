import json
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.io.sequifier_dataset_from_folder_parquet_lazy import (
    SequifierDatasetFromFolderParquetLazy,
)

CONTEXT_LENGTH = 2
FUTURE_CAPACITY = 1
STORED_WIDTH = CONTEXT_LENGTH + FUTURE_CAPACITY


def _folder_metadata(total_samples, batch_files):
    return {
        "total_samples": total_samples,
        "batch_files": batch_files,
        "stored_width": STORED_WIDTH,
        "future_capacity": FUTURE_CAPACITY,
        "stored_window_layout_version": 2,
    }


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.project_root = "."
    config.seed = 42
    config.storage_layout = StoredWindowLayout(
        stored_width=STORED_WIDTH, future_capacity=FUTURE_CAPACITY, version=2
    )
    config.window_view = ModelWindowView(
        context_length=CONTEXT_LENGTH, objective="causal", target_shift=1
    )
    config.column_types = {"item": "Float64"}
    config.training_spec.batch_size = 5
    config.training_spec.num_workers = 0
    config.training_spec.sampling_strategy = "exact"
    config.training_spec.training_objective = "causal"
    config.input_columns = ["item"]
    config.target_columns = ["item"]

    return config


@pytest.fixture
def dataset_path(tmp_path):
    data_dir = tmp_path / "parquet_data"
    data_dir.mkdir()

    # Layout matches the long-format extraction logic
    schema = {
        "sequenceId": pl.Int64,
        "subsequenceId": pl.Int64,
        "startItemPosition": pl.Int64,
        "leftPadLength": pl.Int64,
        "inputCol": pl.String,
        "2": pl.Float64,
        "1": pl.Float64,
        "0": pl.Float64,
    }

    # Populate 4 files with 10 rows (10 sequences) each
    batch_files = []
    for i in range(1, 5):
        filename = f"file_{i}.parquet"
        rows = []
        for s in range(10):
            rows.append((s, 0, s * 2, 0, "item", float(s), float(s + 1), float(s + 2)))

        df = pl.DataFrame(rows, schema=schema, orient="row")
        df.write_parquet(data_dir / filename)
        batch_files.append({"path": filename, "samples": 10})

    metadata = _folder_metadata(40, batch_files)

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return str(data_dir)


def test_initialization(mock_config, dataset_path):
    """Tests that metadata is read correctly and __len__ calculates batches."""
    dataset = SequifierDatasetFromFolderParquetLazy(dataset_path, mock_config)

    # 40 total samples / batch size of 5 = 8 batches
    assert len(dataset) == 8
    assert dataset.total_samples == 40
    assert dataset.target_samples == 40


def test_iteration_yields_correct_batches(mock_config, dataset_path):
    """Tests that the dataset iterates over files and yields structured tensors."""
    dataset = SequifierDatasetFromFolderParquetLazy(
        dataset_path, mock_config, shuffle=False
    )

    batches = list(dataset)
    assert len(batches) == 8  # 40 samples / batch_size 5

    # Check structural integrity of first batch
    batch = batches[0]
    seq_batch = batch.inputs
    tgt_batch = batch.targets
    assert "item" in seq_batch
    assert "item" in tgt_batch
    assert isinstance(seq_batch["item"], torch.Tensor)
    assert seq_batch["item"].shape == (5, 2)  # batch_size=5, seq_len=2
    assert tgt_batch["item"].shape == (5, 2)


def test_iteration_attaches_explicit_padding_masks(mock_config, tmp_path):
    data_dir = tmp_path / "parquet_masks"
    data_dir.mkdir()

    schema = {
        "sequenceId": pl.Int64,
        "subsequenceId": pl.Int64,
        "startItemPosition": pl.Int64,
        "leftPadLength": pl.Int64,
        "inputCol": pl.String,
        "2": pl.Float64,
        "1": pl.Float64,
        "0": pl.Float64,
    }
    rows = [
        (0, 0, 0, 0, "item", 0.0, 1.0, 2.0),
        (1, 0, 0, 1, "item", 0.0, 0.0, 2.0),
        (2, 0, 0, 2, "item", 0.0, 0.0, 2.0),
        (3, 0, 0, 3, "item", 0.0, 0.0, 2.0),
        (4, 0, 0, 0, "item", 0.0, 1.0, 2.0),
    ]
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
        data_dir / "file.parquet"
    )
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(
            _folder_metadata(5, [{"path": "file.parquet", "samples": 5}]),
            f,
        )

    dataset = SequifierDatasetFromFolderParquetLazy(
        str(data_dir), mock_config, shuffle=False
    )
    batch = next(iter(dataset))
    metadata_batch = batch.metadata

    assert torch.equal(
        metadata_batch["attention_valid_mask"],
        torch.tensor(
            [
                [True, True],
                [False, True],
                [False, False],
                [False, False],
                [True, True],
            ]
        ),
    )
    assert torch.equal(
        metadata_batch["target_valid_mask"],
        torch.tensor(
            [
                [True, True],
                [True, True],
                [False, True],
                [False, False],
                [True, True],
            ]
        ),
    )


@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_rank", return_value=0)
@patch("torch.distributed.get_world_size", return_value=2)
def test_distributed_sharding(mock_ws, mock_rank, mock_init, mock_config, dataset_path):
    """Tests that the dataset correctly shards files across distributed GPU ranks."""
    dataset = SequifierDatasetFromFolderParquetLazy(
        dataset_path, mock_config, shuffle=False
    )

    # World size = 2, Total files = 4
    # Rank 0 gets file index 0 and 2 (file_1, file_3) -> 20 samples total -> 4 batches
    batches = list(dataset)
    assert len(batches) == 4

    # Verify input mapping structures
    for batch in batches:
        assert batch.inputs["item"].shape[0] == 5


def test_exact_strategy_uneven_files_exception(mock_config, tmp_path):
    """Tests that FSDP validation raises an Exception when file distribution is uneven."""
    data_dir = tmp_path / "uneven_parquet_data"
    data_dir.mkdir()

    # Write asymmetrical sample quotas across files
    batch_files = [
        {"path": "file_1.parquet", "samples": 15},
        {"path": "file_2.parquet", "samples": 10},
    ]
    metadata = _folder_metadata(25, batch_files)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ):
        with pytest.raises(Exception) as exc_info:
            SequifierDatasetFromFolderParquetLazy(str(data_dir), mock_config)

        assert "different number of samples per rank/GPU" in str(exc_info.value)


def test_oversampling_strategy(mock_config, tmp_path):
    """Tests that shorter ranks oversample to equal the maximal rank count."""
    data_dir = tmp_path / "oversample_parquet_data"
    data_dir.mkdir()

    schema = {
        "sequenceId": pl.Int64,
        "subsequenceId": pl.Int64,
        "startItemPosition": pl.Int64,
        "inputCol": pl.String,
        "2": pl.Float64,
        "1": pl.Float64,
        "0": pl.Float64,
    }

    # File 1 has 15 rows, File 2 has 10 rows
    for i, num_rows in [(1, 15), (2, 10)]:
        rows = [
            (s, 0, s * 2, "item", float(s), float(s + 1), float(s + 2))
            for s in range(num_rows)
        ]
        pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
            data_dir / f"file_{i}.parquet"
        )

    batch_files = [
        {"path": "file_1.parquet", "samples": 15},
        {"path": "file_2.parquet", "samples": 10},
    ]
    metadata = _folder_metadata(25, batch_files)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "oversampling"

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ):
        dataset = SequifierDatasetFromFolderParquetLazy(
            str(data_dir), mock_config, shuffle=False
        )
        # Should match max(15, 10)
        assert dataset.target_samples == 15


def test_undersampling_strategy(mock_config, tmp_path):
    """Tests that longer ranks truncate samples down to match the minimal rank count."""
    data_dir = tmp_path / "undersample_parquet_data"
    data_dir.mkdir()

    schema = {
        "sequenceId": pl.Int64,
        "subsequenceId": pl.Int64,
        "startItemPosition": pl.Int64,
        "inputCol": pl.String,
        "2": pl.Float64,
        "1": pl.Float64,
        "0": pl.Float64,
    }

    for i, num_rows in [(1, 15), (2, 10)]:
        rows = [
            (s, 0, s * 2, "item", float(s), float(s + 1), float(s + 2))
            for s in range(num_rows)
        ]
        pl.DataFrame(rows, schema=schema, orient="row").write_parquet(
            data_dir / f"file_{i}.parquet"
        )

    batch_files = [
        {"path": "file_1.parquet", "samples": 15},
        {"path": "file_2.parquet", "samples": 10},
    ]
    metadata = _folder_metadata(25, batch_files)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    mock_config.training_spec.sampling_strategy = "undersampling"

    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_world_size", return_value=2
    ):
        dataset = SequifierDatasetFromFolderParquetLazy(
            str(data_dir), mock_config, shuffle=False
        )
        # Should match min(15, 10)
        assert dataset.target_samples == 10
