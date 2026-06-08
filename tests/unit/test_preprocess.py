import json

import numpy as np
import polars as pl
import pytest
import torch

from sequifier.preprocess import (
    RESERVED_MASK_COLUMN,
    Preprocessor,
    _apply_reserved_mask_column,
    _get_column_statistics,
    _get_data_columns,
    create_id_map,
    extract_sequences,
    extract_subsequences,
    get_batch_limits,
    get_combined_statistics,
    process_and_write_data_pt,
)

# ==========================================
# 1. Test Sequence Extraction (Sliding Window)
# ==========================================


def test_extract_subsequences_basic():
    """Tests basic sliding window extraction with sufficient length."""
    input_data = {"col1": [10, 11, 12, 13, 14, 15]}
    seq_length = 3
    stride = 1
    columns = ["col1"]

    # Expected behavior: Window size is seq_length + 1 (history + target)
    # Windows: [10,11,12,13], [11,12,13,14], [12,13,14,15]

    result, left_pad_lengths = extract_subsequences(
        input_data, seq_length, stride, columns, subsequence_start_mode="distribute"
    )

    assert len(result["col1"]) == 3
    assert result["col1"][0] == [10, 11, 12, 13]
    assert result["col1"][2] == [12, 13, 14, 15]


def test_extract_subsequences_padding():
    """Tests that sequences shorter than seq_length are padded with 0s."""
    input_data = {"col1": [1, 2]}  # Length 2
    seq_length = 4  # Req length 5 (4+1)
    stride = 1
    columns = ["col1"]

    # Expected: [0, 0, 0, 1, 2] -> 3 zeroes padding

    result, left_pad_lengths = extract_subsequences(
        input_data, seq_length, stride, columns, subsequence_start_mode="distribute"
    )

    assert len(result["col1"]) == 1
    assert result["col1"][0] == [0, 0, 0, 1, 2]


def test_extract_subsequences_returns_left_pad_lengths_when_requested():
    input_data = {"col1": [0.0, 1.5]}
    result, left_pad_lengths = extract_subsequences(
        input_data,
        seq_length=4,
        stride_for_split=1,
        columns=["col1"],
        subsequence_start_mode="distribute",
    )

    assert result["col1"][0] == [0, 0, 0, 0.0, 1.5]
    assert left_pad_lengths == [3]


def test_extract_sequences_persists_left_pad_length_metadata():
    schema = {
        "sequenceId": pl.Int64,
        "subsequenceId": pl.Int64,
        "startItemPosition": pl.Int64,
        "leftPadLength": pl.Int64,
        "inputCol": pl.String,
        "4": pl.Float64,
        "3": pl.Float64,
        "2": pl.Float64,
        "1": pl.Float64,
        "0": pl.Float64,
    }
    data = pl.DataFrame(
        {
            "sequenceId": [10, 10],
            "itemPosition": [0, 1],
            "col1": [0.0, 1.5],
        }
    )

    sequences = extract_sequences(
        data,
        schema,
        seq_length=4,
        stride_for_split=1,
        columns=["col1"],
        subsequence_start_mode="distribute",
    )

    assert sequences.get_column("leftPadLength").to_list() == [3]


def test_process_and_write_data_pt_persists_left_pad_lengths(tmp_path):
    data = pl.DataFrame(
        {
            "sequenceId": [1, 2],
            "subsequenceId": [0, 0],
            "startItemPosition": [0, 0],
            "leftPadLength": [0, 2],
            "inputCol": ["col1", "col1"],
            "3": [1.0, 0.0],
            "2": [2.0, 0.0],
            "1": [3.0, 1.0],
            "0": [4.0, 2.0],
        }
    )
    out_path = tmp_path / "batch.pt"

    process_and_write_data_pt(
        data,
        seq_length=3,
        path=str(out_path),
        column_types={"col1": "Float64"},
    )

    sequences, _, _, _, left_pad_lengths = torch.load(out_path, weights_only=False)

    assert torch.equal(left_pad_lengths, torch.tensor([0, 2]))
    assert torch.equal(
        sequences["col1"],
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 2.0]]),
    )


def test_reserved_mask_column_is_not_a_data_column():
    data = pl.DataFrame(
        {
            "sequenceId": [1],
            "itemPosition": [0],
            "itemId": [101],
            RESERVED_MASK_COLUMN: [1],
        }
    )

    assert _get_data_columns(data) == ["itemId"]


def test_apply_reserved_mask_column_replaces_values_and_drops_column():
    data = pl.DataFrame(
        {
            "cat_col": [3, 4, 5, 6],
            "num_col": [-1.0, 2.5, 3.5, 4.5],
            RESERVED_MASK_COLUMN: ["0", "1", "0", "1"],
        }
    )

    masked = _apply_reserved_mask_column(
        data,
        ["cat_col", "num_col"],
        {"cat_col": "Int64", "num_col": "Float64"},
    )

    assert RESERVED_MASK_COLUMN not in masked.columns
    assert masked["cat_col"].to_list() == [3, 2, 5, 2]
    assert masked["num_col"].to_list() == [-1.0, 0.0, 3.5, 0.0]


def test_preprocessor_applies_reserved_mask_column_end_to_end(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_path = tmp_path / "masked-input.csv"
    pl.DataFrame(
        {
            "sequenceId": [0, 0, 0],
            "itemPosition": [0, 1, 2],
            "itemId": ["a", "b", "c"],
            "itemValue": [10.0, 100.0, 70.0],
            RESERVED_MASK_COLUMN: [0, 1, 0],
        }
    ).write_csv(data_path)

    Preprocessor(
        project_root=str(project_root),
        continue_preprocessing=False,
        data_path=str(data_path),
        read_format="csv",
        write_format="parquet",
        merge_output=True,
        selected_columns=["itemId", "itemValue"],
        split_ratios=[1.0],
        seq_length=2,
        stride_by_split=[1],
        max_rows=None,
        seed=1010,
        n_cores=1,
        batches_per_file=1024,
        process_by_file=True,
        subsequence_start_mode="distribute",
        use_precomputed_maps=None,
        metadata_config_path=None,
    )

    output = pl.read_parquet(project_root / "data" / "masked-input-split0.parquet")
    metadata_path = project_root / "configs" / "metadata_configs" / "masked-input.json"

    item_sequence = (
        output.filter(pl.col("inputCol") == "itemId").select(["2", "1", "0"]).row(0)
    )
    value_sequence = (
        output.filter(pl.col("inputCol") == "itemValue").select(["2", "1", "0"]).row(0)
    )

    assert RESERVED_MASK_COLUMN not in output.columns
    assert item_sequence == (3.0, 2.0, 4.0)
    assert value_sequence[1] == 0.0
    assert value_sequence[0] != 0.0
    assert value_sequence[2] != 0.0

    metadata = json.loads(metadata_path.read_text())
    assert RESERVED_MASK_COLUMN not in metadata["column_types"]
    assert RESERVED_MASK_COLUMN not in metadata["id_maps"]
    assert RESERVED_MASK_COLUMN not in metadata["selected_columns_statistics"]


@pytest.mark.parametrize("mode", ["distribute", "exact"])
def test_extract_subsequences_modes(mode):
    """Tests logic for 'distribute' vs 'exact' modes."""
    # Length 10. Seq_len 2 (window 3).
    # distribute: adjusts stride to cover data evenly.
    # exact: strictly adheres to stride, throws error if misalignment.
    input_data = {"col1": list(range(10))}
    seq_length = 2
    columns = ["col1"]

    if mode == "distribute":
        stride = 4
        # distribute might adjust indices to maximize coverage
        result, left_pad_lengths = extract_subsequences(
            input_data, seq_length, stride, columns, mode
        )
        assert len(result["col1"]) > 0

    elif mode == "exact":
        stride = (
            3  # (10-1) - 2 = 7. 7 % 3 != 0. Should fail or require specific alignment
        )
        # Testing a failing exact case
        with pytest.raises(ValueError):
            extract_subsequences(input_data, seq_length, 4, columns, mode)

        # Testing a passing exact case
        # (10-1) - 2 = 7. If we change input len to 11: (11-1)-2 = 8. stride 4 works.
        input_data_exact = {"col1": list(range(11))}
        result, left_pad_lengths = extract_subsequences(
            input_data_exact, seq_length, 4, columns, mode
        )
        assert len(result["col1"]) > 0


# ==========================================
# 2. Test Batch Limits (Index Math)
# ==========================================


def test_get_batch_limits_perfect_split():
    """Tests splitting where batch boundaries align perfectly with sequences."""
    # 3 sequences, each length 10. Total 30 rows.
    # If we want 3 batches, we should get 3 chunks of 10.
    data = pl.DataFrame({"sequenceId": np.repeat([1, 2, 3], 10), "val": np.arange(30)})

    limits = get_batch_limits(data, n_batches=3)

    assert len(limits) == 3
    assert limits == [(0, 10), (10, 20), (20, 30)]


def test_get_batch_limits_uneven_split():
    """Tests that a batch never splits a sequenceId in half."""
    # Seq 1: 5 rows
    # Seq 2: 15 rows
    # Seq 3: 5 rows
    # Total 25 rows. Request 2 batches. Ideal size 12.5.
    # Split point should occur at index 5 (Seq 1 end) or 20 (Seq 2 end),
    # NOT at index 12 (middle of Seq 2).
    data = pl.DataFrame(
        {"sequenceId": np.repeat([1, 2, 3], [5, 15, 5]), "val": np.arange(25)}
    )

    limits = get_batch_limits(data, n_batches=2)

    # Check that split points are valid sequence boundaries
    for start, end in limits:
        # Start of batch must match start of a sequence (unless 0)
        if start != 0:
            prev_id = data["sequenceId"][start - 1]
            curr_id = data["sequenceId"][start]
            assert prev_id != curr_id, f"Batch split at {start} broke a sequence"


# ==========================================
# 3. Test Statistics (Mathematical Logic)
# ==========================================


def test_get_combined_statistics_logic():
    """Tests Welford's algorithm logic for merging stats."""
    # Create two chunks of data
    chunk1 = np.random.normal(loc=10, scale=2, size=100)
    chunk2 = np.random.normal(loc=20, scale=5, size=50)
    full_data = np.concatenate([chunk1, chunk2])

    # Real stats
    mean1, std1 = np.mean(chunk1), np.std(chunk1, ddof=1)
    mean2, std2 = np.mean(chunk2), np.std(chunk2, ddof=1)

    # Function output
    comb_mean, comb_std = get_combined_statistics(
        len(chunk1), float(mean1), float(std1), len(chunk2), float(mean2), float(std2)
    )

    # Expected stats
    expected_mean = np.mean(full_data)
    expected_std = np.std(full_data, ddof=1)

    np.testing.assert_almost_equal(comb_mean, expected_mean)
    np.testing.assert_almost_equal(comb_std, expected_std)


def test_get_column_statistics_state_accumulation():
    """Tests that processing data in chunks yields same stats as processing at once."""
    data_full = pl.DataFrame(
        {
            "cat_col": ["a", "b", "a", "c", "b", "d"],
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    # Split into two chunks
    chunk1 = data_full.slice(0, 3)
    chunk2 = data_full.slice(3, 3)

    id_maps = {}
    stats = {}
    running_count = 0

    # Pass 1
    id_maps, stats = _get_column_statistics(
        chunk1, ["cat_col", "num_col"], id_maps, stats, running_count, {}
    )
    running_count += len(chunk1)

    # Pass 2
    id_maps, stats = _get_column_statistics(
        chunk2, ["cat_col", "num_col"], id_maps, stats, running_count, {}
    )

    # Validations
    # 1. Categorical: Should contain all unique keys a,b,c,d
    assert len(id_maps["cat_col"]) == 6
    assert set(id_maps["cat_col"].keys()) == {
        "[unknown]",
        "[other]",
        "a",
        "b",
        "c",
        "d",
    }

    # 2. Numerical: Should match full dataset calculation
    expected_mean = data_full["num_col"].mean()
    expected_std = data_full["num_col"].std()

    np.testing.assert_almost_equal(stats["num_col"]["mean"], expected_mean)
    np.testing.assert_almost_equal(stats["num_col"]["std"], expected_std)


def test_create_id_map():
    """Tests basic ID mapping creation."""
    df = pl.DataFrame({"A": ["z", "x", "y", "x"]})
    mapping = create_id_map(df, "A")

    # Sorted unique values: x, y, z -> 2, 3, 4
    assert mapping["x"] == 3
    assert mapping["y"] == 4
    assert mapping["z"] == 5
