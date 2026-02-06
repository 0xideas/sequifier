import numpy as np
import polars as pl
import pytest

from sequifier.preprocess import (
    _get_column_statistics,
    create_id_map,
    extract_subsequences,
    get_batch_limits,
    get_combined_statistics,
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

    result = extract_subsequences(
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

    result = extract_subsequences(
        input_data, seq_length, stride, columns, subsequence_start_mode="distribute"
    )

    assert len(result["col1"]) == 1
    assert result["col1"][0] == [0, 0, 0, 1, 2]


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
        result = extract_subsequences(input_data, seq_length, stride, columns, mode)
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
        result = extract_subsequences(input_data_exact, seq_length, 4, columns, mode)
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
    assert len(id_maps["cat_col"]) == 4
    assert set(id_maps["cat_col"].keys()) == {"a", "b", "c", "d"}

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
    assert mapping["x"] == 2
    assert mapping["y"] == 3
    assert mapping["z"] == 4
