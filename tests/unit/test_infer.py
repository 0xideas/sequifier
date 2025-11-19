import numpy as np
import polars as pl

from sequifier.infer import expand_data_by_autoregression, fill_in_predictions_pl

# ==========================================
# 1. Test Autoregressive Expansion
# ==========================================


def test_expand_data_by_autoregression_structure():
    """Tests that new rows are created with correctly incremented IDs."""
    # Setup: 1 sequence, length 3.
    # Columns "1", "2", "3" represent history.
    data = pl.DataFrame(
        {
            "sequenceId": [1],
            "subsequenceId": [10],
            "startItemPosition": [100],
            "inputCol": ["target"],
            "3": [30.0],
            "2": [20.0],
            "1": [10.0],
        }
    )

    seq_length = 3
    extra_steps = 2

    expanded = expand_data_by_autoregression(data, extra_steps, seq_length)

    # 1. Check Row Count: 1 original + 2 extra = 3
    assert expanded.height == 3

    # 2. Check IDs incrementing
    assert expanded["subsequenceId"].to_list() == [10, 11, 12]
    assert expanded["startItemPosition"].to_list() == [100, 101, 102]


def test_expand_data_by_autoregression_shifting():
    """
    Tests the critical column shifting logic.

    In the code:
    data_cols = ['3', '2', '1'] (for seq_length=3)

    Step 1 (Offset 1):
    - Col '3' gets value from Col '2' (Old '2' was 20.0 -> New '3' is 20.0)
    - Col '2' gets value from Col '1' (Old '1' was 10.0 -> New '2' is 10.0)
    - Col '1' gets np.inf (Placeholder)
    """
    data = pl.DataFrame(
        {
            "sequenceId": [1],
            "subsequenceId": [10],
            "startItemPosition": [100],
            "inputCol": ["target"],
            "3": [30.0],
            "2": [20.0],
            "1": [10.0],
        }
    )

    expanded = expand_data_by_autoregression(
        data, autoregression_extra_steps=1, seq_length=3
    )

    # Get the new row (index 1)
    new_row = expanded.row(1, named=True)

    # Verify Shifting
    assert new_row["3"] == 20.0  # Moved from '2'
    assert new_row["2"] == 10.0  # Moved from '1'

    # Verify Placeholder
    assert new_row["1"] == float("inf")


# ==========================================
# 2. Test Prediction Injection
# ==========================================


def test_fill_in_predictions_pl_matching():
    """Tests that predictions fill the correct placeholder based on offset logic."""
    # Setup: A dataframe with a placeholder at offset 1
    # We simulate the state where expand_data has already run.
    # We need 'subsequenceIdAdjusted' because fill_in_predictions relies on it.

    # Row 0: Original (Adjusted ID 0)
    # Row 1: Future (Adjusted ID 1). Col '1' is Inf.
    data = pl.DataFrame(
        {
            "sequenceId": [1, 1],
            "subsequenceIdAdjusted": [0, 1],
            "inputCol": ["target", "target"],
            "1": [10.0, float("inf")],
            # REMOVED "prediction": [None, None] to avoid join collision
        }
    )

    # Prediction made at step 0 for offset 1
    current_subsequence_id = 0
    seq_length = 1
    sequence_ids_present = pl.Series([1])

    preds = {"target": np.array([999.0])}  # Prediction is 999.0

    result = fill_in_predictions_pl(
        data, preds, current_subsequence_id, sequence_ids_present, seq_length
    )

    # Logic Check:
    # The function looks for:
    # (subsequenceIdAdjusted == current + offset) -> (0 + 1 = 1)
    # AND (col '1' is infinite)
    # AND (prediction exists)

    future_val = result.filter(pl.col("subsequenceIdAdjusted") == 1)["1"].item()
    assert future_val == 999.0


def test_fill_in_predictions_pl_ignores_history():
    """Tests that historical/existing values are NOT overwritten."""
    data = pl.DataFrame(
        {
            "sequenceId": [1],
            "subsequenceIdAdjusted": [1],
            "inputCol": ["target"],
            "1": [50.0],  # This is a concrete value, NOT Inf
        }
    )

    preds = {"target": np.array([999.0])}

    # Even if the ID matches, the value is not Inf, so it should remain 50.0
    result = fill_in_predictions_pl(
        data,
        preds,
        current_subsequence_id=0,
        sequence_ids_present=pl.Series([1]),
        seq_length=1,
    )

    val = result["1"].item()
    assert val == 50.0  # Should NOT be 999.0


def test_fill_in_predictions_pl_multi_sequence():
    """Tests correct broadcasting across multiple sequences."""
    # Seq 1 and Seq 2. Both at step 1 (Future). Both waiting for predictions.
    data = pl.DataFrame(
        {
            "sequenceId": [1, 2],
            "subsequenceIdAdjusted": [1, 1],
            "inputCol": ["target", "target"],
            "1": [float("inf"), float("inf")],
        }
    )

    # Preds must be aligned with sequence_ids_present order or joined correctly.
    # fill_in_predictions_pl creates a DF from preds and sequence_ids_present,
    # then JOINs on sequenceId.
    sequence_ids_present = pl.Series([1, 2])
    preds = {"target": np.array([100.0, 200.0])}  # Seq 1 -> 100, Seq 2 -> 200

    result = fill_in_predictions_pl(
        data,
        preds,
        current_subsequence_id=0,
        sequence_ids_present=sequence_ids_present,
        seq_length=1,
    )

    # Check Seq 1
    val1 = result.filter(pl.col("sequenceId") == 1)["1"].item()
    assert val1 == 100.0

    # Check Seq 2
    val2 = result.filter(pl.col("sequenceId") == 2)["1"].item()
    assert val2 == 200.0
