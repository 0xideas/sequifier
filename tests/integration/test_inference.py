import numpy as np
import polars as pl

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}


def test_predictions_real(predictions):
    for model_name, model_predictions in predictions.items():
        if "categorical" not in model_name or "multitarget" in model_name:
            if "multitarget" not in model_name:
                assert np.all(
                    [
                        v > -10.0 and v < 10.0
                        for v in model_predictions[
                            TARGET_VARIABLE_DICT["real"]
                        ].to_numpy()
                    ]
                )
            else:
                assert np.all(
                    [
                        v > -10.0 and v < 10.0
                        for v in model_predictions["supReal3"].to_numpy()
                    ]
                ), model_predictions


def test_predictions_cat(predictions):
    valid_values = [str(x) for x in np.arange(100, 130)] + ["unknown"]
    for model_name, model_predictions in predictions.items():
        if "categorical" in model_name or "multitarget" in model_name:
            assert np.all(
                [
                    v in valid_values
                    for v in model_predictions[
                        TARGET_VARIABLE_DICT["categorical"]
                    ].to_numpy()
                ]
            ), model_predictions

            if "multitarget" in model_name:
                admssible_vals = [str(x) for x in np.arange(0, 10)] + [
                    "unknown",
                    "other",
                ]
                assert np.all(
                    [
                        v in admssible_vals
                        for v in model_predictions["supCat1"].to_numpy()
                    ]
                ), model_predictions

            if "inf" in model_name:
                prediction_length = 3
                n_test_rows = model_predictions.height
                baseline_preds = predictions["model-categorical-1-best-3"]
                n_baseline_rows = baseline_preds.height

                # 3. Assert the number of rows is scaled by prediction_length
                assert n_test_rows == n_baseline_rows * prediction_length, (
                    f"Expected {n_baseline_rows * prediction_length} rows for prediction_length={prediction_length}, "
                    f"but found {n_test_rows} rows."
                )

                # 4. Assert correct number of predictions per sequence
                baseline_rows_per_seq = (
                    baseline_preds.group_by("sequenceId").len().height
                )
                test_rows_per_seq_groups = model_predictions.group_by(
                    "sequenceId"
                ).len()

                assert baseline_rows_per_seq == test_rows_per_seq_groups.height
                assert (
                    test_rows_per_seq_groups["len"] == prediction_length
                ).all(), (
                    f"Test should have {prediction_length} predictions per sequence"
                )


def test_probabilities(probabilities):
    for model_name, model_probabilities in probabilities.items():
        if "itemId" in model_name:
            assert model_probabilities.shape[1] == 32
        elif "supCat1" in model_name:
            assert model_probabilities.shape[1] == 12

        np.testing.assert_almost_equal(
            model_probabilities.sum_horizontal(),
            np.ones(model_probabilities.shape[0]),
            decimal=5,
        )


def test_multi_pred(predictions):
    preds = predictions["model-categorical-multitarget-5-best-3"]

    assert preds.shape[0] > 0
    assert preds.shape[1] == 5

    admssible_vals = [str(x) for x in np.arange(0, 10)] + ["unknown", "other"]

    assert np.all([v in admssible_vals for v in preds["supCat1"]])
    assert np.all(preds["supReal3"].to_numpy() > -4.0) and np.all(
        preds["supReal3"].to_numpy() < 4.0
    )


def test_embeddings(embeddings):
    for model_name, model_embeddings in embeddings.items():
        if "categorical-1" in model_name:
            assert model_embeddings.shape[0] == 10
            assert model_embeddings.shape[1] == 203
            assert np.abs(model_embeddings[:, 1:].to_numpy().mean()) < 0.3
        if "categorical-3" in model_name:
            assert model_embeddings.shape[0] == 30
            assert model_embeddings.shape[1] == 203
            assert np.abs(model_embeddings[:, 1:].to_numpy().mean()) < 0.3


def test_predictions_item_position(predictions):
    """
    Checks if itemPosition increments correctly within each sequenceId.
    """
    for model_name, preds_df in predictions.items():
        # Ensure correct sorting for comparison
        preds_df_sorted = preds_df.sort("sequenceId", "itemPosition")

        # Calculate differences and sequence changes
        preds_with_diffs = preds_df_sorted.with_columns(
            (pl.col("itemPosition") - pl.col("itemPosition").shift(1)).alias(
                "pos_diff"
            ),
            (pl.col("sequenceId") == pl.col("sequenceId").shift(1)).alias("same_seq"),
        )

        # Filter for rows within the same sequence (excluding the first row of each sequence)
        within_sequence_diffs = preds_with_diffs.filter(pl.col("same_seq"))

        # Check if all position differences within sequences are 1
        incorrect_increments = within_sequence_diffs.filter(pl.col("pos_diff") != 1)

        assert incorrect_increments.height == 0, (
            f"Model '{model_name}': Found incorrect itemPosition increments within sequences:\n"
            f"{incorrect_increments}"
        )


def test_embeddings_subsequence_id(embeddings):
    """
    Checks if subsequenceId increments correctly within each sequenceId
    and starts at 0 for each new sequence.
    """
    for model_name, embeds_df in embeddings.items():
        # Ensure correct sorting
        embeds_df_sorted = embeds_df.sort("sequenceId", "subsequenceId")

        shift_val = 0
        if "categorical-1" in model_name:
            shift_val = 1
        if "categorical-3" in model_name:
            shift_val = 3

        # Calculate differences and sequence changes
        embeds_with_diffs = embeds_df_sorted.with_columns(
            (pl.col("subsequenceId") - pl.col("subsequenceId").shift(shift_val)).alias(
                "subseq_diff"
            ),
            (pl.col("sequenceId") == pl.col("sequenceId").shift(shift_val)).alias(
                "same_seq"
            ),
        )

        # 1. Check if subsequenceId starts at 0 for each new sequence
        first_subsequences = embeds_with_diffs.filter(
            pl.col("same_seq") == False  # noqa: E712
        )  # First row of each sequence
        incorrect_starts = first_subsequences.filter(pl.col("subsequenceId") != 0)
        assert incorrect_starts.height == 0, (
            f"Model '{model_name}': Found sequences where subsequenceId does not start at 0:\n"
            f"{incorrect_starts}"
        )

        # 2. Check if subsequenceId increments by 1 within sequences
        within_sequence_diffs = embeds_with_diffs.filter(pl.col("same_seq"))
        incorrect_increments = within_sequence_diffs.filter(pl.col("subseq_diff") != 1)
        assert incorrect_increments.height == 0, (
            f"Model '{model_name}': Found incorrect subsequenceId increments within sequences:\n"
            f"{incorrect_increments}"
        )
