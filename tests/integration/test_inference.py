import json
import os

import numpy as np
import polars as pl
import torch

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}
BERT_SEQ_LENGTH = 8
BERT_EMBEDDING_DIM = 16
BERT_DATA_NAME = "test-data-categorical-1-lookahead-0"


def _categorical_metadata(project_root):
    metadata_path = os.path.join(
        project_root, "configs", "metadata_configs", f"{BERT_DATA_NAME}.json"
    )
    with open(metadata_path, "r") as f:
        return json.load(f)


def _bert_inference_metadata(project_root):
    data_path = os.path.join(project_root, "data", f"{BERT_DATA_NAME}-split2")
    contents = []
    for root, _, files in os.walk(data_path):
        for file in sorted(files):
            if not file.endswith(".pt"):
                continue

            loaded = torch.load(os.path.join(root, file), weights_only=False)
            if len(loaded) == 5:
                _, sequence_ids, subsequence_ids, _, left_pad_lengths = loaded
            else:
                _, sequence_ids, subsequence_ids, _ = loaded
                left_pad_lengths = None

            if left_pad_lengths is None:
                left_pad_lengths = torch.zeros_like(sequence_ids)

            valid_counts = torch.clamp(
                BERT_SEQ_LENGTH - left_pad_lengths, min=0, max=BERT_SEQ_LENGTH
            )
            contents.append(
                pl.DataFrame(
                    {
                        "sequenceId": sequence_ids.detach().cpu().numpy(),
                        "subsequenceId": subsequence_ids.detach().cpu().numpy(),
                        "valid_count": valid_counts.detach().cpu().numpy(),
                    }
                )
            )

    assert len(contents) > 0, f"no files found for {data_path}"
    return pl.concat(contents, how="vertical")


def _expected_bert_valid_counts(project_root, group_columns):
    metadata = _bert_inference_metadata(project_root)

    return (
        metadata.group_by(group_columns)
        .agg(pl.col("valid_count").sum().alias("expected_len"))
        .filter(pl.col("expected_len") > 0)
    )


def _assert_counts_match_expected(actual, expected, group_columns):
    actual_keys = set(actual.select(group_columns).iter_rows())
    expected_keys = set(expected.select(group_columns).iter_rows())

    assert actual_keys == expected_keys

    comparison = actual.join(expected, on=group_columns)
    assert comparison.select(
        (pl.col("len") == pl.col("expected_len")).all()
    ).item(), comparison


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
    valid_values = [str(x) for x in np.arange(100, 130)] + [
        "[unknown]",
        "[other]",
        "[mask]",
    ]
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
                    "[unknown]",
                    "[other]",
                    "[mask]",
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

                assert n_test_rows == n_baseline_rows * prediction_length, (
                    f"Expected {n_baseline_rows * prediction_length} rows for prediction_length={prediction_length}, "
                    f"but found {n_test_rows} rows."
                )

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
            assert model_probabilities.shape[1] == 33
        elif "supCat1" in model_name:
            assert model_probabilities.shape[1] == 13

        np.testing.assert_almost_equal(
            model_probabilities.sum_horizontal(),
            np.ones(model_probabilities.shape[0]),
            decimal=5,
        )


def test_bert_generative_predictions_default_to_context_length(
    bert_predictions, project_root
):
    metadata = _categorical_metadata(project_root)
    valid_values = {str(v) for v in metadata["id_maps"]["itemId"].keys()}.union(
        {"[unknown]", "[other]"}
    )

    assert bert_predictions.height > 0
    assert set(bert_predictions["itemId"].to_list()).issubset(valid_values)

    rows_per_sequence = bert_predictions.group_by("sequenceId").len()
    expected_rows_per_sequence = _expected_bert_valid_counts(
        project_root, ["sequenceId"]
    )
    _assert_counts_match_expected(
        rows_per_sequence, expected_rows_per_sequence, ["sequenceId"]
    )


def test_bert_probabilities(bert_predictions, bert_probabilities, project_root):
    metadata = _categorical_metadata(project_root)
    expected_class_count = metadata["n_classes"]["itemId"]

    assert bert_probabilities.height == bert_predictions.height
    assert bert_probabilities.shape[1] == expected_class_count
    np.testing.assert_almost_equal(
        bert_probabilities.sum_horizontal(),
        np.ones(bert_probabilities.shape[0]),
        decimal=5,
    )


def test_multi_pred(predictions):
    multitarget_models = [name for name in predictions.keys() if "multitarget" in name]

    for model_name in multitarget_models:
        preds = predictions[model_name]

        assert preds.shape[0] > 0, f"{model_name} has no predictions"
        assert preds.shape[1] == 5, f"{model_name} should have 5 columns"

        admssible_vals = [str(x) for x in np.arange(0, 10)] + [
            "[unknown]",
            "[other]",
            "[mask]",
        ]

        assert np.all(
            [v in admssible_vals for v in preds["supCat1"]]
        ), f"Invalid supCat1 values in {model_name}"
        assert np.all(preds["supReal3"].to_numpy() > -4.0) and np.all(
            preds["supReal3"].to_numpy() < 4.0
        ), f"supReal3 out of bounds in {model_name}"


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


def test_bert_embeddings(bert_predictions, bert_embeddings, project_root):
    expected_embedding_cols = [str(i) for i in range(BERT_EMBEDDING_DIM)]
    expected_columns = ["sequenceId", "subsequenceId", "itemPosition"]

    assert bert_embeddings.height == bert_predictions.height
    assert bert_embeddings.shape[1] == len(expected_columns) + BERT_EMBEDDING_DIM
    assert all(col in bert_embeddings.columns for col in expected_columns)
    assert all(col in bert_embeddings.columns for col in expected_embedding_cols)

    embedding_values = bert_embeddings.select(expected_embedding_cols).to_numpy()
    assert np.isfinite(embedding_values).all()

    rows_per_subsequence = bert_embeddings.group_by(
        ["sequenceId", "subsequenceId"]
    ).len()
    expected_rows_per_subsequence = _expected_bert_valid_counts(
        project_root, ["sequenceId", "subsequenceId"]
    )
    _assert_counts_match_expected(
        rows_per_subsequence,
        expected_rows_per_subsequence,
        ["sequenceId", "subsequenceId"],
    )


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
    """Check subsequenceId increments within each sequence."""
    for model_name, embeds_df in embeddings.items():
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

        first_subsequences = embeds_with_diffs.filter(
            pl.col("same_seq") == False  # noqa: E712
        )
        incorrect_starts = first_subsequences.filter(pl.col("subsequenceId") != 0)
        assert incorrect_starts.height == 0, (
            f"Model '{model_name}': Found sequences where subsequenceId does not start at 0:\n"
            f"{incorrect_starts}"
        )

        within_sequence_diffs = embeds_with_diffs.filter(pl.col("same_seq"))
        incorrect_increments = within_sequence_diffs.filter(pl.col("subseq_diff") != 1)
        assert incorrect_increments.height == 0, (
            f"Model '{model_name}': Found incorrect subsequenceId increments within sequences:\n"
            f"{incorrect_increments}"
        )
