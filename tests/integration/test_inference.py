import os

import numpy as np
import polars as pl
import pytest

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}


@pytest.fixture()
def model_names_preds():
    model_names_preds = [
        f"model-{variant}-{model_number}-best-3"
        for variant in ["categorical", "real"]
        for model_number in [1, 3, 5, 50]
    ]
    model_names_preds += [
        "model-categorical-multitarget-5-best-3",
        "model-real-1-best-3-autoregression",
        "model-categorical-1-best-3-autoregression",
    ]

    return model_names_preds


@pytest.fixture()
def model_names_probs():
    model_names_probs = [
        f"model-categorical-{model_number}-best-3-itemId"
        for model_number in [1, 3, 5, 50]
    ]
    model_names_probs += [
        f"model-categorical-multitarget-5-best-3-{col}" for col in ["itemId", "sup1"]
    ]
    return model_names_probs


@pytest.fixture()
def model_names_embeddings():
    model_names_embeddings = ["model-categorical-1-best-embedding-3"]
    return model_names_embeddings


@pytest.fixture()
def targets(model_names_preds, model_names_probs, model_names_embeddings):
    target_dict = {"preds": {}, "probs": {}, "embeds": {}}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-predictions"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "predictions", file_name
        )
        target_dict["preds"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    for model_name in model_names_probs:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-probabilities"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "probabilities", file_name
        )
        target_dict["probs"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    for model_name in model_names_embeddings:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-embeddings"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "embeddings", file_name
        )
        target_dict["embeds"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    return target_dict


def read_multi_file_preds(path, target_type, file_suffix=None):
    dtype = (
        {TARGET_VARIABLE_DICT[target_type]: str}
        if target_type == "categorical"
        else None
    )
    if target_type == "categorical":
        contents = []
        for root, dirs, files in os.walk(path):
            for file in sorted(list(files)):
                if file_suffix is None or file.endswith(file_suffix):
                    contents.append(
                        pl.read_csv(os.path.join(root, file), schema_overrides=dtype)
                    )
        assert len(contents) > 0, f"no files found for {path}"
        return pl.concat(contents, how="vertical")
    else:
        return pl.read_csv(f"{path}.csv", separator=",", schema_overrides=dtype)


@pytest.fixture()
def predictions(run_inference, model_names_preds, project_path):
    preds = {}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"

        prediction_path = os.path.join(
            project_path,
            "outputs",
            "predictions",
            f"sequifier-{model_name}-predictions",
        )

        preds[model_name] = read_multi_file_preds(prediction_path, target_type)

    return preds


@pytest.fixture()
def probabilities(run_inference, model_names_probs, project_path):
    probs = {}
    for model_name in model_names_probs:
        probabilities_path = os.path.join(
            project_path,
            "outputs",
            "probabilities",
            f"sequifier-{model_name}-probabilities",
        )
        probs[model_name] = read_multi_file_preds(
            probabilities_path, "categorical", "csv"
        )

    return probs


@pytest.fixture()
def embeddings(run_inference, model_names_embeddings, project_path):
    embeds = {}
    for model_name in model_names_embeddings:
        embeddings_path = os.path.join(
            project_path,
            "outputs",
            "embeddings",
            f"sequifier-{model_name}-embeddings",
        )
        embeds[model_name] = read_multi_file_preds(
            embeddings_path, "categorical", "csv"
        )
    return embeds


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
                        for v in model_predictions["sup3"].to_numpy()
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
                valid_values_sup1 = [x for x in np.arange(0, 10)] + ["unknown"]
                assert np.all(
                    [
                        v in valid_values_sup1
                        for v in model_predictions["sup1"].to_numpy()
                    ]
                ), model_predictions


def test_probabilities(probabilities):
    for model_name, model_probabilities in probabilities.items():
        if "itemId" in model_name:
            assert model_probabilities.shape[1] == 31
        elif "sup1" in model_name:
            assert model_probabilities.shape[1] == 11

        np.testing.assert_almost_equal(
            model_probabilities.sum_horizontal(),
            np.ones(model_probabilities.shape[0]),
            decimal=5,
        )


def test_multi_pred(predictions):
    preds = predictions["model-categorical-multitarget-5-best-3"]

    assert preds.shape[0] > 0
    assert preds.shape[1] == 5
    assert np.all(preds["sup1"].to_numpy() >= 0) and np.all(
        preds["sup1"].to_numpy() < 10
    )
    assert np.all(preds["sup3"].to_numpy() > -4.0) and np.all(
        preds["sup3"].to_numpy() < 4.0
    )


def test_embeddings(embeddings):
    for model_name, model_embeddings in embeddings.items():
        assert model_embeddings.shape[0] == 10
        assert model_embeddings.shape[1] == 203
        assert np.abs(model_embeddings[:, 1:].to_numpy().mean()) < 0.1


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

        # Calculate differences and sequence changes
        embeds_with_diffs = embeds_df_sorted.with_columns(
            (pl.col("subsequenceId") - pl.col("subsequenceId").shift(1)).alias(
                "subseq_diff"
            ),
            (pl.col("sequenceId") == pl.col("sequenceId").shift(1)).alias("same_seq"),
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


@pytest.mark.optional
def test_identities(targets, predictions, probabilities, embeddings):
    for model_name, preds in predictions.items():
        print(f"{model_name = }")
        equal = preds.to_numpy() == targets["preds"][model_name].to_numpy()
        mean_equal = np.mean(equal.astype(int))
        if model_name != "model-real-1-best-3-autoregression":
            assert (
                mean_equal == 1.0
            ), f"{model_name} preds are not identical to target: {preds.to_numpy() = } != {targets['preds'][model_name].to_numpy() = }: {equal = }, {mean_equal = }"
        else:
            assert (
                mean_equal < 1.0
            ), f"{model_name} preds are not randomized, {preds.to_numpy() = } == {targets['preds'][model_name].to_numpy() = }: {equal = }, {mean_equal = }"

    for model_name, probs in probabilities.items():
        print(f"{model_name = }")
        all_equal = np.all(probs.to_numpy() == targets["probs"][model_name].to_numpy())
        assert all_equal, f"{model_name} probs are not identical to target"

    for model_name, embeds in embeddings.items():
        print(f"{model_name = }")
        all_equal = np.all(
            embeds.to_numpy() == targets["embeds"][model_name].to_numpy()
        )
        assert all_equal, f"{model_name} embeddings are not identical to target"
