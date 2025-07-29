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
def targets(model_names_preds, model_names_probs):
    target_dict = {"preds": {}, "probs": {}}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"
        dtype = (
            {TARGET_VARIABLE_DICT[target_type]: str}
            if target_type == "categorical"
            else None
        )

        file_name = f"sequifier-{model_name}-predictions.csv"
        target = pl.read_csv(
            os.path.join(
                "tests", "resources", "target_outputs", "predictions", file_name
            ),
            schema_overrides=dtype,
        )
        target_dict["preds"][model_name] = target

    for model_name in model_names_probs:
        file_name = f"sequifier-{model_name}-probabilities.csv"
        target = pl.read_csv(
            os.path.join(
                "tests", "resources", "target_outputs", "probabilities", file_name
            )
        )
        target_dict["probs"][model_name] = target

    return target_dict


@pytest.fixture()
def predictions(run_inference, model_names_preds, project_path):
    preds = {}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"

        dtype = (
            {TARGET_VARIABLE_DICT[target_type]: str}
            if target_type == "categorical"
            else None
        )
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "predictions",
            f"sequifier-{model_name}-predictions.csv",
        )
        preds[model_name] = pl.read_csv(
            prediction_path, separator=",", schema_overrides=dtype
        )

    return preds


@pytest.fixture()
def probabilities(run_inference, model_names_probs, project_path):
    probs = {}
    for model_name in model_names_probs:
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "probabilities",
            f"sequifier-{model_name}-probabilities.csv",
        )
        probs[model_name] = pl.read_csv(prediction_path, separator=",")
    return probs


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
    assert preds.shape[1] == 4
    assert np.all(preds["sup1"].to_numpy() >= 0) and np.all(
        preds["sup1"].to_numpy() < 10
    )
    assert np.all(preds["sup3"].to_numpy() > -4.0) and np.all(
        preds["sup3"].to_numpy() < 4.0
    )


def test_identities(targets, predictions, probabilities):
    for model_name, preds in predictions.items():
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
        all_equal = np.all(probs.to_numpy() == targets["probs"][model_name].to_numpy())
        assert all_equal, f"{model_name} probs are not identical to target"
