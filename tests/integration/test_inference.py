import os

import numpy as np
import polars as pl
import pytest

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}


@pytest.fixture()
def predictions(run_inference, project_path):
    preds = {"categorical": {}, "real": {}}
    model_names = [
        f"model-{variant}-{model_number}-best-3"
        for variant in ["categorical", "real"]
        for model_number in [1, 3, 5, 50]
    ]
    model_names += [
        "model-categorical-multitarget-5-best-3",
        "model-real-1-best-3-autoregression",
    ]
    for model_name in model_names:
        target_type = "categorical" if "categorical" in model_name else "real"
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "predictions",
            f"sequifier-{model_name}-predictions.csv",
        )
        variant = model_name.split("-")[1]
        dtype = (
            {TARGET_VARIABLE_DICT[target_type]: str}
            if target_type == "categorical"
            else None
        )
        preds[variant][model_name] = pl.read_csv(
            prediction_path, separator=",", dtypes=dtype
        )

    return preds


@pytest.fixture()
def probabilities(run_inference, project_path):
    probs = {}
    for model_number in [1, 3, 5, 50]:
        model_name = f"model-categorical-{model_number}"
        prediction_path = os.path.join(
            project_path,
            "outputs",
            "probabilities",
            f"sequifier-{model_name}-best-3-itemId-probabilities.csv",
        )
        probs[model_name] = pl.read_csv(prediction_path, separator=",")
    return probs


def test_predictions_real(predictions):
    for model_name, model_predictions in predictions["real"].items():
        assert np.all(
            [
                v > -10.0 and v < 10.0
                for v in model_predictions[TARGET_VARIABLE_DICT["real"]].to_numpy()
            ]
        )


def test_predictions_cat(predictions):
    valid_values = [str(x) for x in np.arange(100, 130)] + ["unknown"]
    for model_name, model_predictions in predictions["categorical"].items():
        assert np.all(
            [
                v in valid_values
                for v in model_predictions[
                    TARGET_VARIABLE_DICT["categorical"]
                ].to_numpy()
            ]
        ), model_predictions


def test_probabilities(probabilities):
    for model_name, model_probabilities in probabilities.items():
        assert model_probabilities.shape[1] == 31

        np.testing.assert_almost_equal(
            model_probabilities.sum(1),
            np.ones(model_probabilities.shape[0]),
            decimal=5,
        )


def test_multi_pred(predictions):
    preds = predictions["categorical"]["model-categorical-multitarget-5-best-3"]

    assert preds.shape[0] > 0
    assert preds.shape[1] == 4
    assert np.all(preds["sup1"].to_numpy() >= 0) and np.all(
        preds["sup1"].to_numpy() < 10
    )
    assert np.all(preds["sup3"].to_numpy() > -4.0) and np.all(
        preds["sup3"].to_numpy() < 4.0
    )
