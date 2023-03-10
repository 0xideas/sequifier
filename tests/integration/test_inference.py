import json
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def predictions(run_inference, project_path):
    prediction_path = os.path.join(
        project_path, "outputs", "predictions", "sequifier-default-best_predictions.csv"
    )
    preds = pd.read_csv(
        prediction_path, sep=",", decimal=".", index_col=None
    ).values.flatten()
    return preds


@pytest.fixture()
def probabilities(run_inference, project_path):
    prediction_path = os.path.join(
        project_path,
        "outputs",
        "probabilities",
        "sequifier-default-best_probabilities.csv",
    )
    probs = pd.read_csv(prediction_path, sep=",", decimal=".", index_col=None)
    return probs


def test_predictions(predictions):
    valid_values = np.arange(100, 130)
    assert np.all([v in valid_values for v in predictions])


def test_probabilities(probabilities):
    assert probabilities.shape[1] == 31

    np.testing.assert_almost_equal(
        probabilities.sum(1), np.ones(probabilities.shape[0]), decimal=5
    )
