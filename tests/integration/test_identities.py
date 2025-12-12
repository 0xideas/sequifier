import numpy as np
import pytest


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
