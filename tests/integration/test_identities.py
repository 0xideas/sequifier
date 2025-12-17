import numpy as np
import pytest


@pytest.mark.optional
def test_identities(targets, predictions, probabilities, embeddings):
    failed_models = []
    for model_name, preds in predictions.items():
        equal = preds.to_numpy() == targets["preds"][model_name].to_numpy()
        mean_equal = np.mean(equal.astype(int))
        if model_name != "model-real-1-best-3-autoregression":
            if not mean_equal == 1.0:
                failed_models.append(
                    (
                        model_name,
                        equal,
                        f"{model_name} preds are not identical to target: {preds.to_numpy() = } != {targets['preds'][model_name].to_numpy() = }: {equal = }, {mean_equal = }",
                    )
                )
        else:
            if mean_equal == 1.0:
                failed_models.append(
                    (
                        model_name,
                        equal,
                        f"{model_name} preds are not randomized, {preds.to_numpy() = } == {targets['preds'][model_name].to_numpy() = }: {equal = }, {mean_equal = }",
                    )
                )

    for model_name, probs in probabilities.items():
        equal = probs.to_numpy() == targets["probs"][model_name].to_numpy()
        all_equal = np.all(equal)
        if not all_equal:
            failed_models.append(
                (model_name, equal, f"{model_name} probs are not identical to target")
            )

    for model_name, embeds in embeddings.items():
        equal = embeds.to_numpy() == targets["embeds"][model_name].to_numpy()
        all_equal = np.all(equal)
        if not all_equal:
            failed_models.append(
                (
                    model_name,
                    equal,
                    f"{model_name} embeddings are not identical to target",
                )
            )

    if len(failed_models):
        print(failed_models)
        failed_models_subset = [(model, message) for model, _, message in failed_models]
        assert (
            len(failed_models) == 0
        ), f"{len(failed_models) = } - {failed_models_subset}"
