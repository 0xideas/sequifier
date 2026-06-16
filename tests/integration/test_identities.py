import numpy as np
import pytest


def _frame_mismatches(actual, expected, *, rtol=1e-5, atol=1e-6):
    mismatches = []
    for column in actual.columns:
        actual_values = actual[column].to_numpy()
        expected_values = expected[column].to_numpy()
        if np.issubdtype(actual_values.dtype, np.floating):
            try:
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as exc:
                mismatches.append((column, str(exc)))
        elif not np.array_equal(actual_values, expected_values):
            mismatches.append(
                (
                    column,
                    f"{actual_values = } != {expected_values = }",
                )
            )
    return mismatches


@pytest.mark.optional
def test_identities(targets, predictions, probabilities, embeddings):
    failed_models = []
    for model_name, preds in predictions.items():
        mismatches = _frame_mismatches(preds, targets["preds"][model_name])
        if model_name != "model-real-1-best-3-autoregression":
            if mismatches:
                failed_models.append(
                    (
                        model_name,
                        mismatches,
                        f"{model_name} preds differ from target: {mismatches = }",
                    )
                )
        else:
            equal = preds.to_numpy() == targets["preds"][model_name].to_numpy()
            mean_equal = np.mean(equal.astype(int))
            if mean_equal == 1.0:
                failed_models.append(
                    (
                        model_name,
                        equal,
                        f"{model_name} preds are not randomized, {preds.to_numpy() = } == {targets['preds'][model_name].to_numpy() = }: {equal = }, {mean_equal = }",
                    )
                )

    for model_name, probs in probabilities.items():
        mismatches = _frame_mismatches(probs, targets["probs"][model_name])
        if mismatches:
            failed_models.append(
                (
                    model_name,
                    mismatches,
                    f"{model_name} probs differ from target: {mismatches = }",
                )
            )

    for model_name, embeds in embeddings.items():
        mismatches = _frame_mismatches(embeds, targets["embeds"][model_name])
        if mismatches:
            failed_models.append(
                (
                    model_name,
                    mismatches,
                    f"{model_name} embeddings differ from target: {mismatches = }",
                )
            )

    if len(failed_models):
        print(failed_models)
        failed_models_subset = [(model, message) for model, _, message in failed_models]
        assert (
            len(failed_models) == 0
        ), f"{len(failed_models) = } - {failed_models_subset}"
