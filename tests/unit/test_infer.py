from unittest.mock import patch

import numpy as np
import pytest
import torch

from sequifier.config.infer_config import InfererModel
from sequifier.infer import (
    Inferer,
    get_probs_preds_from_dict,
    normalize,
    sample_with_cumsum,
)


def test_normalize():
    """Tests the softmax normalization of raw logits."""
    # Create raw logits
    # Row 0: exp(0)=1, exp(0)=1 -> probs: 0.5, 0.5
    # Row 1: exp(1)=e, exp(2)=e^2 -> probs: e/(e+e^2), e^2/(e+e^2)
    logits = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.0]])
    outs = {"target_col": logits}

    probs_dict = normalize(outs)

    assert "target_col" in probs_dict
    probs = probs_dict["target_col"]
    assert probs.shape == (3, 2)

    # Assert Row 0 is 50/50
    np.testing.assert_allclose(probs[0], [0.5, 0.5])

    # Assert Row 1 is proportionally correct
    e1, e2 = np.exp(1), np.exp(2)
    np.testing.assert_allclose(probs[1], [e1 / (e1 + e2), e2 / (e1 + e2)])

    # Assert all rows sum to 1.0 (valid probability distribution)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0, 1.0])


@patch("numpy.random.rand")
def test_sample_with_cumsum(mock_rand):
    """Tests inverse CDF sampling with both raw logits and pure probabilities."""
    # Mock the random thresholds to strictly control the sampling outcome.
    mock_rand.return_value = np.array([[0.05], [0.90]])

    # Path 1: Test with logits=True (default)
    raw_logits = np.array([[np.log(0.1), np.log(0.9)], [np.log(0.8), np.log(0.2)]])
    sampled_from_logits = sample_with_cumsum(raw_logits, is_log_probs=True)
    np.testing.assert_array_equal(sampled_from_logits, [0, 1])

    # Path 2: Test with logits=False (pre-normalized probabilities)
    pure_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    sampled_from_probs = sample_with_cumsum(pure_probs, is_log_probs=False)
    np.testing.assert_array_equal(sampled_from_probs, [0, 1])


@pytest.fixture
def mock_inferer():
    """Sets up an Inferer instance with mocked heavy dependencies (ONNX/PyTorch)."""
    with patch("sequifier.infer.onnxruntime.InferenceSession"), patch(
        "sequifier.infer.load_inference_model"
    ):
        inferer = Inferer(
            model_type="generative",
            model_path="dummy_model.onnx",
            project_root=".",
            id_maps={"cat_col": {"A": 2, "B": 3}},
            selected_columns_statistics={"real_col": {"mean": 10.0, "std": 2.0}},
            map_to_id=True,
            categorical_columns=["cat_col"],
            real_columns=["real_col"],
            input_columns=["cat_col", "real_col"],
            target_columns=["cat_col", "real_col"],
            target_column_types={"cat_col": "categorical", "real_col": "real"},
            sample_from_distribution_columns=None,
            infer_with_dropout=False,
            prediction_length=1,
            inference_batch_size=4,
            device="cpu",
            args_config={},
            training_config_path="dummy.yaml",
        )
        return inferer


def test_inferer_invert_normalization(mock_inferer):
    """Tests that normalized real outputs are scaled back correctly."""
    # Normalized values
    values = np.array([-1.0, 0.0, 1.0])

    # Configuration dictates mean = 10.0, std = 2.0
    # Unnormalized logic: (val * (std - 1e-9)) + mean
    # Expected approx: [8.0, 10.0, 12.0]
    unnormalized = mock_inferer.invert_normalization(values, "real_col")

    np.testing.assert_allclose(unnormalized, [8.0, 10.0, 12.0], atol=1e-5)


def test_inferer_expand_to_batch_size(mock_inferer):
    """Tests the array padding logic for strictly sized batches (e.g., ONNX)."""
    mock_inferer.inference_batch_size = 5

    # Input has 2 samples, batch size needs to be 5
    x = np.array([[10], [20]])

    # Should repeat the full array twice [10, 20, 10, 20], then append the remainder [10]
    expanded = mock_inferer.expand_to_batch_size(x)

    assert expanded.shape == (5, 1)
    np.testing.assert_array_equal(expanded, [[10], [20], [10], [20], [10]])


def test_inferer_prepare_inference_batches_pad(mock_inferer):
    """Tests chunking data when padding is requested."""
    mock_inferer.inference_batch_size = 4
    x = {"cat_col": np.array([[1], [2], [3]])}  # 3 total samples

    batches = mock_inferer.prepare_inference_batches(x, pad_to_batch_size=True)

    assert len(batches) == 1
    # Check that the 3 samples were padded up to the target batch size of 4
    assert batches[0]["cat_col"].shape == (4, 1)
    np.testing.assert_array_equal(batches[0]["cat_col"], [[1], [2], [3], [1]])


def test_inferer_prepare_inference_batches_no_pad(mock_inferer):
    """Tests chunking data without padding."""
    mock_inferer.inference_batch_size = 4
    x = {"cat_col": np.array([[1], [2], [3]])}  # 3 total samples

    batches = mock_inferer.prepare_inference_batches(x, pad_to_batch_size=False)

    assert len(batches) == 1
    # Check that it retained its original short size
    assert batches[0]["cat_col"].shape == (3, 1)


def test_inferer_prepare_inference_batches_split(mock_inferer):
    """Tests chunking data into multiple separated batches."""
    mock_inferer.inference_batch_size = 2
    x = {"cat_col": np.array([[1], [2], [3], [4], [5]])}  # 5 total samples

    batches = mock_inferer.prepare_inference_batches(x, pad_to_batch_size=False)

    assert len(batches) == 3
    assert batches[0]["cat_col"].shape == (2, 1)
    assert batches[1]["cat_col"].shape == (2, 1)
    assert batches[2]["cat_col"].shape == (1, 1)  # The remainder
    np.testing.assert_array_equal(batches[2]["cat_col"], [[5]])


# ==========================================
# Test Autoregressive Tensor Inference
# ==========================================


@pytest.fixture
def ar_config():
    """Provides an actual InfererModel configuration for autoregressive inference."""
    return InfererModel(
        project_root=".",
        metadata_config_path="dummy.json",
        model_path="dummy.onnx",
        model_type="generative",
        training_objective="causal",
        data_path="tests/unit/data/empty.parquet",
        input_columns=["target_col"],
        categorical_columns=[],
        real_columns=["target_col"],
        target_columns=["target_col"],
        column_types={"target_col": "float64"},
        target_column_types={"target_col": "real"},
        seed=42,
        device="cpu",
        prediction_length=1,
        seq_length=3,
        inference_batch_size=2,
        output_probabilities=False,
        map_to_id=False,  # Set to False to bypass ID mapping requirements
        autoregression=True,
        autoregression_total_steps=1,
    )


@pytest.fixture
def ar_inferer(ar_config):
    """Sets up an actual Inferer instance with mocked heavy dependencies."""
    with patch("sequifier.infer.onnxruntime.InferenceSession"), patch(
        "sequifier.infer.load_inference_model"
    ):
        return Inferer(
            model_type=ar_config.model_type,
            model_path=ar_config.model_path,
            project_root=ar_config.project_root,
            id_maps=None,
            selected_columns_statistics={"target_col": {"mean": 0.0, "std": 1.0}},
            map_to_id=ar_config.map_to_id,
            categorical_columns=ar_config.categorical_columns,
            real_columns=ar_config.real_columns,
            input_columns=ar_config.input_columns,
            target_columns=ar_config.target_columns,
            target_column_types=ar_config.target_column_types,
            sample_from_distribution_columns=ar_config.sample_from_distribution_columns,
            infer_with_dropout=ar_config.infer_with_dropout,
            prediction_length=ar_config.prediction_length,
            inference_batch_size=ar_config.inference_batch_size,
            device=ar_config.device,
            args_config={},
            training_config_path=ar_config.training_config_path,
        )


def test_get_probs_preds_from_dict_shifting_and_looping(ar_config, ar_inferer):
    """
    Tests that the autoregressive loop calls the model the correct number of times
    and accurately shifts the input tensor to append the latest prediction.
    """
    initial_data = {"target_col": torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])}

    # Patch the method on the actual instance
    with patch.object(ar_inferer, "infer_generative") as mock_infer:
        # Step 0: Predicts 4.0 for row 0, 40.0 for row 1
        # Step 1: Predicts 5.0 for row 0, 50.0 for row 1
        mock_infer.side_effect = [
            {"target_col": np.array([[4.0], [40.0]])},
            {"target_col": np.array([[5.0], [50.0]])},
        ]

        total_steps = 2
        probs, preds = get_probs_preds_from_dict(
            ar_config, ar_inferer, initial_data, total_steps=total_steps
        )

        # 1. Verify Loop Count
        assert mock_infer.call_count == 2

        # 2. Verify Tensor Shifting
        expected_shifted_x = {
            "target_col": np.array([[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]])
        }
        second_call_args = mock_infer.call_args_list[1][0][0]

        np.testing.assert_array_equal(
            second_call_args["target_col"], expected_shifted_x["target_col"]
        )

        # 3. Verify Output Reshaping
        assert "target_col" in preds
        np.testing.assert_array_equal(preds["target_col"], [4.0, 5.0, 40.0, 50.0])


@patch("sequifier.infer.sample_with_cumsum")
def test_get_probs_preds_from_dict_with_probabilities(
    mock_sample, ar_config, ar_inferer
):
    """
    Tests the probability branching logic. When output_probabilities=True,
    `infer_generative` normalizes outputs, and passes them as probabilities
    to the second call which triggers sampling with logits=False.
    """
    # 1. Override the config and inferer to treat the column as categorical
    ar_config.target_column_types["target_col"] = "categorical"
    ar_inferer.target_column_types["target_col"] = "categorical"

    # 2. Force it to route to the sampling branch
    ar_config.output_probabilities = True
    ar_config.sample_from_distribution_columns = ["target_col"]
    ar_inferer.sample_from_distribution_columns = ["target_col"]

    initial_data = {"target_col": torch.tensor([[1.0, 2.0]])}

    # Raw model output (Logits)
    dummy_logits = {"target_col": np.array([[np.log(0.2), np.log(0.8)]])}

    # What we want our mock sample_with_cumsum to return
    mock_sample.return_value = np.array([1])

    # Mock the *inner* backend call, allowing infer_generative to execute its actual logic
    with patch.object(
        ar_inferer, "adjust_and_infer_generative", return_value=dummy_logits
    ) as mock_adjust:
        probs, preds = get_probs_preds_from_dict(
            ar_config, ar_inferer, initial_data, total_steps=1
        )

        assert mock_adjust.call_count == 1

        # Verify sample_with_cumsum was called correctly during the second pass
        mock_sample.assert_called_once()
        args, kwargs = mock_sample.call_args

        # 1. Assert it was passed the normalized probabilities, NOT the raw logits
        np.testing.assert_allclose(args[0], [[0.2, 0.8]])

        # 2. Assert the new flag was toggled correctly based on (probs is None)
        assert kwargs.get("is_log_probs") is False

        # Verify final outputs
        assert probs is not None
        np.testing.assert_allclose(probs["target_col"], [[0.2, 0.8]])
        np.testing.assert_array_equal(preds["target_col"], [1])


def test_get_probs_preds_from_dict_ignores_unselected_columns(ar_config, ar_inferer):
    """
    Tests that columns not explicitly defined in `config.input_columns`
    are filtered out before inference.
    """
    initial_data = {
        "target_col": torch.tensor([[1.0, 2.0]]),
        "ignored_col": torch.tensor([[9.0, 9.0]]),
    }

    with patch.object(ar_inferer, "infer_generative") as mock_infer:
        mock_infer.return_value = {"target_col": np.array([[3.0]])}

        _, _ = get_probs_preds_from_dict(
            ar_config, ar_inferer, initial_data, total_steps=1
        )

        first_call_args = mock_infer.call_args_list[0][0][0]

        assert "target_col" in first_call_args
        assert "ignored_col" not in first_call_args
