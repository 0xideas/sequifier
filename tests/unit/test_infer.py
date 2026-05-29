from unittest.mock import patch

import numpy as np
import pytest

from sequifier.infer import Inferer, normalize, sample_with_cumsum


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
    """Tests inverse CDF sampling based on logits."""
    # We supply raw logits that translate roughly to probabilities:
    # Row 0: ~[0.1, 0.9] -> Cumsum: ~[0.1, 1.0]
    # Row 1: ~[0.8, 0.2] -> Cumsum: ~[0.8, 1.0]
    logits = np.array([[np.log(0.1), np.log(0.9)], [np.log(0.8), np.log(0.2)]])

    # Mock the random thresholds to strictly control the sampling outcome.
    # Shape must be (batch_size, 1) -> (2, 1)
    mock_rand.return_value = np.array([[0.05], [0.90]])

    sampled_indices = sample_with_cumsum(logits)

    # Row 0: threshold 0.05 < 0.1 (cumsum index 0), so it picks class 0
    # Row 1: threshold 0.90 > 0.8 (cumsum index 0) but < 1.0 (cumsum index 1), so it picks class 1
    np.testing.assert_array_equal(sampled_indices, [0, 1])


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
