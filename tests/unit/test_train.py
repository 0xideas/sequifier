import copy

import pytest
import torch
from pydantic import ValidationError

from sequifier.config.probabilities import PoissonDistributionFloor
from sequifier.config.train_config import (
    BERTSpecModel,
    ModelSpecModel,
    ReplacementDistribution,
    TrainingSpecModel,
    TrainModel,
)
from sequifier.train import TransformerModel


def _training_spec_kwargs(**overrides):
    values = {
        "training_objective": "causal",
        "device": "cpu",
        "epochs": 1,
        "save_interval_epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "criterion": {"cat_col": "CrossEntropyLoss", "real_col": "MSELoss"},
        "optimizer": {"name": "Adam"},
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.1},
        "loss_weights": {"cat_col": 1.0, "real_col": 1.0},
    }
    values.update(overrides)
    return values


def _bert_spec():
    return BERTSpecModel(
        masking_probability=0.5,
        replacement_distribution={
            "masked": 1.0,
            "random": 0.0,
            "identical": 0.0,
        },
        span_masking={"type": "GeometricDistribution", "p": 1.0},
    )


def test_replacement_distribution_allows_zero_probabilities():
    distribution = ReplacementDistribution(masked=1.0, random=0.0, identical=0.0)

    assert distribution.masked == 1.0
    assert distribution.random == 0.0
    assert distribution.identical == 0.0


def test_training_spec_model_requires_bert_spec_for_bert_objective():
    with pytest.raises(ValidationError, match="BERT hyperparameters must be set"):
        TrainingSpecModel(**_training_spec_kwargs(training_objective="bert"))


def test_training_spec_model_rejects_bert_spec_for_causal_objective():
    with pytest.raises(
        ValidationError,
        match="BERT hyperparameters should only be configured",
    ):
        TrainingSpecModel(**_training_spec_kwargs(bert_spec=_bert_spec()))


def test_training_spec_model_dump_excludes_runtime_offsets():
    training_spec = TrainingSpecModel(**_training_spec_kwargs())

    dumped = training_spec.model_dump()

    assert "data_offset" not in dumped
    assert "target_offset" not in dumped
    assert TrainingSpecModel(**dumped).target_offset == 0


def test_poisson_span_masking_samples_at_least_one_token():
    distribution = PoissonDistributionFloor(rate=0.1)

    samples = distribution.sample((1000,), device=torch.device("cpu"))

    assert samples.min().item() >= 1


@pytest.fixture
def model_config(tmp_path):
    """Creates a valid TrainModel configuration for testing."""
    project_root = str(tmp_path)

    # Ensure necessary directories exist to avoid init errors (logging)
    (tmp_path / "logs").mkdir(exist_ok=True)

    model_spec = ModelSpecModel(
        initial_embedding_dim=16,
        dim_model=16,
        n_head=4,
        dim_feedforward=32,
        num_layers=2,
        prediction_length=1,
        # Embedding dims must sum to dim_model (15 + 1 = 16)
        feature_embedding_dims={"cat_col": 15, "real_col": 1},
    )

    training_spec = TrainingSpecModel(
        training_objective="causal",
        device="cpu",
        epochs=1,
        save_interval_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        criterion={"cat_col": "CrossEntropyLoss", "real_col": "MSELoss"},
        optimizer={"name": "Adam"},
        scheduler={"name": "StepLR", "step_size": 1, "gamma": 0.1},
        loss_weights={"cat_col": 1.0, "real_col": 1.0},
    )

    config = TrainModel(
        project_root=project_root,
        model_name="unit-test-model",
        metadata_config_path="metadata.json",  # Dummy path
        training_data_path="data/train.pt",  # Dummy path
        validation_data_path="data/val.pt",  # Dummy path
        input_columns=["cat_col", "real_col"],
        target_columns=["cat_col", "real_col"],
        target_column_types={"cat_col": "categorical", "real_col": "real"},
        column_types={"cat_col": "int64", "real_col": "float64"},
        categorical_columns=["cat_col"],
        real_columns=["real_col"],
        # id_maps is needed for constructing index_maps in model init
        id_maps={"cat_col": {"a": 1, "b": 2, "c": 3, "d": 4}},
        n_classes={"cat_col": 5},  # 0 + 4 classes
        seq_length=10,
        inference_batch_size=4,
        seed=42,
        export_generative_model=True,
        export_embedding_model=False,
        model_spec=model_spec,
        training_spec=training_spec,
    )
    return config


@pytest.fixture
def model(model_config):
    """Instantiates the TransformerModel with the mock config."""
    return TransformerModel(model_config)


@pytest.fixture
def causal_model(model_config):
    config = copy.deepcopy(model_config)
    config.training_spec.training_objective = "causal"
    return TransformerModel(config)


@pytest.fixture
def bert_model(model_config):
    config_values = model_config.model_dump()
    config_values["model_spec"]["prediction_length"] = model_config.seq_length
    config_values["training_spec"] = _training_spec_kwargs(
        training_objective="bert",
        bert_spec=_bert_spec(),
    )
    config = TrainModel(**config_values)
    return TransformerModel(config)


def _all_valid_metadata(batch_size, seq_len, device=None):
    valid_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.bool,
        device=device,
    )
    return {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }


def test_transformer_model_initialization(model, model_config):
    """Tests that the model initializes with the correct layers."""
    # Check if encoder dicts were created
    assert "cat_col" in model.encoder
    assert model.pos_encoder is None or "cat_col" in model.pos_encoder

    # Check decoder existence
    assert "cat_col" in model.decoder
    assert "real_col" in model.decoder

    # Check embedding sizes
    assert model.dim_model == 16
    assert model.encoder["cat_col"].embedding_dim == 15
    # Real column 'real_col' has feature_embedding_dims=1, but it goes through a Linear layer
    # if dim > 1, or direct if dim == 1. In setup:
    # if feature_embedding_dims[col] > 1 -> Linear
    # else -> checked to be 1, then no encoder added to self.encoder for 'direct' columns?
    # Let's check logic in train.py:
    #   if self.feature_embedding_dims[col] > 1: ... self.encoder[col] = nn.Linear(...)
    #   else: self.real_columns_direct.append(col)
    # Since we set real_col dim to 1, it should NOT be in model.encoder
    if model.feature_embedding_dims["real_col"] == 1:
        assert "real_col" not in model.encoder
    else:
        assert "real_col" in model.encoder


def test_train_model_requires_bert_prediction_length_to_equal_seq_length(model_config):
    config_values = model_config.model_dump()
    config_values["model_spec"]["prediction_length"] = model_config.seq_length - 1
    config_values["training_spec"] = _training_spec_kwargs(
        training_objective="bert",
        bert_spec=_bert_spec(),
    )

    with pytest.raises(ValidationError, match="prediction_length must be equal"):
        TrainModel(**config_values)


def test_forward_train_shapes(model, model_config):
    """Tests the output shapes of the forward_train method."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.seq_length

    # Create dummy inputs
    # Categorical: (batch, seq_len) integers
    x_cat = torch.randint(0, model_config.n_classes["cat_col"], (batch_size, seq_len))
    # Real: (batch, seq_len) floats
    x_real = torch.randn(batch_size, seq_len)

    src = {"cat_col": x_cat, "real_col": x_real}
    metadata = _all_valid_metadata(batch_size, seq_len, device=x_cat.device)

    # forward_train returns a dict of tensors
    outputs = model.forward_train(src, metadata)

    assert "cat_col" in outputs
    assert "real_col" in outputs

    # Expected output shape for training: (seq_len, batch_size, n_classes_or_1)
    # Note: PyTorch Transformer default is (S, N, E) unless batch_first=True.
    # Sequifier seems to use default, so (S, B, OutputDim)

    out_cat = outputs["cat_col"]
    assert out_cat.shape == (seq_len, batch_size, model_config.n_classes["cat_col"])

    out_real = outputs["real_col"]
    assert out_real.shape == (seq_len, batch_size, 1)


def test_forward_inference_shapes(model, model_config):
    """Tests the output shapes of the forward (inference) method."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.seq_length
    prediction_length = model_config.model_spec.prediction_length  # 1

    x_cat = torch.randint(0, model_config.n_classes["cat_col"], (batch_size, seq_len))
    x_real = torch.randn(batch_size, seq_len)
    src = {"cat_col": x_cat, "real_col": x_real}
    metadata = _all_valid_metadata(batch_size, seq_len, device=x_cat.device)

    # forward returns predictions for the *last* prediction_length tokens
    # And applies softmax to categorical outputs
    outputs = model.forward(src, metadata)

    # Expected shape: (prediction_length, batch_size, n_classes_or_1)
    # If prediction_length is 1, dim 0 is size 1.

    out_cat = outputs["cat_col"]
    assert out_cat.shape == (
        prediction_length,
        batch_size,
        model_config.n_classes["cat_col"],
    )

    # Check if Softmax/LogSoftmax was applied (values should be <= 0 for LogSoftmax)
    # Sequifier uses LogSoftmax for categorical
    assert (out_cat <= 0).all()

    out_real = outputs["real_col"]
    assert out_real.shape == (prediction_length, batch_size, 1)


def test_calculate_loss(model, model_config):
    """Tests that loss calculation returns a scalar tensor."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.seq_length

    # Inputs
    x_cat = torch.randint(0, model_config.n_classes["cat_col"], (batch_size, seq_len))
    x_real = torch.randn(batch_size, seq_len)
    src = {"cat_col": x_cat, "real_col": x_real}

    # Targets (must match sequence length for training)
    y_cat = torch.randint(0, model_config.n_classes["cat_col"], (batch_size, seq_len))
    y_real = torch.randn(batch_size, seq_len)
    targets = {"cat_col": y_cat, "real_col": y_real}
    metadata = _all_valid_metadata(batch_size, seq_len, device=x_cat.device)

    # Run forward pass
    outputs = model.forward_train(src, metadata)

    # Calculate loss
    total_loss, component_losses = model._calculate_loss(outputs, targets, metadata)

    # Assertions
    assert total_loss.dim() == 0  # Scalar
    assert total_loss.item() > 0  # Valid loss value
    assert "cat_col" in component_losses
    assert "real_col" in component_losses


def test_calculate_loss_uses_explicit_target_mask_for_real_zero_targets():
    model = TransformerModel.__new__(TransformerModel)
    model.target_column_types = {"real_col": "real"}
    model.criterion = {"real_col": torch.nn.MSELoss(reduction="none")}
    model.loss_weights = None
    model.categorical_columns = []
    outputs = {
        "real_col": torch.tensor(
            [
                [[1.0]],
                [[2.0]],
                [[1.0]],
            ]
        )
    }
    targets = {
        "real_col": torch.tensor([[0.0, 0.0, 2.0]]),
    }
    metadata = {
        "attention_valid_mask": torch.tensor([[True, True, True]]),
        "target_valid_mask": torch.tensor([[True, True, True]]),
    }

    total_loss, component_losses = TransformerModel._calculate_loss(
        model, outputs, targets, metadata
    )

    assert torch.isclose(total_loss, torch.tensor(2.0))
    assert torch.isclose(component_losses["real_col"], torch.tensor(2.0))


def test_calculate_loss_uses_target_columns_for_fallback_mask_inference():
    model = TransformerModel.__new__(TransformerModel)
    model.target_column_types = {"real_target": "real"}
    model.criterion = {"real_target": torch.nn.MSELoss(reduction="none")}
    model.loss_weights = None
    model.categorical_columns = ["context_cat"]
    outputs = {
        "real_target": torch.tensor(
            [
                [[10.0]],
                [[20.0]],
                [[3.0]],
            ]
        )
    }
    targets = {
        "real_target": torch.tensor([[0.0, 0.0, 2.0]]),
    }
    metadata = {
        "attention_valid_mask": torch.tensor([[False, False, True]]),
        "target_valid_mask": torch.tensor([[False, False, True]]),
    }

    total_loss, component_losses = TransformerModel._calculate_loss(
        model, outputs, targets, metadata
    )

    assert torch.isclose(total_loss, torch.tensor(1.0))
    assert torch.isclose(component_losses["real_target"], torch.tensor(1.0))


def test_padding_keys_are_masked(bert_model):
    seq_len = bert_model.seq_length

    valid_mask = torch.ones(
        2,
        seq_len,
        dtype=torch.bool,
        device=bert_model.src_mask.device,
    )

    valid_mask[0, :2] = False
    valid_mask[1, :1] = False

    attn_mask = bert_model._build_attention_mask(valid_mask, dtype=torch.float32)

    assert attn_mask.shape == (2, 1, seq_len, seq_len)

    # Batch 0: keys 0 and 1 are padding.
    assert torch.all(attn_mask[0, :, :, 0] < -1e20)
    assert torch.all(attn_mask[0, :, :, 1] < -1e20)

    # Batch 0: keys 2 onward are not padding-masked.
    assert torch.all(attn_mask[0, :, :, 2:] > -1e20)

    # Batch 1: key 0 is padding.
    assert torch.all(attn_mask[1, :, :, 0] < -1e20)
    assert torch.all(attn_mask[1, :, :, 1:] > -1e20)


def test_causal_and_padding_masks_are_combined(causal_model):
    seq_len = causal_model.seq_length

    valid_mask = torch.ones(
        1,
        seq_len,
        dtype=torch.bool,
        device=causal_model.src_mask.device,
    )

    valid_mask[0, 0] = False

    attn_mask = causal_model._build_attention_mask(
        valid_mask,
        dtype=torch.float32,
    )

    assert attn_mask.shape == (1, 1, seq_len, seq_len)

    # Padding key 0 is masked for every query.
    assert torch.all(attn_mask[0, :, :, 0] < -1e20)

    # Causal future positions are masked.
    assert attn_mask[0, 0, 1, 2] < -1e20
    assert attn_mask[0, 0, 1, seq_len - 1] < -1e20

    # A past valid key should not be masked by causal masking.
    assert attn_mask[0, 0, 1, 1] > -1e20
    assert attn_mask[0, 0, 2, 1] > -1e20
    assert attn_mask[0, 0, seq_len - 1, seq_len - 1] > -1e20


@pytest.fixture
def batch(model):
    seq_len = model.seq_length

    return {
        "cat_col": torch.tensor(
            [
                [0, 0] + [1] * (seq_len - 2),
                [0] + [2] * (seq_len - 1),
            ],
            dtype=torch.long,
            device=model.src_mask.device,
        ),
        "real_col": torch.tensor(
            [
                [0.0, 0.0] + [0.5] * (seq_len - 2),
                [0.0] + [1.5] * (seq_len - 1),
            ],
            dtype=torch.float32,
            device=model.src_mask.device,
        ),
    }


@pytest.fixture
def batch_metadata(model):
    seq_len = model.seq_length
    valid_mask = torch.ones(
        2,
        seq_len,
        dtype=torch.bool,
        device=model.src_mask.device,
    )
    valid_mask[0, :2] = False
    valid_mask[1, :1] = False

    return {
        "attention_valid_mask": valid_mask,
        "target_valid_mask": valid_mask.clone(),
    }


def test_forward_no_nan_with_padding(model, batch, batch_metadata):
    out = model.forward_train(batch, batch_metadata)

    for tensor in out.values():
        assert torch.isfinite(tensor).all()
