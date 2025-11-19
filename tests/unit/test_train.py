import pytest
import torch

from sequifier.config.train_config import ModelSpecModel, TrainingSpecModel, TrainModel
from sequifier.train import TransformerModel


@pytest.fixture
def model_config(tmp_path):
    """Creates a valid TrainModel configuration for testing."""
    project_root = str(tmp_path)

    # Ensure necessary directories exist to avoid init errors (logging)
    (tmp_path / "logs").mkdir(exist_ok=True)

    model_spec = ModelSpecModel(
        dim_model=16,
        n_head=4,
        dim_feedforward=32,
        num_layers=2,
        prediction_length=1,
        # Embedding dims must sum to dim_model (15 + 1 = 16)
        feature_embedding_dims={"cat_col": 15, "real_col": 1},
    )

    training_spec = TrainingSpecModel(
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


def test_transformer_model_initialization(model, model_config):
    """Tests that the model initializes with the correct layers."""
    # Check if encoder dicts were created
    assert "cat_col" in model.encoder
    assert "cat_col" in model.pos_encoder

    # Check decoder existence
    assert "cat_col" in model.decoder
    assert "real_col" in model.decoder

    # Check embedding sizes
    assert model.embedding_size == 16
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

    # forward_train returns a dict of tensors
    outputs = model.forward_train(src)

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

    # forward returns predictions for the *last* prediction_length tokens
    # And applies softmax to categorical outputs
    outputs = model.forward(src)

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

    # Run forward pass
    outputs = model.forward_train(src)

    # Calculate loss
    total_loss, component_losses = model._calculate_loss(outputs, targets)

    # Assertions
    assert total_loss.dim() == 0  # Scalar
    assert total_loss.item() > 0  # Valid loss value
    assert "cat_col" in component_losses
    assert "real_col" in component_losses
