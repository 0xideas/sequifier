import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from torch.utils.data import DataLoader

from sequifier.config.probabilities import PoissonDistributionFloor
from sequifier.config.train_config import (
    BERTSpecModel,
    ModelSpecModel,
    ReplacementDistribution,
    TrainingSpecModel,
    TrainModel,
    load_train_config,
)
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.io.batch import SequifierBatch
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.train import (
    TransformerModel,
    _get_evaluation_loss_mask,
    accumulate_class_counts,
)


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
    assert "stored_context_width" not in dumped
    assert "max_target_offset" not in dumped


def test_poisson_span_masking_samples_at_least_one_token():
    distribution = PoissonDistributionFloor(rate=0.1)

    samples = distribution.sample((1000,), device=torch.device("cpu"))

    assert samples.min().item() >= 1


@pytest.fixture
def model_config(tmp_path):
    """Valid TrainModel config."""
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
        storage_layout=StoredWindowLayout(
            stored_context_width=11, max_target_offset=1, version=2
        ),
        window_view=ModelWindowView(
            context_length=10, objective="causal", target_offset=1
        ),
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
    config_values["model_spec"]["prediction_length"] = (
        model_config.window_view.context_length
    )
    config_values["training_spec"] = _training_spec_kwargs(
        training_objective="bert",
        bert_spec=_bert_spec(),
    )
    config_values["window_view"] = {
        **config_values["window_view"],
        "objective": "bert",
        "target_offset": 0,
    }
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


def _loss_shell_model(
    target_column_types,
    *,
    loss_weights=None,
    class_weights=None,
):
    model = TransformerModel.__new__(TransformerModel)
    model.target_columns = list(target_column_types)
    model.target_column_types = dict(target_column_types)
    model.loss_weights = loss_weights
    model.device = "cpu"
    model.criterion = {}
    model.n_classes = {}

    for col, col_type in target_column_types.items():
        if col_type == "categorical":
            weight = None
            n_classes = 3
            if class_weights is not None and col in class_weights:
                col_class_weights = class_weights[col]
                weight = torch.tensor(col_class_weights, dtype=torch.float32)
                n_classes = len(col_class_weights)
            model.criterion[col] = torch.nn.CrossEntropyLoss(
                reduction="none",
                weight=weight,
            )
            model.n_classes[col] = n_classes
        elif col_type == "real":
            model.criterion[col] = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(col_type)

    return model


class _IdentityScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


class _NoopLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def test_transformer_model_initialization(model, model_config):
    """Expected model layers."""
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


def test_train_model_requires_bert_prediction_length_to_equal_context_length(
    model_config,
):
    config_values = model_config.model_dump()
    config_values["model_spec"]["prediction_length"] = (
        model_config.window_view.context_length - 1
    )
    config_values["training_spec"] = _training_spec_kwargs(
        training_objective="bert",
        bert_spec=_bert_spec(),
    )
    config_values["window_view"] = {
        **config_values["window_view"],
        "objective": "bert",
        "target_offset": 0,
    }

    with pytest.raises(ValidationError, match="prediction_length must be equal"):
        TrainModel(**config_values)


def test_train_model_rejects_mismatched_special_token_ids(model_config):
    config_values = model_config.model_dump()
    config_values["special_token_ids"] = {
        "[unknown]": 10,
        "[other]": 11,
        "[mask]": 12,
    }

    with pytest.raises(ValidationError, match="special_token_ids must match"):
        TrainModel(**config_values)


def test_load_train_config_rejects_mismatched_metadata_special_token_ids(
    tmp_path, model_config
):
    config_path = tmp_path / "train.yaml"
    metadata_path = tmp_path / "metadata.json"
    config_values = model_config.model_dump()
    config_values["project_root"] = str(tmp_path)
    config_values["metadata_config_path"] = metadata_path.name
    storage_layout = config_values.pop("storage_layout")
    config_values.pop("window_view")
    config_values["context_length"] = model_config.window_view.context_length
    config_path.write_text(yaml.safe_dump(config_values))
    metadata_path.write_text(
        json.dumps(
            {
                "split_paths": ["data/train.pt", "data/val.pt"],
                "stored_context_width": storage_layout["stored_context_width"],
                "max_target_offset": storage_layout["max_target_offset"],
                "stored_window_layout_version": storage_layout["version"],
                "column_types": config_values["column_types"],
                "n_classes": config_values["n_classes"],
                "id_maps": config_values["id_maps"],
                "special_token_ids": {
                    "[unknown]": 10,
                    "[other]": 11,
                    "[mask]": 12,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="special_token_ids must match"):
        load_train_config(str(config_path), {}, skip_metadata=False)


def test_load_train_config_defaults_missing_metadata_special_token_ids(
    tmp_path, model_config
):
    config_path = tmp_path / "train.yaml"
    metadata_path = tmp_path / "metadata.json"
    config_values = model_config.model_dump()
    config_values["project_root"] = str(tmp_path)
    config_values["metadata_config_path"] = metadata_path.name
    storage_layout = config_values.pop("storage_layout")
    config_values.pop("window_view")
    config_values["context_length"] = model_config.window_view.context_length
    config_path.write_text(yaml.safe_dump(config_values))
    metadata_path.write_text(
        json.dumps(
            {
                "split_paths": ["data/train.pt", "data/val.pt"],
                "stored_context_width": storage_layout["stored_context_width"],
                "max_target_offset": storage_layout["max_target_offset"],
                "stored_window_layout_version": storage_layout["version"],
                "column_types": config_values["column_types"],
                "n_classes": config_values["n_classes"],
                "id_maps": config_values["id_maps"],
            }
        )
    )

    config = load_train_config(str(config_path), {}, skip_metadata=False)

    assert config.special_token_ids == SPECIAL_TOKEN_IDS.ids_by_label


@pytest.mark.parametrize(
    ("class_share_column", "metadata_overrides", "config_overrides", "match"),
    [
        (
            "missing_col",
            {},
            {},
            "Class-share column 'missing_col' must be a target column",
        ),
        (
            "real_col",
            {},
            {},
            "Class-share column 'real_col' must be a categorical target column",
        ),
        (
            "cat_col",
            {"n_classes": {}},
            {"drop_n_classes": True},
            "Class-share column 'cat_col' has no configured class count",
        ),
        (
            "cat_col",
            {"id_maps": {}},
            {},
            "Class-share column 'cat_col' has no index map for logging",
        ),
    ],
)
def test_load_train_config_validates_class_share_columns_after_metadata_load(
    tmp_path,
    model_config,
    class_share_column,
    metadata_overrides,
    config_overrides,
    match,
):
    config_path = tmp_path / "train.yaml"
    metadata_path = tmp_path / "metadata.json"
    config_values = model_config.model_dump()
    config_values["project_root"] = str(tmp_path)
    config_values["metadata_config_path"] = metadata_path.name
    config_values["training_spec"]["class_share_log_columns"] = [class_share_column]

    storage_layout = config_values.pop("storage_layout")
    config_values.pop("window_view")
    config_values["context_length"] = model_config.window_view.context_length

    if config_overrides.get("drop_n_classes"):
        config_values.pop("n_classes")

    metadata = {
        "split_paths": ["data/train.pt", "data/val.pt"],
        "stored_context_width": storage_layout["stored_context_width"],
        "max_target_offset": storage_layout["max_target_offset"],
        "stored_window_layout_version": storage_layout["version"],
        "column_types": config_values["column_types"],
        "n_classes": model_config.n_classes,
        "id_maps": model_config.id_maps,
        "special_token_ids": SPECIAL_TOKEN_IDS.ids_by_label,
    }
    metadata.update(metadata_overrides)

    config_path.write_text(yaml.safe_dump(config_values))
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=match):
        load_train_config(str(config_path), {}, skip_metadata=False)


def test_forward_train_shapes(model, model_config):
    """forward_train output shapes."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.window_view.context_length

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
    """Inference output shapes."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.window_view.context_length
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
    """Scalar loss tensor."""
    batch_size = model_config.training_spec.batch_size
    seq_len = model_config.window_view.context_length

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


def test_calculate_loss_uses_explicit_target_mask_for_target_columns():
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


def test_calculate_local_loss_components_categorical_target_with_padding():
    model = _loss_shell_model({"cat_col": "categorical"})
    model.n_classes["cat_col"] = 3
    logits = torch.tensor(
        [
            [[2.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0]],
            [[0.0, 0.0, 2.0]],
        ]
    )
    targets = {"cat_col": torch.tensor([[0, 1, 2]])}
    valid_mask = torch.tensor([[True, False, True]])

    loss_sums, token_count = TransformerModel._calculate_local_loss_components(
        model,
        {"cat_col": logits},
        targets,
        valid_mask,
    )
    raw_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        targets["cat_col"].T.reshape(-1),
        reduction="none",
    )

    assert torch.isclose(loss_sums["cat_col"], raw_loss[[0, 2]].sum())
    assert torch.equal(token_count, torch.tensor(2))


def test_calculate_local_loss_components_real_target_with_padding():
    model = _loss_shell_model({"real_col": "real"})
    output = {"real_col": torch.tensor([[[1.0]], [[2.0]], [[4.0]]])}
    targets = {"real_col": torch.tensor([[0.0, 10.0, 1.0]])}
    valid_mask = torch.tensor([[True, False, True]])

    loss_sums, token_count = TransformerModel._calculate_local_loss_components(
        model, output, targets, valid_mask
    )

    assert torch.isclose(loss_sums["real_col"], torch.tensor(10.0))
    assert torch.equal(token_count, torch.tensor(2))


def test_calculate_training_loss_intersects_target_and_bert_masks():
    model = _loss_shell_model({"real_col": "real"})
    output = {"real_col": torch.tensor([[[1.0]], [[2.0]], [[4.0]]])}
    targets = {"real_col": torch.tensor([[0.0, 10.0, 1.0]])}
    metadata = {
        "target_valid_mask": torch.tensor([[True, True, True]]),
        "bert_mask": torch.tensor([[True, False, True]]),
    }

    total_loss, component_losses = TransformerModel._calculate_loss(
        model, output, targets, metadata
    )

    assert torch.isclose(total_loss, torch.tensor(5.0))
    assert torch.isclose(component_losses["real_col"], torch.tensor(5.0))


def test_calculate_training_loss_applies_unequal_target_loss_weights():
    model = _loss_shell_model(
        {"cat_col": "categorical", "real_col": "real"},
        loss_weights={"cat_col": 0.25, "real_col": 2.0},
    )
    model.n_classes["cat_col"] = 3
    output = {
        "cat_col": torch.tensor(
            [
                [[2.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0]],
            ]
        ),
        "real_col": torch.tensor([[[1.0]], [[3.0]]]),
    }
    targets = {
        "cat_col": torch.tensor([[0, 1]]),
        "real_col": torch.tensor([[0.0, 1.0]]),
    }
    metadata = {"target_valid_mask": torch.ones(1, 2, dtype=torch.bool)}

    total_loss, component_losses = TransformerModel._calculate_loss(
        model, output, targets, metadata
    )
    cat_mean = torch.nn.functional.cross_entropy(
        output["cat_col"].reshape(-1, 3),
        targets["cat_col"].T.reshape(-1),
        reduction="mean",
    )
    real_mean = torch.nn.functional.mse_loss(
        output["real_col"].reshape(-1),
        targets["real_col"].T.reshape(-1),
        reduction="mean",
    )

    assert torch.isclose(component_losses["cat_col"], cat_mean * 0.25)
    assert torch.isclose(component_losses["real_col"], real_mean * 2.0)
    assert torch.isclose(total_loss, cat_mean * 0.25 + real_mean * 2.0)


def test_calculate_local_loss_components_keeps_class_weights_inside_criterion():
    class_weights = {"cat_col": [1.0, 4.0, 1.0]}
    model = _loss_shell_model({"cat_col": "categorical"}, class_weights=class_weights)
    logits = torch.tensor(
        [
            [[2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )
    targets = {"cat_col": torch.tensor([[0, 1]])}
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    loss_sums, token_count = TransformerModel._calculate_local_loss_components(
        model,
        {"cat_col": logits},
        targets,
        valid_mask,
    )
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        targets["cat_col"].T.reshape(-1),
        reduction="none",
        weight=torch.tensor(class_weights["cat_col"]),
    ).sum()

    assert torch.isclose(loss_sums["cat_col"], expected)
    assert torch.equal(token_count, torch.tensor(2))


def test_calculate_local_loss_components_zero_selected_tokens_stays_connected():
    model = _loss_shell_model({"real_col": "real"})
    output = {"real_col": torch.ones(3, 1, 1, requires_grad=True)}
    targets = {"real_col": torch.zeros(1, 3)}
    valid_mask = torch.zeros(1, 3, dtype=torch.bool)

    loss_sums, token_count = TransformerModel._calculate_local_loss_components(
        model,
        output,
        targets,
        valid_mask,
    )
    loss_sums["real_col"].backward()

    assert torch.equal(token_count, torch.tensor(0))
    assert torch.equal(loss_sums["real_col"].detach(), torch.tensor(0.0))
    assert torch.equal(output["real_col"].grad, torch.zeros_like(output["real_col"]))


def test_calculate_local_loss_components_raises_on_output_mask_mismatch():
    model = _loss_shell_model({"real_col": "real"})
    output = {"real_col": torch.zeros(3, 1, 1)}
    targets = {"real_col": torch.zeros(1, 3)}
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="Loss/mask size mismatch"):
        TransformerModel._calculate_local_loss_components(
            model,
            output,
            targets,
            valid_mask,
        )


def test_training_and_validation_loss_finalization_share_weighted_sum_count_semantics():
    model = _loss_shell_model(
        {"cat_col": "categorical", "real_col": "real"},
        loss_weights={"cat_col": 0.5, "real_col": 3.0},
    )
    model.n_classes["cat_col"] = 3
    output = {
        "cat_col": torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ]
        ),
        "real_col": torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
    }
    targets = {
        "cat_col": torch.tensor([[0, 1, 2]]),
        "real_col": torch.tensor([[0.0, 0.0, 0.0]]),
    }
    valid_mask = torch.tensor([[True, False, True]])
    metadata = {"target_valid_mask": valid_mask}

    training_loss, training_components = TransformerModel._calculate_loss(
        model, output, targets, metadata
    )
    sums, count = TransformerModel._calculate_loss_components(
        model, output, targets, valid_mask
    )
    finalized_loss, finalized_components = TransformerModel._finalize_loss_components(
        model,
        sums,
        count.double(),
        ["cat_col", "real_col"],
        "test",
    )

    assert torch.isclose(training_loss.double(), finalized_loss)
    assert torch.isclose(
        training_components["cat_col"].double(), finalized_components["cat_col"]
    )
    assert torch.isclose(
        training_components["real_col"].double(), finalized_components["real_col"]
    )


@pytest.mark.parametrize(
    ("data_parallelism", "expected"),
    [
        ("DDP", 7),
        ("FSDP", 7),
    ],
)
def test_gradient_reduction_factor_uses_active_world_size(
    data_parallelism,
    expected,
    monkeypatch,
):
    model = _loss_shell_model({"real_col": "real"})
    model.hparams = SimpleNamespace(
        training_spec=SimpleNamespace(data_parallelism=data_parallelism)
    )
    group = object()
    model._data_parallel_process_group = lambda: group

    monkeypatch.setattr("sequifier.train.dist.is_available", lambda: True)
    monkeypatch.setattr("sequifier.train.dist.is_initialized", lambda: True)

    def fake_get_world_size(group=None):
        assert group is model._data_parallel_process_group()
        return 7

    monkeypatch.setattr("sequifier.train.dist.get_world_size", fake_get_world_size)

    assert TransformerModel._gradient_reduction_factor(model) == expected


def test_calculate_loss_requires_all_configured_targets():
    model = _loss_shell_model(
        {"cat_col": "categorical", "real_col": "real"},
        loss_weights={"cat_col": 1.0, "real_col": 1.0},
    )
    output = {
        "cat_col": torch.zeros(1, 1, 3),
        "real_col": torch.zeros(1, 1, 1),
    }
    targets = {"cat_col": torch.zeros(1, 1, dtype=torch.long)}
    metadata = {"target_valid_mask": torch.ones(1, 1, dtype=torch.bool)}

    with pytest.raises(RuntimeError, match="Missing target columns: \\['real_col'\\]"):
        TransformerModel._calculate_loss(model, output, targets, metadata)


def test_training_loss_enables_fsdp_token_weighting(
    monkeypatch,
):
    model = _loss_shell_model({"real_col": "real"})
    model.hparams = SimpleNamespace(
        training_spec=SimpleNamespace(data_parallelism="FSDP")
    )
    output = {"real_col": torch.zeros(1, 1, 1, requires_grad=True)}
    targets = {"real_col": torch.ones(1, 1)}
    metadata = {"target_valid_mask": torch.ones(1, 1, dtype=torch.bool)}
    process_group = object()
    model._data_parallel_process_group = lambda: process_group

    monkeypatch.setattr("sequifier.train.dist.is_available", lambda: True)
    monkeypatch.setattr("sequifier.train.dist.is_initialized", lambda: True)

    def fake_get_world_size(group=None):
        assert group is process_group
        return 2

    monkeypatch.setattr("sequifier.train.dist.get_world_size", fake_get_world_size)

    reduce_calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        reduce_calls.append((tensor.detach().clone(), op, group))
        tensor.fill_(4)

    monkeypatch.setattr("sequifier.train.dist.all_reduce", fake_all_reduce)

    loss, _ = TransformerModel._calculate_loss(model, output, targets, metadata)
    loss.backward()

    assert reduce_calls
    assert reduce_calls[0][2] is process_group
    assert torch.isclose(loss, torch.tensor(0.5))
    assert torch.allclose(output["real_col"].grad, torch.tensor([[[-1.0]]]))


def test_evaluation_loss_mask_intersects_target_bert_and_sample_masks():
    metadata = {
        "target_valid_mask": torch.tensor(
            [
                [True, True, False],
                [True, True, True],
            ]
        ),
        "bert_mask": torch.tensor(
            [
                [True, False, True],
                [True, True, True],
            ]
        ),
        "sample_valid_mask": torch.tensor([True, False]),
    }

    loss_mask = _get_evaluation_loss_mask(metadata)

    assert torch.equal(
        loss_mask,
        torch.tensor(
            [
                [True, False, False],
                [False, False, False],
            ]
        ),
    )


def _categorical_logits_from_batch_predictions(batch_predictions, n_classes):
    prediction_ids = torch.tensor(batch_predictions, dtype=torch.long).T.contiguous()
    return torch.nn.functional.one_hot(prediction_ids, num_classes=n_classes).float()


def test_accumulate_class_counts_all_samples_valid():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [[0, 1], [1, 2]],
            n_classes=3,
        )
    }
    valid_mask = torch.ones(2, 2, dtype=torch.bool)

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([1, 2, 1]))


def test_accumulate_class_counts_excludes_synthetic_samples():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [
                [0, 1],
                [2, 2],
            ],
            n_classes=3,
        )
    }
    valid_mask = _get_evaluation_loss_mask(
        {
            "target_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "sample_valid_mask": torch.tensor([True, False]),
        }
    )

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([1, 1, 0]))


def test_accumulate_class_counts_excludes_target_padding():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [[0, 2]],
            n_classes=3,
        )
    }
    valid_mask = torch.tensor([[True, False]])

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([1, 0, 0]))


def test_accumulate_class_counts_combines_sample_and_token_masks():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [
                [0, 1],
                [2, 2],
            ],
            n_classes=3,
        )
    }
    valid_mask = _get_evaluation_loss_mask(
        {
            "target_valid_mask": torch.tensor(
                [
                    [True, False],
                    [True, True],
                ]
            ),
            "sample_valid_mask": torch.tensor([True, False]),
        }
    )

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([1, 0, 0]))


def test_accumulate_class_counts_respects_bert_mask():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [
                [0, 1],
                [2, 2],
            ],
            n_classes=3,
        )
    }
    valid_mask = _get_evaluation_loss_mask(
        {
            "target_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "bert_mask": torch.tensor(
                [
                    [False, True],
                    [True, True],
                ]
            ),
            "sample_valid_mask": torch.tensor([True, False]),
        }
    )

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([0, 1, 0]))


def test_accumulate_class_counts_retains_missing_class_slots():
    counts = {"cat_col": torch.zeros(5, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [[3, 3]],
            n_classes=5,
        )
    }
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 5})

    assert torch.equal(counts["cat_col"], torch.tensor([0, 0, 0, 2, 0]))


def test_accumulate_class_counts_all_zero_when_no_positions_are_valid():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [[0, 1]],
            n_classes=3,
        )
    }
    valid_mask = torch.zeros(1, 2, dtype=torch.bool)

    accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})

    assert torch.equal(counts["cat_col"], torch.tensor([0, 0, 0]))


def test_accumulate_class_counts_raises_on_shape_mismatch():
    counts = {"cat_col": torch.zeros(3, dtype=torch.int64)}
    output = {
        "cat_col": _categorical_logits_from_batch_predictions(
            [[0, 1, 2]],
            n_classes=3,
        )
    }
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="Prediction/mask size mismatch"):
        accumulate_class_counts(counts, output, valid_mask, {"cat_col": 3})


def test_calculate_loss_components_aggregate_by_token_count():
    model = TransformerModel.__new__(TransformerModel)
    model.target_column_types = {"real_col": "real"}
    model.criterion = {"real_col": torch.nn.MSELoss(reduction="none")}
    model.loss_weights = {"real_col": 1.0}

    first_sums, first_count = TransformerModel._calculate_loss_components(
        model,
        {"real_col": torch.zeros(1, 1, 1)},
        {"real_col": torch.tensor([[10.0]])},
        torch.ones(1, 1, dtype=torch.bool),
    )
    second_sums, second_count = TransformerModel._calculate_loss_components(
        model,
        {"real_col": torch.zeros(100, 1, 1)},
        {"real_col": torch.ones(1, 100)},
        torch.ones(1, 100, dtype=torch.bool),
    )

    aggregate = (first_sums["real_col"] + second_sums["real_col"]) / (
        first_count + second_count
    )

    assert torch.isclose(aggregate, torch.tensor(200.0 / 101.0, dtype=torch.float64))


class _QueuedEvalModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)

    def forward(self, data, metadata=None, return_logits=False):
        return next(self.outputs)


def _evaluation_shell_model():
    model = TransformerModel.__new__(TransformerModel)
    model.input_columns = ["cat_col"]
    model.target_columns = ["cat_col"]
    model.target_column_types = {"cat_col": "categorical"}
    model.n_classes = {"cat_col": 3}
    model.loss_weights = None
    model.criterion = {"cat_col": torch.nn.CrossEntropyLoss(reduction="none")}
    model.device = "cpu"
    model.class_share_log_columns = ["cat_col"]
    model.index_maps = {  # type: ignore
        "cat_col": {
            0: "[unknown]",
            1: "a",
            2: "b",
        }
    }  # type: ignore
    model.decoder = {"cat_col": torch.nn.Linear(1, 1)}
    model.hparams = SimpleNamespace(
        seed=42,
        training_spec=SimpleNamespace(
            distributed=False,
            layer_autocast=False,
            data_parallelism="none",
            training_objective="causal",
        ),
    )
    return model


def _validation_batch(predictions, sample_valid_mask=None):
    batch_size = len(predictions)
    seq_len = len(predictions[0])
    inputs = {
        "cat_col": torch.zeros(batch_size, seq_len, dtype=torch.long),
    }
    targets = {
        "cat_col": torch.tensor(predictions, dtype=torch.long),
    }
    valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    metadata = {
        "attention_valid_mask": valid_mask.clone(),
        "target_valid_mask": valid_mask,
    }
    if sample_valid_mask is not None:
        metadata["sample_valid_mask"] = torch.tensor(
            sample_valid_mask,
            dtype=torch.bool,
        )
    return SequifierBatch(inputs=inputs, targets=targets, metadata=metadata)


def test_evaluate_returns_class_counts_across_all_validation_batches():
    model = _evaluation_shell_model()
    batches = [
        _validation_batch([[0, 0, 1]]),
        _validation_batch([[2, 2]]),
    ]
    outputs = [
        {"cat_col": _categorical_logits_from_batch_predictions([[0, 0, 1]], 3)},
        {"cat_col": _categorical_logits_from_batch_predictions([[2, 2]], 3)},
    ]
    eval_model = _QueuedEvalModel(outputs)
    valid_loader = DataLoader(batches, batch_size=None)

    total_loss, total_losses, class_counts = TransformerModel._evaluate(
        model,
        valid_loader,
        eval_model,
    )

    assert np.isfinite(total_loss)
    assert np.isfinite(total_losses["cat_col"])
    assert torch.equal(class_counts["cat_col"], torch.tensor([2, 1, 2]))
    assert eval_model.training
    assert model.baseline_loss >= 0


def test_evaluate_excludes_synthetic_final_batch_from_class_counts():
    model = _evaluation_shell_model()
    batches = [
        _validation_batch([[0, 1]]),
        _validation_batch([[2, 2]], sample_valid_mask=[False]),
    ]
    outputs = [
        {"cat_col": _categorical_logits_from_batch_predictions([[0, 1]], 3)},
        {"cat_col": _categorical_logits_from_batch_predictions([[2, 2]], 3)},
    ]
    valid_loader = DataLoader(batches, batch_size=None)

    _, _, class_counts = TransformerModel._evaluate(
        model,
        valid_loader,
        _QueuedEvalModel(outputs),
    )

    assert torch.equal(class_counts["cat_col"], torch.tensor([1, 1, 0]))


def test_calculate_loss_zero_token_training_batch_is_differentiable():
    model = TransformerModel.__new__(TransformerModel)
    model.target_column_types = {"real_col": "real"}
    model.criterion = {"real_col": torch.nn.MSELoss(reduction="none")}
    model.loss_weights = None

    output = {"real_col": torch.ones(3, 1, 1, requires_grad=True)}
    targets = {"real_col": torch.zeros(1, 3)}
    metadata = {
        "attention_valid_mask": torch.zeros(1, 3, dtype=torch.bool),
        "target_valid_mask": torch.zeros(1, 3, dtype=torch.bool),
    }

    total_loss, component_losses = TransformerModel._calculate_loss(
        model, output, targets, metadata
    )
    total_loss.backward()

    assert torch.equal(total_loss.detach(), torch.tensor(0.0))
    assert torch.equal(component_losses["real_col"].detach(), torch.tensor(0.0))
    assert torch.equal(output["real_col"].grad, torch.zeros_like(output["real_col"]))


def _train_epoch_test_batch(model, target_valid_mask):
    seq_len = model.window_view.context_length
    return SequifierBatch(
        inputs={
            "cat_col": torch.ones(1, seq_len, dtype=torch.long),
            "real_col": torch.ones(1, seq_len, dtype=torch.float32),
        },
        targets={
            "cat_col": torch.ones(1, seq_len, dtype=torch.long),
            "real_col": torch.ones(1, seq_len, dtype=torch.float32),
        },
        metadata={
            "attention_valid_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "target_valid_mask": target_valid_mask,
        },
    )


def test_train_epoch_skips_optimizer_and_batch_scheduler_for_empty_accumulation_window(
    model,
):
    model.rank = 0
    model.log_interval = 2
    model.accumulation_steps = 2
    model.scheduler_step_on = "batch"
    model.start_batch = 0
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.1,
    )
    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        step_size=1,
        gamma=0.1,
    )
    seq_len = model.window_view.context_length
    empty_batch = _train_epoch_test_batch(
        model,
        torch.zeros(1, seq_len, dtype=torch.bool),
    )
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    lr_before = model.scheduler.get_last_lr()[0]

    TransformerModel._train_epoch(
        model,
        DataLoader([empty_batch, empty_batch], batch_size=None),
        DataLoader([], batch_size=None),
        epoch=1,
    )

    after = dict(model.named_parameters())
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert len(model.optimizer.state) == 0
    assert model.scheduler.get_last_lr()[0] == lr_before


def test_train_epoch_steps_once_for_mixed_empty_and_nonempty_accumulation_window(
    model,
):
    model.rank = 0
    model.log_interval = 2
    model.accumulation_steps = 2
    model.scheduler_step_on = "batch"
    model.start_batch = 0
    model.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.1,
    )
    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        step_size=1,
        gamma=0.1,
    )
    seq_len = model.window_view.context_length
    empty_batch = _train_epoch_test_batch(
        model,
        torch.zeros(1, seq_len, dtype=torch.bool),
    )
    nonempty_batch = _train_epoch_test_batch(
        model,
        torch.ones(1, seq_len, dtype=torch.bool),
    )
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    TransformerModel._train_epoch(
        model,
        DataLoader([empty_batch, nonempty_batch], batch_size=None),
        DataLoader([], batch_size=None),
        epoch=1,
    )

    after = dict(model.named_parameters())
    assert any(not torch.equal(before[name], after[name]) for name in before)
    assert len(model.optimizer.state) > 0
    assert model.scheduler.last_epoch == 1
    assert model.scheduler.get_last_lr()[0] == pytest.approx(0.001)


def test_train_epoch_divides_backward_loss_by_accumulation_steps(model, monkeypatch):
    model.rank = 0
    model.log_interval = 100
    model.accumulation_steps = 2
    model.scheduler_step_on = "batch"
    model.start_batch = 0
    model.scaler = _IdentityScaler()
    model.logger = _NoopLogger()
    model.save_latest_interval_minutes = None
    model.save_batch_interval_minutes = None
    model.save_batch_interval_minutes_val_loss = False
    model.last_latest_save_time = 0.0
    model.last_batch_save_time = 0.0
    tracked_param = next(model.parameters())
    tracked_value_before = tracked_param.detach().reshape(-1)[0].clone()
    model.optimizer = torch.optim.SGD([tracked_param], lr=0.1)
    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        step_size=1,
        gamma=0.1,
    )
    coefficients = iter([0.2, 0.4])

    def fake_forward(data, metadata=None, return_logits=True):
        return {}

    def fake_calculate_training_loss(output, targets, metadata):
        coefficient = next(coefficients)
        loss = tracked_param.reshape(-1)[0] * coefficient
        loss_sums = {
            target_name: loss.detach()
            for target_name in model._loss_target_names(targets)
        }
        count = torch.tensor(1, dtype=torch.int64)
        return loss, {}, loss_sums, count, count

    monkeypatch.setattr(model, "forward", fake_forward)
    monkeypatch.setattr(
        model,
        "_calculate_training_loss",
        fake_calculate_training_loss,
    )

    seq_len = model.window_view.context_length
    batch = _train_epoch_test_batch(
        model,
        torch.ones(1, seq_len, dtype=torch.bool),
    )

    TransformerModel._train_epoch(
        model,
        DataLoader([batch, batch], batch_size=None),
        DataLoader([], batch_size=None),
        epoch=1,
    )

    tracked_value_after = tracked_param.detach().reshape(-1)[0]

    assert torch.allclose(
        tracked_value_after,
        tracked_value_before - torch.tensor(0.03),
        atol=1e-6,
    )


def test_padding_keys_are_masked(bert_model):
    seq_len = bert_model.window_view.context_length

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
    seq_len = causal_model.window_view.context_length

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
    seq_len = model.context_length

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
    seq_len = model.context_length
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
