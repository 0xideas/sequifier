import os

import numpy as np
import onnxruntime
import torch

from sequifier.config.train_config import ModelSpecModel, TrainingSpecModel, TrainModel
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.train import TransformerModel


def test_bert_onnx_export_accepts_attention_valid_mask(tmp_path):
    project_root = str(tmp_path)
    (tmp_path / "logs").mkdir()

    context_length = 4
    inference_batch_size = 2
    config = TrainModel(
        project_root=project_root,
        model_name="bert-onnx-mask",
        metadata_config_path="metadata.json",
        training_data_path="data/train.pt",
        validation_data_path="data/val.pt",
        input_columns=["cat_col", "real_col"],
        target_columns=["cat_col", "real_col"],
        target_column_types={"cat_col": "categorical", "real_col": "real"},
        column_types={"cat_col": "int64", "real_col": "float64"},
        categorical_columns=["cat_col"],
        real_columns=["real_col"],
        id_maps={"cat_col": {"a": 3, "b": 4, "c": 5}},
        n_classes={"cat_col": 6},
        storage_layout=StoredWindowLayout(
            stored_context_width=context_length + 1,
            max_target_offset=1,
            version=2,
        ),
        window_view=ModelWindowView(
            context_length=context_length,
            objective="bert",
            target_offset=0,
        ),
        inference_batch_size=inference_batch_size,
        seed=42,
        export_generative_model=True,
        export_embedding_model=False,
        export_onnx=True,
        export_pt=False,
        model_spec=ModelSpecModel(
            initial_embedding_dim=8,
            dim_model=8,
            n_head=2,
            dim_feedforward=8,
            num_layers=1,
            prediction_length=context_length,
            feature_embedding_dims={"cat_col": 7, "real_col": 1},
            activation_fn="relu",
            normalization="layer_norm",
            positional_encoding="learned",
            attention_type="mha",
            norm_first=False,
            n_kv_heads=2,
        ),
        training_spec=TrainingSpecModel(
            training_objective="bert",
            bert_spec={
                "masking_probability": 0.5,
                "replacement_distribution": {
                    "masked": 1.0,
                    "random": 0.0,
                    "identical": 0.0,
                },
                "span_masking": {"type": "GeometricDistribution", "p": 1.0},
            },
            device="cpu",
            epochs=1,
            save_interval_epochs=1,
            batch_size=2,
            learning_rate=0.001,
            criterion={"cat_col": "CrossEntropyLoss", "real_col": "MSELoss"},
            optimizer={"name": "Adam"},
            scheduler={"name": "StepLR", "step_size": 1, "gamma": 0.1},
            loss_weights={"cat_col": 1.0, "real_col": 1.0},
            torch_compile="none",
            layer_autocast=False,
        ),
    )
    model = TransformerModel(config)
    model.eval()

    model._export_model(model, "best", 1)
    export_path = os.path.join(
        project_root, "models", "sequifier-bert-onnx-mask-best-1.onnx"
    )

    session = onnxruntime.InferenceSession(
        export_path, providers=["CPUExecutionProvider"]
    )
    input_names = [session_input.name for session_input in session.get_inputs()]

    assert "attention_valid_mask" in input_names

    ort_inputs = {
        "cat_col_in": np.array([[3, 4, 5, 3], [0, 0, 3, 4]], dtype=np.int64),
        "real_col_in": np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.5, 0.6]], dtype=np.float32
        ),
        "attention_valid_mask": np.array(
            [[True, True, True, True], [False, False, True, True]], dtype=np.bool_
        ),
    }
    outputs = session.run(None, ort_inputs)

    assert len(outputs) == 2
    assert outputs[0].shape[1] == inference_batch_size


def test_onnx_export_preserves_feature_name_order(tmp_path):
    project_root = str(tmp_path)
    (tmp_path / "logs").mkdir()

    context_length = 4
    inference_batch_size = 2
    config = TrainModel(
        project_root=project_root,
        model_name="onnx-feature-order",
        metadata_config_path="metadata.json",
        training_data_path="data/train.pt",
        validation_data_path="data/val.pt",
        input_columns=["cat2", "cat10"],
        target_columns=["cat2"],
        target_column_types={"cat2": "categorical"},
        column_types={"cat2": "int64", "cat10": "int64"},
        categorical_columns=["cat2", "cat10"],
        real_columns=[],
        id_maps={
            "cat2": {"a": 3, "b": 4, "c": 5, "d": 6},
            "cat10": {"x": 3, "y": 4, "z": 5},
        },
        n_classes={"cat2": 7, "cat10": 6},
        storage_layout=StoredWindowLayout(
            stored_context_width=context_length + 1,
            max_target_offset=1,
            version=2,
        ),
        window_view=ModelWindowView(
            context_length=context_length,
            objective="causal",
            target_offset=1,
        ),
        inference_batch_size=inference_batch_size,
        seed=42,
        export_generative_model=True,
        export_embedding_model=False,
        export_onnx=True,
        export_pt=False,
        model_spec=ModelSpecModel(
            initial_embedding_dim=8,
            dim_model=8,
            n_head=2,
            dim_feedforward=8,
            num_layers=1,
            prediction_length=1,
            feature_embedding_dims={"cat2": 4, "cat10": 4},
            activation_fn="relu",
            normalization="layer_norm",
            positional_encoding="learned",
            attention_type="mha",
            norm_first=False,
            n_kv_heads=2,
        ),
        training_spec=TrainingSpecModel(
            training_objective="causal",
            device="cpu",
            epochs=1,
            save_interval_epochs=1,
            batch_size=2,
            learning_rate=0.001,
            criterion={"cat2": "CrossEntropyLoss"},
            optimizer={"name": "Adam"},
            scheduler={"name": "StepLR", "step_size": 1, "gamma": 0.1},
            loss_weights={"cat2": 1.0},
            torch_compile="none",
            layer_autocast=False,
        ),
    )
    model = TransformerModel(config)

    model._export_model(model, "best", 1)
    export_path = os.path.join(
        project_root, "models", "sequifier-onnx-feature-order-best-1.onnx"
    )

    session = onnxruntime.InferenceSession(
        export_path, providers=["CPUExecutionProvider"]
    )
    input_names = [session_input.name for session_input in session.get_inputs()]

    assert input_names == ["cat2_in", "cat10_in", "attention_valid_mask"]

    ort_inputs = {
        "cat2_in": np.array([[3, 4, 5, 6], [6, 5, 4, 3]], dtype=np.int64),
        "cat10_in": np.array([[3, 4, 5, 3], [5, 4, 3, 5]], dtype=np.int64),
        "attention_valid_mask": np.ones(
            (inference_batch_size, context_length), dtype=np.bool_
        ),
    }
    outputs = session.run(None, ort_inputs)

    assert outputs[0].shape == (1, inference_batch_size, config.n_classes["cat2"])

    def assert_onnx_matches_torch(onnx_inputs):
        torch_inputs_for_case = {
            "cat2": torch.tensor(onnx_inputs["cat2_in"], dtype=torch.long),
            "cat10": torch.tensor(onnx_inputs["cat10_in"], dtype=torch.long),
        }
        torch_metadata_for_case = {
            "attention_valid_mask": torch.tensor(
                onnx_inputs["attention_valid_mask"], dtype=torch.bool
            )
        }
        with torch.no_grad():
            torch_outputs = model(torch_inputs_for_case, torch_metadata_for_case)

        onnx_outputs = session.run(None, onnx_inputs)
        np.testing.assert_allclose(
            onnx_outputs[0],
            torch_outputs["cat2"].detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        return onnx_outputs[0]

    base_outputs = assert_onnx_matches_torch(ort_inputs)

    cat2_changed_inputs = {
        **ort_inputs,
        "cat2_in": np.array([[6, 6, 5, 4], [3, 3, 4, 5]], dtype=np.int64),
    }
    cat2_changed_outputs = assert_onnx_matches_torch(cat2_changed_inputs)
    assert not np.allclose(base_outputs, cat2_changed_outputs, rtol=1e-5, atol=1e-6)

    cat10_changed_inputs = {
        **ort_inputs,
        "cat10_in": np.array([[5, 5, 4, 3], [3, 3, 5, 4]], dtype=np.int64),
    }
    cat10_changed_outputs = assert_onnx_matches_torch(cat10_changed_inputs)
    assert not np.allclose(base_outputs, cat10_changed_outputs, rtol=1e-5, atol=1e-6)
