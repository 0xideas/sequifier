import pytest
import torch

from sequifier.config.train_config import (
    GroupedIngestionConfig,
    ModelSpecModel,
    TrainModel,
)
from sequifier.helpers import ModelWindowView, StoredWindowLayout
from sequifier.model.ingestions import (
    CompositeFeatureIngestion,
    DirectEmbedFeatureIngestion,
    GroupedFeatureIngestion,
    IngestionMerge,
    TemporalConvFeatureIngestion,
)


def _model_spec(**overrides):
    values = {
        "dim_model": 4,
        "n_head": 2,
        "dim_feedforward": 8,
        "num_layers": 1,
        "prediction_length": 1,
    }
    values.update(overrides)
    return ModelSpecModel(**values)


def _train_config(
    ingestion_spec,
    *,
    input_columns=None,
    categorical_columns=None,
    real_columns=None,
    dim_model=4,
):
    input_columns = input_columns or ["cat", "real"]
    categorical_columns = (
        ["cat"] if categorical_columns is None else categorical_columns
    )
    real_columns = ["real"] if real_columns is None else real_columns
    return {
        "project_root": ".",
        "metadata_config_path": "metadata.json",
        "model_name": "model",
        "training_data_path": "train.parquet",
        "validation_data_path": "val.parquet",
        "input_columns": input_columns,
        "column_types": {
            "target": "Float32",
            "cat": "Int64",
            "real": "Float32",
            "extra": "Float32",
        },
        "categorical_columns": categorical_columns,
        "real_columns": real_columns,
        "target_columns": ["target"],
        "target_column_types": {"target": "real"},
        "id_maps": {"cat": {"a": 0}},
        "storage_layout": StoredWindowLayout(
            stored_context_width=6,
            max_target_offset=1,
            version=2,
        ),
        "window_view": ModelWindowView(
            context_length=4,
            objective="causal",
            target_offset=1,
        ),
        "n_classes": {"cat": 3},
        "inference_batch_size": 2,
        "seed": 1,
        "export_generative_model": True,
        "export_embedding_model": False,
        "model_spec": {
            "ingestion_spec": ingestion_spec,
            "dim_model": dim_model,
            "n_head": 2,
            "dim_feedforward": 8,
            "num_layers": 1,
            "prediction_length": 1,
        },
        "training_spec": {
            "training_objective": "causal",
            "device": "cpu",
            "epochs": 1,
            "save_interval_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "criterion": {"target": "MSELoss"},
            "optimizer": {"name": "Adam"},
            "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.99},
        },
    }


def test_provided_ingestion_configs_require_output_dim():
    with pytest.raises(Exception, match="output_dim"):
        _model_spec(ingestion_spec={"type": "direct_embed"})

    spec = _model_spec()
    ingestion_spec = spec.ingestion_spec

    assert ingestion_spec is not None
    assert not isinstance(ingestion_spec, dict)
    assert ingestion_spec.output_dim == 4


def test_module_dict_unsafe_branch_and_group_names_are_rejected():
    with pytest.raises(Exception, match="cannot contain"):
        _model_spec(
            ingestion_spec={
                "bad.name": {
                    "type": "direct_embed",
                    "columns": ["cat"],
                    "output_dim": 4,
                    "feature_embedding_dims": {"cat": 4},
                }
            }
        )

    with pytest.raises(Exception, match="cannot contain"):
        GroupedIngestionConfig(
            type="grouped",
            output_dim=4,
            groups={"bad.name": ["real"]},
        )


def test_ingestion_must_consume_all_input_columns():
    with pytest.raises(Exception, match="unused columns"):
        TrainModel(
            **_train_config(
                {
                    "type": "direct_embed",
                    "columns": ["cat"],
                    "output_dim": 4,
                    "feature_embedding_dims": {"cat": 4},
                }
            )
        )

    with pytest.raises(Exception, match="unused columns"):
        TrainModel(
            **_train_config(
                {
                    "cat_branch": {
                        "type": "direct_embed",
                        "columns": ["cat"],
                        "output_dim": 4,
                        "feature_embedding_dims": {"cat": 4},
                    }
                }
            )
        )


def test_manual_feature_embedding_dims_must_match_output_dim():
    with pytest.raises(Exception, match="feature_embedding_dims sum"):
        TrainModel(
            **_train_config(
                {
                    "type": "direct_embed",
                    "output_dim": 4,
                    "feature_embedding_dims": {"cat": 2, "real": 1},
                }
            )
        )

    TrainModel(
        **_train_config(
            {
                "type": "direct_embed",
                "output_dim": 4,
                "feature_embedding_dims": {"cat": 2, "real": 2},
            }
        )
    )


def test_runtime_module_dict_key_guards():
    with pytest.raises(ValueError, match="cannot contain"):
        CompositeFeatureIngestion(
            branches={"bad.name": torch.nn.Identity()},
            merge_type="concat",
            output_dim=4,
        )

    with pytest.raises(ValueError, match="cannot contain"):
        GroupedFeatureIngestion(
            groups={"bad.name": ["real"]},
            categorical_columns=[],
            real_columns=["real"],
            n_classes={},
            context_length=4,
            output_dim=4,
            use_rope=False,
            dropout=0.0,
        )


def test_merge_and_temporal_conv_layers_use_custom_initializer():
    torch.manual_seed(1)
    merge = IngestionMerge("gated", {"a": 2, "b": 4}, 4)
    for projection in merge.branch_projections.values():
        if isinstance(projection, torch.nn.Linear):
            torch.nn.init.constant_(projection.bias, 1.0)
    torch.nn.init.constant_(merge.gate.bias, 1.0)

    merge.initialize_weights()

    for projection in merge.branch_projections.values():
        if isinstance(projection, torch.nn.Linear):
            assert torch.allclose(projection.bias, torch.zeros_like(projection.bias))
    assert torch.allclose(merge.gate.bias, torch.zeros_like(merge.gate.bias))

    attention_merge = IngestionMerge("attention", {"a": 4}, 4)
    assert torch.count_nonzero(attention_merge.query) == 0

    attention_merge.initialize_weights()

    assert torch.count_nonzero(attention_merge.query) > 0

    base = DirectEmbedFeatureIngestion(
        categorical_columns=[],
        real_columns=["real"],
        n_classes={},
        context_length=4,
        embedding_size=4,
        feature_embedding_dims=None,
        use_rope=False,
        dropout=0.0,
        output_dim=4,
    )
    temporal_conv = TemporalConvFeatureIngestion(
        base_ingestion=base,
        output_dim=4,
        kernel_size=3,
        dilation=1,
        num_layers=1,
        causal=True,
        activation_fn="gelu",
        dropout=0.0,
    )
    torch.nn.init.constant_(temporal_conv.layers[0].bias, 1.0)

    temporal_conv.initialize_weights()

    assert torch.allclose(
        temporal_conv.layers[0].bias,
        torch.zeros_like(temporal_conv.layers[0].bias),
    )
