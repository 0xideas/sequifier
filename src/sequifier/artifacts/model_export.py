from typing import Any


def model_execution_config(training_config: Any) -> dict[str, Any]:
    """Return only configuration required to reconstruct model execution."""

    if hasattr(training_config, "dataset_training_spec"):
        interfaces = {}
        for dataset in training_config.dataset_training_spec.values():
            interface = dataset.interface
            if interface.name in interfaces:
                continue
            interfaces[interface.name] = {
                "input_columns": interface.input_columns,
                "target_columns": interface.target_columns,
                "target_column_types": interface.target_column_types,
                "column_data_types": interface.column_data_types,
                "categorical_columns": interface.categorical_columns,
                "real_columns": interface.real_columns,
                "categorical_decoder_special_tokens": (
                    interface.categorical_decoder_special_tokens
                ),
                "feature_layout": (
                    interface.feature_layout.model_dump(mode="python")
                    if interface.feature_layout is not None
                    else None
                ),
                "ingestion": interface.ingestion.model_dump(
                    mode="python",
                    exclude={"freezing", "freezing_except", "initialization"},
                ),
                "decoder": interface.decoder.model_dump(
                    mode="python",
                    exclude={"freezing", "freezing_except", "initialization"},
                ),
                "n_classes": interface.n_classes,
                "target_decoder_ids": interface.target_decoder_ids,
                "target_n_classes": interface.target_n_classes,
                "target_global_to_decoder": interface.target_global_to_decoder,
            }
        spec = training_config.global_training_spec
        return {
            "training_objective": spec.training_objective,
            "context_length": spec.context_length,
            "target_offset": spec.target_offset,
            "backbone": training_config.model_spec.backbone.model_dump(
                mode="python",
                exclude={
                    "repository",
                    "freezing",
                    "freezing_except",
                    "initialization",
                },
            ),
            "layer_type_dtypes": spec.layer_type_dtypes,
            "interfaces": interfaces,
        }

    # Hyperparameter search remains isolated on its historical concrete model,
    # but its exported inference artifact follows the lean format too.
    return {
        "training_objective": training_config.training_objective,
        "context_length": training_config.context_length,
        "target_offset": training_config.target_offset,
        "backbone": training_config.model_spec.backbone.model_dump(
            mode="python",
            exclude={
                "repository",
                "freezing",
                "freezing_except",
                "initialization",
            },
        ),
        "layer_type_dtypes": training_config.training_spec.layer_type_dtypes,
        "interfaces": {
            "default": {
                "input_columns": training_config.input_columns,
                "target_columns": training_config.target_columns,
                "target_column_types": training_config.target_column_types,
                "column_data_types": training_config.column_data_types,
                "categorical_columns": training_config.categorical_columns,
                "real_columns": training_config.real_columns,
                "categorical_decoder_special_tokens": getattr(
                    training_config, "categorical_decoder_special_tokens", {}
                ),
                "feature_layout": (
                    training_config.feature_layout.model_dump(mode="python")
                    if training_config.feature_layout is not None
                    else None
                ),
                "ingestion": training_config.model_spec.ingestion.model_dump(
                    mode="python",
                    exclude={"freezing", "freezing_except", "initialization"},
                ),
                "decoder": training_config.model_spec.decoder.model_dump(
                    mode="python",
                    exclude={"freezing", "freezing_except", "initialization"},
                ),
                "n_classes": training_config.n_classes,
            }
        },
    }


def pt_bundle(model: Any, training_config: Any) -> dict[str, Any]:
    """Return the canonical lean PyTorch inference bundle."""

    model_state_dict = model.state_dict()
    if not hasattr(training_config, "dataset_training_spec"):
        route_components = ("ingestion.", "ingestion_adapter.", "decoder.")
        model_state_dict = {
            (
                f"interfaces.default.{name}"
                if name.startswith(route_components)
                else name
            ): value
            for name, value in model_state_dict.items()
        }
    return {
        "artifact_type": "sequifier_model",
        "format_version": 2,
        "model_state_dict": model_state_dict,
        "model_config": model_execution_config(training_config),
    }
