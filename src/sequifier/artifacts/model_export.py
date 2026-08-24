from typing import Any


def model_execution_config(training_config: Any) -> dict[str, Any]:
    """Return only configuration required to reconstruct model execution."""

    if not hasattr(training_config, "dataset_training_spec"):
        raise TypeError("Model export requires a canonical training config")
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
                exclude={"initialization"},
            ),
            "decoder": interface.decoder.model_dump(
                mode="python",
                exclude={"initialization"},
            ),
            "n_classes": interface.n_classes,
            "id_maps": interface.id_maps,
            "special_token_ids": interface.special_token_ids,
            "selected_columns_statistics": (interface.selected_columns_statistics),
            "normalize_real_columns": interface.normalize_real_columns,
            "target_decoder_ids": interface.target_decoder_ids,
            "target_n_classes": interface.target_n_classes,
            "target_global_to_decoder": interface.target_global_to_decoder,
        }
    spec = training_config.global_training_spec
    return {
        "training_objective": spec.training_objective,
        "context_length": spec.context_length,
        "target_offset": spec.target_offset,
        "next_occurrence_config": (
            spec.next_occurrence_config.model_dump(mode="python")
            if spec.next_occurrence_config is not None
            else None
        ),
        "backbone": training_config.model_spec.backbone.model_dump(
            mode="python",
            exclude={"repository", "initialization"},
        ),
        "layer_type_dtypes": spec.layer_type_dtypes,
        "interfaces": interfaces,
    }


def pt_bundle(model: Any, training_config: Any) -> dict[str, Any]:
    """Return the canonical lean PyTorch inference bundle."""

    return {
        "artifact_type": "sequifier_model",
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "model_config": model_execution_config(training_config),
    }
