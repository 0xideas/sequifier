"""Reconstruct an execution-only resolved model config from a lean PT bundle."""

from __future__ import annotations

from pydantic import TypeAdapter

from sequifier.config.composable_train_config import (
    DatasetFreezingSpecModel,
    GlobalTrainingSpecModel,
    ModelInterfaceSpecModel,
    ModelSpecModel,
    ResolvedDatasetTrainingSpec,
    ResolvedModelInterface,
    ResolvedSequifierConfig,
)
from sequifier.config.train_config import (
    BackboneComponentConfig,
    DecoderComponentConfig,
    FeatureLayoutRegistryModel,
    IngestionComponentConfig,
)
from sequifier.helpers import ModelWindowView, StoredWindowLayout, resolve_window_view
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.typechecking import beartype


@beartype
def resolved_config_from_model_config(
    values: dict,
    *,
    device: str,
    interface_name: str | None = None,
) -> tuple[ResolvedSequifierConfig, str]:
    """Build an execution-only config and return its selected interface name."""

    interface_values = values.get("interfaces")
    if not isinstance(interface_values, dict) or not interface_values:
        raise ValueError("model_config.interfaces must be a non-empty mapping")
    if interface_name is None:
        if len(interface_values) != 1:
            raise ValueError(
                "A model interface selection is required for this PT bundle"
            )
        interface_name = next(iter(interface_values))
    if interface_name not in interface_values:
        raise ValueError(f"Unknown PT model interface {interface_name!r}")
    assert isinstance(interface_name, str)

    backbone = BackboneComponentConfig.model_validate(values["backbone"])
    context_length = int(values["context_length"])
    target_offset = int(values.get("target_offset", 1))
    objective = str(values["training_objective"])
    global_spec = GlobalTrainingSpecModel(
        read_format="parquet",
        training_objective=objective,
        context_length=context_length,
        target_offset=target_offset,
        inference_batch_size=1,
        batch_size=1,
        learning_rate=1e-3,
        layer_type_dtypes=values.get("layer_type_dtypes"),
        next_occurrence_config=values.get("next_occurrence_config"),
        torch_compile="none",
    )
    authored_interfaces = {}
    resolved_datasets = {}
    fallback_storage_layout = StoredWindowLayout(
        stored_context_width=context_length + max(1, target_offset),
        max_target_offset=max(1, target_offset),
        version=2,
    )
    window_view = ModelWindowView(
        context_length=context_length,
        objective=objective,
        target_offset=(0 if objective == "bert" else target_offset),
    )
    interface_order = [interface_name] + [
        name for name in interface_values if name != interface_name
    ]
    for name in interface_order:
        interface = interface_values[name]
        storage_layout_values = interface.get("storage_layout")
        storage_layout = (
            StoredWindowLayout(**storage_layout_values)
            if storage_layout_values is not None
            else fallback_storage_layout
        )
        resolve_window_view(storage_layout, window_view)
        ingestion = TypeAdapter(IngestionComponentConfig).validate_python(
            interface["ingestion"]
        )
        decoder = TypeAdapter(DecoderComponentConfig).validate_python(
            interface["decoder"]
        )
        feature_layout = (
            FeatureLayoutRegistryModel.model_validate(interface["feature_layout"])
            if interface.get("feature_layout") is not None
            else None
        )
        authored_interfaces[name] = ModelInterfaceSpecModel(
            input_columns=interface["input_columns"],
            target_columns=interface["target_columns"],
            categorical_decoder_special_tokens=interface.get(
                "categorical_decoder_special_tokens", {}
            ),
            feature_layout=feature_layout,
            ingestion=ingestion,
            decoder=decoder,
        )
        target_decoder_ids = interface.get("target_decoder_ids", {})
        target_n_classes = interface.get(
            "target_n_classes",
            {column: len(ids) for column, ids in target_decoder_ids.items()},
        )
        global_to_decoder = interface.get("target_global_to_decoder", {})
        resolved = ResolvedModelInterface(
            name=name,
            input_columns=interface["input_columns"],
            target_columns=interface["target_columns"],
            target_column_types=interface["target_column_types"],
            column_data_types=interface["column_data_types"],
            categorical_columns=interface["categorical_columns"],
            real_columns=interface["real_columns"],
            categorical_decoder_special_tokens=interface.get(
                "categorical_decoder_special_tokens", {}
            ),
            feature_layout=feature_layout,
            ingestion=ingestion,
            decoder=decoder,
            n_classes=interface.get("n_classes", target_n_classes),
            id_maps=interface.get("id_maps", {}),
            special_token_ids=interface.get(
                "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
            ),
            selected_columns_statistics=interface.get(
                "selected_columns_statistics", {}
            ),
            normalize_real_columns=interface.get("normalize_real_columns", True),
            target_decoder_ids=target_decoder_ids,
            target_n_classes=target_n_classes,
            target_global_to_decoder=global_to_decoder,
            storage_layout=storage_layout,
            window_view=window_view,
        )
        criteria = {
            target: (
                "CrossEntropyLoss"
                if resolved.target_column_types[target] == "categorical"
                else "MSELoss"
            )
            for target in resolved.target_columns
        }
        resolved_datasets[name] = ResolvedDatasetTrainingSpec(
            name=name,
            model_interface=name,
            interface=resolved,
            parts={},
            criterion=criteria,
            class_share_log_columns=[],
            freezing=DatasetFreezingSpecModel(),
        )
    model_spec = ModelSpecModel(
        backbone=backbone,
        interfaces=authored_interfaces,
    )
    config = ResolvedSequifierConfig(
        project_root=".",
        model_name="loaded-model",
        device=device,
        seed=0,
        global_training_spec=global_spec,
        model_spec=model_spec,
        dataset_training_spec=resolved_datasets,
        training_plan=[],
        evaluation_sources=[],
        evaluation_monitor=None,
        export_generative_model=True,
        export_embedding_model=False,
        embedding_layer_names=values.get(
            "embedding_layer_names", ["backbone.final_norm"]
        ),
        export_onnx=False,
        export_pt=False,
    )
    return config, interface_name
