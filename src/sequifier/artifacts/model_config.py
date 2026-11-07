"""Reconstruct an execution-only resolved model config from a lean PT bundle."""

from __future__ import annotations

from sequifier.config.policy_config import CategoricalPolicyConfig, PolicySchemaConfig
from sequifier.config.train_config import (
    BackboneComponentConfig,
    DatasetFreezingSpecModel,
    GlobalTrainingSpecModel,
    ModelInterfaceSpecModel,
    ModelSpecModel,
    ResolvedDatasetTrainingSpec,
    ResolvedModelInterface,
    ResolvedSequifierConfig,
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
        interface_name = next(iter(interface_values))
    if interface_name not in interface_values:
        raise ValueError(f"Unknown PT model interface {interface_name!r}")
    assert isinstance(interface_name, str)

    backbone = BackboneComponentConfig.model_validate(values["backbone"])
    context_length = int(values["context_length"])
    target_offset = int(values.get("target_offset", 1))
    objective = str(values["training_objective"])
    global_spec = GlobalTrainingSpecModel(
        read_format=(
            "pt"
            if any(i.get("depth_layouts") for i in interface_values.values())
            else "parquet"
        ),
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
    resolved_interfaces = {}
    fallback_storage_layout = StoredWindowLayout(
        window_length=context_length + max(1, target_offset),
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
        authored = {
            key: value
            for key, value in interface.items()
            if key in ModelInterfaceSpecModel.model_fields
        }
        policy_spec = interface.get("policy")
        if policy_spec is not None:
            authored["policy"] = PolicySchemaConfig.model_validate(
                {
                    key: value
                    for key, value in policy_spec.items()
                    if key in PolicySchemaConfig.model_fields
                }
            )
        authored_interfaces[name] = ModelInterfaceSpecModel.model_validate(authored)
        if policy_spec is not None:
            policy_spec = {"categorical_encoding": "legacy", **policy_spec}
        target_decoder_ids = interface.get("target_decoder_ids", {})
        target_n_classes = interface.get(
            "target_n_classes",
            {column: len(ids) for column, ids in target_decoder_ids.items()},
        )
        resolved = ResolvedModelInterface.model_validate(
            {
                **{
                    key: value
                    for key, value in interface.items()
                    if key in ResolvedModelInterface.model_fields
                },
                "name": name,
                "policy": CategoricalPolicyConfig.model_validate(policy_spec)
                if policy_spec is not None
                else None,
                "n_classes": interface.get("n_classes", target_n_classes),
                "target_decoder_ids": target_decoder_ids,
                "target_n_classes": target_n_classes,
                "target_global_to_decoder": interface.get(
                    "target_global_to_decoder", {}
                ),
                "id_maps": interface.get("id_maps", {}),
                "special_token_ids": interface.get(
                    "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
                ),
                "categorical_decoder_special_tokens": interface.get(
                    "categorical_decoder_special_tokens", {}
                ),
                "storage_layout": storage_layout,
                "window_view": window_view,
            }
        )
        if resolved.depth_layouts and resolved.tensor_payload_version not in {2, 3}:
            raise ValueError(
                "Depth model execution requires tensor_payload_version 2 or 3"
            )
        persisted_schema = interface.get("execution_schema")
        if persisted_schema is not None:
            from sequifier.model.execution_schema import ExecutionSchema

            actual_schema = ExecutionSchema.from_dict(persisted_schema)
            expected_schema = ExecutionSchema.from_interface(resolved, context_length)
            import json

            if json.dumps(actual_schema.to_dict(), sort_keys=True) != json.dumps(
                expected_schema.to_dict(), sort_keys=True
            ):
                raise ValueError(f"Execution schema contradicts interface {name!r}")
        resolved_interfaces[name] = resolved
    config = execution_config(
        ModelSpecModel(backbone=backbone, interfaces=authored_interfaces),
        resolved_interfaces,
        global_spec,
        device=device,
        embedding_layer_names=values.get(
            "embedding_layer_names", ["backbone.final_norm"]
        ),
    )
    return config, interface_name


def execution_config(model, interfaces, global_training, *, device="cpu", **options):
    """Compose the same execution contract for loaded and newly resolved models."""
    return ResolvedSequifierConfig(
        global_training=global_training,
        model=model,
        dataset_training={
            name: ResolvedDatasetTrainingSpec(
                name=name,
                model_interface=name,
                interface=interface,
                parts={},
                criterion={
                    target: "CrossEntropyLoss" if kind == "categorical" else "MSELoss"
                    for target, kind in interface.target_column_types.items()
                },
                class_share_log_columns=[],
                freeze=DatasetFreezingSpecModel(),
            )
            for name, interface in interfaces.items()
        },
        **{
            "project_root": ".",
            "model_name": "loaded-model",
            "seed": 0,
            "device": device,
            "training_plan": [],
            "evaluation_sources": [],
            "evaluation_monitor": None,
            "export_generative_model": True,
            "export_embedding_model": False,
            "export_onnx": False,
            "export_pt": False,
            "embedding_layer_names": ["backbone.final_norm"],
            **options,
        },
    )
