"""Reconstruct an execution-only resolved model config from a lean PT bundle."""

from __future__ import annotations

import json

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.train_config import (
    BackboneComponentConfig,
    ModelInterfaceSpecModel,
    ModelSpecModel,
    ResolvedModelInterface,
    ResolvedSequifierConfig,
    TrainingRuntimeSettings,
)
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import ModelWindowView, StoredWindowLayout, resolve_window_view
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.special_tokens import SPECIAL_TOKEN_IDS
from sequifier.typechecking import beartype


def validate_execution_attachment(source: dict, current: dict) -> None:
    """Require existing prediction semantics when reusing complete model weights."""
    from pydantic_core import to_jsonable_python

    from sequifier.artifacts.model_export import model_execution_config

    source_config, _ = model_execution_from_model_config(source, device="cpu")
    current_config, _ = model_execution_from_model_config(current, device="cpu")
    source = model_execution_config(source_config)
    current = model_execution_config(current_config)
    if source["backbone"]["architecture"] != current["backbone"]["architecture"]:
        raise ValueError("model attachment cannot change backbone architecture")
    if source.get("layer_type_dtypes") != current.get("layer_type_dtypes"):
        raise ValueError("model attachment cannot silently change parameter precision")
    if source["context_length"] != current["context_length"]:
        raise ValueError("model attachment cannot change context length")

    def attention(values):
        return values.get("forward_attention") or (
            "bidirectional" if values["training_objective"] == "bert" else "causal"
        )

    if attention(source) != attention(current):
        raise ValueError("model attachment cannot change forward attention")
    for name, interface in source["interfaces"].items():
        if name not in current["interfaces"]:
            raise ValueError(f"initial model interface {name!r} was dropped")
        for target, distribution in interface.get(
            "prediction_distributions", {}
        ).items():
            if to_jsonable_python(distribution) != to_jsonable_python(
                current["interfaces"][name]
                .get("prediction_distributions", {})
                .get(target)
            ):
                raise ValueError(
                    f"initial model prediction parameters differ at {name}.{target}"
                )
        for key in (
            "input_columns",
            "target_columns",
            "ingestion",
            "decoder",
            "id_maps",
            "target_decoder_ids",
            "target_global_to_decoder",
            "selected_columns_statistics",
            "normalize_real_columns",
            "depth_layouts",
            "static_conditioning",
            "feature_layout",
        ):
            default = (
                {}
                if key == "depth_layouts"
                else True
                if key == "normalize_real_columns"
                else None
            )
            if to_jsonable_python(interface.get(key, default)) != to_jsonable_python(
                current["interfaces"][name].get(key, default)
            ):
                raise ValueError(
                    f"initial model execution differs at interfaces.{name}.{key}"
                )


@beartype
def resolved_config_from_model_config(
    values: dict,
    *,
    device: str,
    interface_name: str | None = None,
) -> tuple[ResolvedSequifierConfig, str]:
    """Public compatibility projection of a portable model's execution facts."""
    execution, selected = model_execution_from_model_config(
        values, device=device, interface_name=interface_name, validate=False
    )
    return resolved_training_config(execution), selected


def validate_interface_storage_versions(interface: dict) -> None:
    storage = interface.get("storage_layout")
    require_contract_version(
        "stored-window layout",
        storage.get("version") if isinstance(storage, dict) else None,
        ContractVersions.STORED_WINDOW_LAYOUT,
    )
    # The interface stores only the depth columns it consumes. A shallow projection
    # can still refer to a dataset that uses the depth metadata version.
    expected = (
        ContractVersions.DEPTH_TENSOR_METADATA
        if interface.get("depth_layouts")
        else (
            ContractVersions.DATASET_TENSOR_METADATA,
            ContractVersions.DEPTH_TENSOR_METADATA,
        )
    )
    require_contract_version(
        "model dataset tensor metadata",
        interface.get("tensor_payload_version"),
        expected,
    )


def validate_interface_versions(interface: dict) -> None:
    validate_interface_storage_versions(interface)
    schema = interface.get("execution_schema")
    require_contract_version(
        "model execution schema",
        schema.get("version") if isinstance(schema, dict) else None,
        ContractVersions.MODEL_EXECUTION_SCHEMA,
    )


@beartype
def model_execution_from_model_config(
    values: dict,
    *,
    device: str,
    interface_name: str | None = None,
    validate: bool = True,
) -> tuple[ModelExecution, str]:
    """Resolve portable model execution without inventing datasets or losses."""

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
    global_spec = TrainingRuntimeSettings(
        training_objective=objective,
        forward_attention=values.get("forward_attention"),
        context_length=context_length,
        target_offset=target_offset,
        inference_batch_size=1,
        layer_type_dtypes=values.get("layer_type_dtypes"),
        torch_compile="none",
    )
    authored_interfaces = {}
    resolved_interfaces = {}
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
        validate_interface_versions(interface)
        storage_layout = StoredWindowLayout(**interface["storage_layout"])
        resolve_window_view(storage_layout, window_view)
        authored = {
            key: value
            for key, value in interface.items()
            if key in ModelInterfaceSpecModel.model_fields
        }
        policy_spec = interface.get("policy")
        authored_interfaces[name] = ModelInterfaceSpecModel.model_validate(authored)
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
                "policy": ResolvedPolicyBinding.model_validate(policy_spec)
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
        persisted_schema = interface.get("execution_schema")
        actual_schema = ExecutionSchema.from_dict(persisted_schema)
        expected_schema = ExecutionSchema.from_interface(resolved, context_length)
        if json.dumps(actual_schema.to_dict(), sort_keys=True) != json.dumps(
            expected_schema.to_dict(), sort_keys=True
        ):
            raise ValueError(f"Execution schema contradicts interface {name!r}")
        resolved_interfaces[name] = resolved
    config = ModelExecution(
        ModelSpecModel(backbone=backbone, interfaces=authored_interfaces),
        resolved_interfaces,
        global_spec,
        device=device,
        embedding_layer_names=values.get(
            "embedding_layer_names", ["backbone.final_norm"]
        ),
    )
    return config.validate() if validate else config, interface_name


def execution_config(model, interfaces, runtime_settings, *, device="cpu", **options):
    """Adapt model execution to the existing public resolved training contract."""
    return resolved_training_config(
        ModelExecution(model, interfaces, runtime_settings, device=device), **options
    )


def resolved_training_config(
    execution: ModelExecution, **options
) -> ResolvedSequifierConfig:
    """Project a model onto the resolved training schema for compatibility callers."""
    values = {
        "project_root": execution.project_root,
        "model_name": execution.model_name,
        "seed": execution.seed,
        "device": execution.device,
        "runtime_settings": execution.runtime_settings,
        "model": execution.model,
        "interfaces": execution.interfaces,
        "dataset_training": {},
        "training_plan": [],
        "export_generative_model": True,
        "export_embedding_model": False,
        "export_onnx": False,
        "export_pt": False,
        "embedding_layer_names": list(execution.embedding_layer_names),
    }
    return ResolvedSequifierConfig(**{**values, **options})


def extend_prediction_distributions(execution, declared_model):
    """Allow explicit new scale parameters while preserving an initialized model."""
    exclude = {"interfaces": {"__all__": {"prediction_distributions"}}}
    if hasattr(declared_model, "architecture_model"):
        declared_model = declared_model.architecture_model()
    declared = declared_model.model_dump(exclude=exclude)
    existing = execution.model.model_dump(exclude=exclude)
    for name, interface in declared["interfaces"].items():
        if name in existing["interfaces"]:
            tokens = interface["categorical_decoder_special_tokens"]
            for target, value in existing["interfaces"][name][
                "categorical_decoder_special_tokens"
            ].items():
                tokens.setdefault(target, value)
    if declared != existing:
        raise ValueError(
            "declared model architecture differs from initialize_from model"
        )
    for name, declared in declared_model.interfaces.items():
        interface = execution.interface(name)
        distributions = dict(interface.prediction_distributions)
        for target, field in declared.prediction_distributions.items():
            if target in distributions and distributions[target] != field:
                raise ValueError(
                    f"{name}.{target}: declared prediction parameters differ from initialize_from"
                )
            if (
                target not in interface.target_columns
                or interface.target_column_types.get(target) != "real"
            ):
                raise ValueError(
                    f"{name}.{target}: prediction_distributions requires a real model target"
                )
            distributions[target] = field
        interface.prediction_distributions = distributions
        execution.model.interfaces[name].prediction_distributions = dict(distributions)
    return execution


def load_attached_model_weights(network, artifact):
    """Restore source weights, initializing only newly declared model scale parameters."""
    from sequifier.model.prediction_distribution import PredictionScales

    expected = network.state_dict()
    source = artifact.model_state_dict
    missing = expected.keys() - source.keys()
    allowed = set()
    for name, route in network.interfaces.items():
        scales = getattr(route, "prediction_scales", None)
        if not isinstance(scales, PredictionScales):
            continue
        original = artifact.model_config.values["interfaces"][name].get(
            "prediction_distributions", {}
        )
        allowed.update(
            f"interfaces.{name}.prediction_scales.log_scales.{target}"
            for target in scales.log_scales
            if target not in original
        )
    if missing - allowed:
        raise ValueError(
            f"initial model is missing weights: {sorted(missing - allowed)}"
        )
    network.load_state_dict(
        {**{key: expected[key] for key in missing}, **source}, strict=True
    )
