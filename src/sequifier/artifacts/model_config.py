"""Reconstruct an execution-only resolved model config from a lean PT bundle."""

from __future__ import annotations

import json

from sequifier.config.column_names import NEXT_ACTION_OFFSET
from sequifier.config.components import BackboneComponentConfig
from sequifier.config.metadata import validate_tensor_payload_contract
from sequifier.config.model_interface import (
    ResolvedInterfaceStorage,
    ResolvedModelInterface,
)
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import ModelWindowView
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_options import (
    DEFAULT_MAX_CONCAT_TENSORS,
    ModelExecutionOptions,
)
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.objectives import target_offset_for_objective
from sequifier.typechecking import beartype


def validate_execution_attachment(
    source: dict, current_execution: ModelExecution
) -> None:
    """Require existing prediction semantics when reusing complete model weights."""
    from pydantic_core import to_jsonable_python

    from sequifier.artifacts.model_export import model_execution_config

    source_config, _ = model_execution_from_model_config(source)
    source = model_execution_config(source_config)
    current = model_execution_config(current_execution)
    if source["backbone"]["architecture"] != current["backbone"]["architecture"]:
        raise ValueError("model attachment cannot change backbone architecture")
    if source.get("layer_type_dtypes") != current.get("layer_type_dtypes"):
        raise ValueError("model attachment cannot silently change parameter precision")
    if source["context_length"] != current["context_length"]:
        raise ValueError("model attachment cannot change context length")

    if source["forward_attention"] != current["forward_attention"]:
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
            "n_classes",
            "target_decoder_ids",
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


def validate_interface_storage_versions(values: dict, depth_layouts=None) -> None:
    storage = values.get("storage_layout")
    require_contract_version(
        "stored-window layout",
        storage.get("version") if isinstance(storage, dict) else None,
        ContractVersions.STORED_WINDOW_LAYOUT,
    )
    validate_tensor_payload_contract(
        values.get("tensor_payload_kind"),
        values.get("tensor_payload_version"),
        values.get("depth_layouts") if depth_layouts is None else depth_layouts,
        projected=True,
    )


def split_interface_storage(
    values: dict,
) -> tuple[dict, ResolvedInterfaceStorage | None]:
    """Read the flat storage fields used by existing artifact and checkpoint formats."""
    if not ResolvedInterfaceStorage.model_fields.keys() & values.keys():
        return values, None
    validate_interface_storage_versions(values)
    model_values = dict(values)
    storage = {
        name: model_values.pop(name)
        for name in ResolvedInterfaceStorage.model_fields
        if name in model_values
    }
    return model_values, ResolvedInterfaceStorage.model_validate(storage)


def validate_interface_versions(interface: dict) -> None:
    if ResolvedInterfaceStorage.model_fields.keys() & interface.keys():
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
    interface_name: str | None = None,
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
    # Older policy artifacts described the decoder label slot rather than the
    # physical arrival row. Their weights and decision histories are compatible.
    if objective == "policy" and target_offset == 0:
        target_offset = NEXT_ACTION_OFFSET
    options = ModelExecutionOptions(
        training_objective=objective,
        forward_attention=values.get("forward_attention"),
        context_length=context_length,
        target_offset=target_offset,
        layer_type_dtypes=values.get("layer_type_dtypes"),
        max_concat_tensors=values.get("max_concat_tensors", DEFAULT_MAX_CONCAT_TENSORS),
    )
    resolved_interfaces = {}
    interface_storage = {}
    window_view = ModelWindowView(
        context_length=context_length,
        objective=objective,
        target_offset=target_offset_for_objective(objective, target_offset),
    )
    interface_order = [interface_name] + [
        name for name in interface_values if name != interface_name
    ]
    for name in interface_order:
        interface = interface_values[name]
        validate_interface_versions(interface)
        interface, storage = split_interface_storage(interface)
        if storage is not None:
            storage.validate_view(window_view)
            interface_storage[name] = storage
        fields = dict(interface)
        actual_schema = ExecutionSchema.from_dict(fields.pop("execution_schema"))
        resolved = ResolvedModelInterface.model_validate(
            {**fields, "name": name, "window_view": window_view}
        )
        expected_schema = ExecutionSchema.from_interface(resolved, context_length)
        if json.dumps(actual_schema.to_dict(), sort_keys=True) != json.dumps(
            expected_schema.to_dict(), sort_keys=True
        ):
            raise ValueError(f"Execution schema contradicts interface {name!r}")
        resolved_interfaces[name] = resolved
    config = ModelExecution(
        backbone,
        resolved_interfaces,
        options,
        interface_storage=interface_storage,
        embedding_layer_names=values.get(
            "embedding_layer_names", ["backbone.final_norm"]
        ),
    )
    return config.validate(), interface_name


def extend_prediction_distributions(execution, declared_model):
    """Allow explicit new scale parameters while preserving an initialized model."""
    exclude = {
        "backbone": {"repository"},
        "interfaces": {"__all__": {"prediction_distributions"}},
    }
    from sequifier.config.train_config import OwnedModelSpec

    if isinstance(declared_model, OwnedModelSpec):
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
    return execution


def load_attached_model_weights(network, artifact):
    """Restore source weights, initializing only newly declared model scale parameters."""
    from sequifier.model.prediction_distribution import PredictionScales

    expected = network.state_dict()
    source = artifact.model_state_dict
    missing = expected.keys() - source.keys()
    allowed = set()
    for name, route in network.interfaces.items():
        scales = route.prediction_scales
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
