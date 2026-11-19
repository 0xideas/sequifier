"""Compatibility and weight copying for the model's decision-history path."""

from sequifier.model.execution import ModelExecution
from sequifier.model.execution_schema import ExecutionSchema

_HISTORY_COMPONENTS = (
    "ingestion",
    "ingestion_adapter",
    "static_condition_encoder",
    "preconditioning_image_encoder",
    "prefix_projection",
)


def _ingestion_contract(ingestion):
    values = ingestion.model_dump(exclude={"initialization", "initialization_seed"})
    if ingestion.type == "composite":
        values["branches"] = [
            (name, _ingestion_contract(branch))
            for name, branch in ingestion.branches.items()
        ]
    return values


def _history_contract(config, interface_name):
    execution = ModelExecution.from_config(config)
    if interface_name not in execution.interfaces:
        raise ValueError(f"history transfer requires interface {interface_name!r}")
    interface = execution.interface(interface_name)
    settings = execution.options
    static = interface.static_conditioning
    columns = interface.input_columns + (static.columns if static else [])
    media_fields = {
        "path",
        "signature",
        "time_index_path",
        "time_index_signature",
        "video_index_paths",
    }
    return {
        "backbone.architecture": execution.model.backbone.architecture,
        "layer_type_dtypes": settings.layer_type_dtypes,
        "context_length": settings.context_length,
        "forward_attention": settings.attention,
        "inputs": ExecutionSchema.from_interface(
            interface, settings.context_length
        ).inputs,
        "input_columns": interface.input_columns,
        "ingestion": _ingestion_contract(interface.ingestion),
        "static_conditioning": static,
        "feature_layout": interface.feature_layout,
        "depth_layouts": interface.depth_layouts.compatibility_signature(columns),
        "image_sources": {
            name: item.model_dump(exclude=media_fields)
            for name, item in interface.image_sources.items()
        },
        "preconditioning_image": interface.preconditioning_image.compatibility_signature()
        if interface.preconditioning_image is not None
        else None,
        "preconditioning_image_encoder": interface.preconditioning_image_encoder,
        "normalize_real_columns": interface.normalize_real_columns,
        "special_token_ids": interface.special_token_ids,
        "input_metadata": {
            column: {
                "column_data_types": interface.column_data_types.get(column),
                "n_classes": interface.n_classes.get(column),
                "id_maps": interface.id_maps.get(column),
                "categorical_hashing": interface.categorical_hashing.get(column),
                "selected_columns_statistics": interface.selected_columns_statistics.get(
                    column
                ),
            }
            for column in columns
        },
    }


def validate_history_transfer(source, destination, interface_name):
    expected = _history_contract(source, interface_name)
    actual = _history_contract(destination, interface_name)
    for field, value in expected.items():
        if value != actual[field]:
            if field == "input_metadata":
                field = next(
                    f"{field}.{column}.{key}"
                    for column, metadata in value.items()
                    for key, entry in metadata.items()
                    if entry != actual[field][column][key]
                )
            raise ValueError(f"history transfer differs at {interface_name}.{field}")


def history_components(network, interface_name):
    route = network.resolve_interface(interface_name)
    return {
        "backbone": network.backbone,
        **{
            f"interfaces.{interface_name}.{name}": component
            for name, component in (
                ("ingestion", route.ingestion),
                ("ingestion_adapter", route.ingestion_adapter),
                ("static_condition_encoder", route.static_condition_encoder),
                ("preconditioning_image_encoder", route.preconditioning_image_encoder),
                ("prefix_projection", route.prefix_projection),
            )
            if component is not None
        },
    }


def history_parameter_ids(network, interface_name):
    return {
        id(parameter)
        for module in history_components(network, interface_name).values()
        for parameter in module.parameters()
    }


def copy_history_weights(source, destination, interface_name):
    """Copy parameters and persistent buffers without adopting source storage or flags."""
    source_modules = history_components(source, interface_name)
    destination_modules = history_components(destination, interface_name)
    if source_modules.keys() != destination_modules.keys():
        raise ValueError(f"history transfer components differ at {interface_name}")
    weights = {}
    for path, module in destination_modules.items():
        incoming = source_modules[path].state_dict()
        expected = module.state_dict()
        if incoming.keys() != expected.keys():
            raise ValueError(f"history transfer tensor keys differ at {path}")
        for name, tensor in expected.items():
            if (
                tensor.shape != incoming[name].shape
                or tensor.dtype != incoming[name].dtype
            ):
                raise ValueError(f"history transfer tensor differs at {path}.{name}")
        weights[path] = incoming
    for path, module in destination_modules.items():
        module.load_state_dict(weights[path], strict=True)
    return list(weights)
