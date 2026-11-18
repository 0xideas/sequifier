from typing import Any

from sequifier.model.execution import ModelExecution
from sequifier.model.execution_schema import ExecutionSchema
from sequifier.typechecking import beartype


@beartype
def model_execution_config(config: Any) -> dict[str, Any]:
    """Return only configuration required to reconstruct model execution."""

    execution = ModelExecution.from_config(config)
    spec = execution.runtime_settings
    interfaces = {}
    for name, interface in execution.interfaces.items():
        values = interface.model_dump(
            mode="python",
            exclude={
                "name": True,
                "window_view": True,
                "policy": True,
                "image_sources": {"__all__": {"path"}},
                "preconditioning_image": {"paths"},
            },
        )
        values["execution_schema"] = ExecutionSchema.from_interface(
            interface, spec.context_length
        ).to_dict()
        values["depth_layouts"] = interface.depth_layouts.model_dump(mode="json")
        interfaces[name] = values
        if interface.policy is not None:
            values["policy"] = interface.policy.model_dump(mode="json")
    return {
        "training_objective": spec.training_objective,
        "forward_attention": spec.forward_attention
        or ("bidirectional" if spec.training_objective == "bert" else "causal"),
        "context_length": spec.context_length,
        "target_offset": spec.target_offset,
        "backbone": execution.model.backbone.model_dump(
            mode="python",
            exclude={"repository"},
        ),
        "embedding_layer_names": list(execution.embedding_layer_names),
        "layer_type_dtypes": spec.layer_type_dtypes,
        "interfaces": interfaces,
    }
