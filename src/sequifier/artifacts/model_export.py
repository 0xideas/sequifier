from typing import Any

from sequifier.typechecking import beartype


@beartype
def model_execution_config(training_config: Any) -> dict[str, Any]:
    """Return only configuration required to reconstruct model execution."""

    if not hasattr(training_config, "dataset_training"):
        raise TypeError("Model export requires a canonical training config")
    from sequifier.model.execution_schema import ExecutionSchema

    interfaces = {}
    for dataset in training_config.dataset_training.values():
        interface = dataset.interface
        if interface.name in interfaces:
            continue
        values = interface.model_dump(
            mode="python",
            exclude={
                "name": True,
                "window_view": True,
                "policy": True,
                "image_sources": {"__all__": {"path"}},
                "preconditioning_image": {"path"},
            },
        )
        values["execution_schema"] = ExecutionSchema.from_interface(
            interface, training_config.global_training.context_length
        ).to_dict()
        values["depth_layouts"] = interface.depth_layouts.model_dump(mode="json")
        interfaces[interface.name] = values
        if interface.policy is not None:
            interfaces[interface.name]["policy"] = interface.policy.model_dump(
                mode="json"
            )
    spec = training_config.global_training
    return {
        "training_objective": spec.training_objective,
        "forward_attention": spec.forward_attention
        or ("bidirectional" if spec.training_objective == "bert" else "causal"),
        "context_length": spec.context_length,
        "target_offset": spec.target_offset,
        "next_occurrence_config": (
            spec.next_occurrence_config.model_dump(mode="python")
            if spec.next_occurrence_config is not None
            else None
        ),
        "backbone": training_config.model.backbone.model_dump(
            mode="python",
            exclude={"repository"},
        ),
        "embedding_layer_names": list(training_config.embedding_layer_names),
        "layer_type_dtypes": spec.layer_type_dtypes,
        "interfaces": interfaces,
    }
