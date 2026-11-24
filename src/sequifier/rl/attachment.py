"""Attach a policy task to a resolved model without rebuilding its interface."""

from sequifier.config.policy_config import PolicyBinding, ResolvedPolicyBinding
from sequifier.config.policy_inputs import observation_columns
from sequifier.model.execution_schema import ExecutionSchema


def bind_policy(execution, interface_name, binding=None) -> ResolvedPolicyBinding:
    """Compile the task once and keep its portable declaration on the interface."""
    interface = execution.interface(interface_name)
    if binding is None:
        binding = interface.policy
    if binding is None:
        raise ValueError(f"Model interface {interface_name!r} has no policy alignment")
    schema = ExecutionSchema.from_interface(interface, execution.options.context_length)
    if (
        -binding.decision_position
        > execution.options.context_length - interface.decoder.support_length + 1
    ):
        raise ValueError("decision_position lies outside the source decoder output")
    values = {name: getattr(binding, name) for name in PolicyBinding.model_fields}
    actions = dict(binding.actions)
    for name, field in actions.items():
        if name not in interface.target_columns:
            raise ValueError(f"{name}: policy action is not a model target")
        if field.scale == "model" and name not in interface.prediction_distributions:
            raise ValueError(
                f"{name}: scale: model requires a model prediction_distributions entry"
            )
        if field.type != interface.target_column_types[name]:
            raise ValueError(f"{name}: action type differs from the model")
        if field.type == "categorical" and not field.values:
            inverse = {v: k for k, v in interface.id_maps.get(name, {}).items()}
            decoder_ids = interface.target_decoder_ids[name]
            labels = [inverse[i] for i in decoder_ids if i in inverse]
            # A partial string codec has no domain label for its unused slots.
            # Numeric codecs can also execute a decoder ID as an integer action.
            if not any(isinstance(label, str) for label in labels):
                labels = [inverse.get(i, i) for i in decoder_ids]
            actions[name] = type(field).model_validate(
                {
                    **field.model_dump(),
                    "values": labels,
                }
            )
    static = [
        item.key
        for item in schema.inputs
        if item.role == "static_feature" or item.key == "__preconditioning_image"
    ]
    temporal = [
        item.key
        for item in schema.inputs
        if item.key not in static and item.key != "attention_valid_mask"
    ]
    if binding.return_conditioned and "__policy_return" not in temporal:
        raise ValueError(
            "return conditioning requires an explicit __policy_return model input"
        )
    values.update(
        transforms=interface,
        raw_inputs=execution.interface_storage.get(interface_name),
        observation_columns=observation_columns(temporal, binding),
        static_columns=static,
        actions=actions,
        action_order=[name for name in interface.target_columns if name in actions],
        input_schema=schema,
        context_length=execution.options.context_length,
        interface_name=interface_name,
    )
    for column in binding.previous_action_inputs:
        if column not in temporal:
            raise ValueError(f"{column}: previous action binding is not a model input")
    resolved = ResolvedPolicyBinding.model_validate(values)
    interface.policy = resolved.task_binding
    return resolved


def attachment_report(actor, original_parameters):
    policy = actor.config
    return {
        "mode": "adapt"
        if policy.previous_action_inputs
        or any(
            field.lower is not None or field.upper is not None
            for field in policy.actions.values()
        )
        or any(
            field.type == "categorical"
            and set(actor.action_decoder_ids(name))
            != set(actor.interface.target_decoder_ids[name])
            for name, field in policy.actions.items()
        )
        else "attach",
        "interface": actor.interface_name,
        "reused_parameters": [
            name
            for name, _ in actor.network.named_parameters()
            if name in original_parameters
        ],
        "new_parameters": [
            name
            for name, _ in actor.network.named_parameters()
            if name not in original_parameters
        ],
        "source_preprocessing": "preserved",
        "action_support": {
            name: field.model_dump(mode="json")
            for name, field in policy.actions.items()
        },
    }
