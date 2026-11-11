"""Attach a policy task to a resolved model without rebuilding its interface."""

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_inputs import PolicyInputLayout
from sequifier.model.execution_schema import ExecutionSchema


def resolve_binding(execution, interface_name, binding):
    interface = execution.interface(interface_name)
    schema = ExecutionSchema.from_interface(
        interface, execution.global_training.context_length
    )
    if (
        -binding.decision_position
        > execution.global_training.context_length
        - interface.decoder.support_length
        + 1
    ):
        raise ValueError("decision_position lies outside the source decoder output")
    values = binding.model_dump()
    actions = dict(binding.actions)
    if not actions:
        raise ValueError(
            "model attachment requires named actions with explicit real scales"
        )
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
            actions[name] = field.model_copy(
                update={
                    "values": [
                        inverse.get(i, i) for i in interface.target_decoder_ids[name]
                    ]
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
    conditioned = binding.return_conditioned
    if conditioned and "__policy_return" not in temporal:
        raise ValueError(
            "return conditioning requires an explicit __policy_return model input"
        )
    layout = PolicyInputLayout.select(
        temporal,
        static,
        binding,
        [k for k in interface.categorical_columns if k in temporal],
        interface.n_classes,
        interface.id_maps,
    )
    values.update(
        **layout.binding_values(),
        actions=actions,
        action_order=[name for name in interface.target_columns if name in actions],
        return_conditioned=conditioned,
        input_schema=schema.to_dict(),
        input_statistics={
            name: stats
            for name, stats in interface.selected_columns_statistics.items()
            if name in set(temporal) | set(static) and interface.normalize_real_columns
        },
        context_length=execution.global_training.context_length,
        interface_name=interface_name,
    )
    for column in binding.previous_action_inputs:
        if column not in temporal:
            raise ValueError(f"{column}: previous action binding is not a model input")
    return ResolvedPolicyBinding.model_validate(values)


def apply_binding(execution, interface_name, policy):
    """Attach task behavior without changing model-owned distribution parameters."""
    interface = execution.interface(interface_name)
    for name, action in policy.actions.items():
        if action.scale == "model" and name not in interface.prediction_distributions:
            raise ValueError(
                f"{name}: scale: model requires a model prediction_distributions entry"
            )
    interface.policy = policy


def attach_policy(network, execution, interface_name, binding):
    from sequifier.rl.policy import Policy

    original_parameters = {name for name, _ in network.named_parameters()}
    policy = resolve_binding(execution, interface_name, binding)
    apply_binding(execution, interface_name, policy)
    actor = Policy.from_network(network, execution, interface_name)
    actor.transfer_report = {
        "mode": "adapt"
        if binding.previous_action_inputs
        or any(
            field.lower is not None or field.upper is not None
            for field in binding.actions.values()
        )
        or any(
            field.type == "categorical"
            and set(actor.action_decoder_ids(name))
            != set(actor.interface.target_decoder_ids[name])
            for name, field in policy.actions.items()
        )
        else "attach",
        "interface": interface_name,
        "reused_parameters": [
            name
            for name, _ in network.named_parameters()
            if name in original_parameters
        ],
        "new_parameters": [
            name
            for name, _ in network.named_parameters()
            if name not in original_parameters
        ],
        "source_preprocessing": "preserved",
        "action_support": {
            name: field.model_dump(mode="json")
            for name, field in policy.actions.items()
        },
    }
    return actor
