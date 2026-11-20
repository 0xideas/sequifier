"""Policy views over shared portable models and execution checkpoints."""

from pathlib import Path

import torch

from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    build_model_artifact,
    read_artifact,
    restore_model_artifact,
    save_model_artifact,
)
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.rl.policy import Policy


def phase_actor_path(directory, index, name):
    return PhaseArtifacts.for_phase(directory, index, name).model


def validate_policy_weights(policy: Policy) -> None:
    for name, value in policy.network.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"policy tensor {name!r} must be finite")
    statistics = policy.interface.selected_columns_statistics
    policy.set_normalization_statistics(statistics)


def save_policy(
    policy: Policy, path: str | Path, *, metadata: dict | None = None
) -> None:
    save_model_artifact(policy_model_artifact(policy, metadata=metadata), path)


def policy_model_artifact(
    policy: Policy,
    *,
    metadata: dict | None = None,
    weights: dict[str, torch.Tensor] | None = None,
) -> ModelArtifact:
    validate_policy_weights(policy)
    if weights is None:
        weights = {
            name: value.detach().cpu()
            for name, value in policy.network.state_dict().items()
        }
    return build_model_artifact(
        policy.network,
        policy.execution_config,
        provenance={
            **(metadata or {}),
            "transfer": policy.transfer_report,
        },
        state_dict=weights,
    )


def load_policy(
    path: str | Path | ModelArtifact,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
    binding=None,
    model=None,
    metadata_config_path: str | Path | None = None,
) -> Policy:
    artifact = (
        path
        if isinstance(path, ModelArtifact)
        else read_artifact(path, expected_type="sequifier_model")[0]
    )
    network, execution, _ = restore_model_artifact(
        artifact, device=device, interface_name=interface_name
    )
    if interface_name is None:
        candidates = (
            list(execution.interface_names)
            if binding is not None
            else [
                name
                for name, interface in execution.interfaces.items()
                if interface.policy is not None
            ]
        )
        if len(candidates) != 1:
            raise ValueError("Select one policy model_interface")
        interface_name = candidates[0]
    assert interface_name is not None
    if model is not None:
        from sequifier.artifacts.model_config import extend_prediction_distributions

        extend_prediction_distributions(execution, model)
    if binding is not None:
        from sequifier.rl.attachment import attach_policy

        policy = attach_policy(network, execution, interface_name, binding)
    else:
        policy = Policy.from_network(network, execution, interface_name)
    if metadata_config_path is not None:
        from sequifier.config.infer_config import validate_policy_metadata

        validate_policy_metadata(execution, interface_name, metadata_config_path)
    policy.config = policy.config.model_copy(
        update={"input_column_types": policy.interface.column_data_types}
    )
    policy.interface.policy = policy.config
    policy.units = dict(policy.config.units)
    validate_policy_weights(policy)
    policy.eval()
    return policy
