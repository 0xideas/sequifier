"""Policies use the same portable model and run-checkpoint artifacts as forecasts."""

from pathlib import Path

import torch

from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    build_model_artifact,
    load_model_artifact,
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
    validate_policy_weights(policy)
    save_model_artifact(
        build_model_artifact(
            policy.network,
            policy.execution_config,
            provenance={
                **(metadata or {}),
                "policy_units": dict(getattr(policy, "units", {})),
                "transfer": getattr(policy, "transfer_report", None),
            },
            state_dict={
                name: value.detach().cpu()
                for name, value in policy.network.state_dict().items()
            },
        ),
        path,
    )


def load_policy(
    path: str | Path | ModelArtifact,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
    binding=None,
) -> Policy:
    if isinstance(path, ModelArtifact):
        network, config, artifact = restore_model_artifact(
            path, device=device, interface_name=interface_name
        )
    else:
        network, config, artifact = load_model_artifact(
            path, device=device, interface_name=interface_name
        )
    if interface_name is None:
        candidates = (
            list(config.interface_names)
            if binding is not None
            else [
                name
                for name, dataset in config.dataset_training.items()
                if dataset.interface.policy is not None
            ]
        )
        if len(candidates) != 1:
            raise ValueError("Select one policy model_interface")
        interface_name = candidates[0]
    assert interface_name is not None
    if binding is not None:
        from sequifier.rl.attachment import attach_policy

        policy = attach_policy(network, config, interface_name, binding)
    else:
        policy = Policy.from_network(network, config, interface_name)
    policy.units = dict(artifact.metadata.provenance.get("policy_units", {}))
    validate_policy_weights(policy)
    policy.eval()
    return policy
