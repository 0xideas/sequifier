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
from sequifier.rl.policy import CategoricalPolicy


def validate_policy_weights(policy: CategoricalPolicy) -> None:
    for name, value in policy.network.state_dict().items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"policy tensor {name!r} requires FP32 precision")
        if not torch.isfinite(value).all():
            raise ValueError(f"policy tensor {name!r} must be finite")
    statistics = policy.interface.selected_columns_statistics
    policy.set_normalization_statistics(statistics)


def save_policy(
    policy: CategoricalPolicy, path: str | Path, *, metadata: dict | None = None
) -> None:
    validate_policy_weights(policy)
    save_model_artifact(
        build_model_artifact(
            policy.network,
            policy.execution_config,
            provenance={
                **(metadata or {}),
                "policy_units": dict(getattr(policy, "units", {})),
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
) -> CategoricalPolicy:
    if isinstance(path, ModelArtifact):
        network, config, artifact = restore_model_artifact(
            path, device=device, interface_name=interface_name
        )
    else:
        network, config, artifact = load_model_artifact(
            path, device=device, interface_name=interface_name
        )
    if interface_name is None:
        candidates = [
            name
            for name, dataset in config.dataset_training.items()
            if dataset.interface.policy is not None
        ]
        if len(candidates) != 1:
            raise ValueError("Select one policy model_interface")
        interface_name = candidates[0]
    assert interface_name is not None
    policy = CategoricalPolicy.from_network(network, config, interface_name)
    policy.units = dict(artifact.metadata.provenance.get("policy_units", {}))
    validate_policy_weights(policy)
    policy.eval()
    return policy
