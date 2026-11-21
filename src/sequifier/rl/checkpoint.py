"""Policy views over shared portable models and execution checkpoints."""

from pathlib import Path

import torch

from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    build_model_artifact,
    read_artifact,
    restore_execution_network,
    save_model_artifact,
)
from sequifier.artifacts.model_config import model_execution_from_model_config
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.config.inference_common import ResolvedInferenceModel
from sequifier.io.preprocessing import normalization_statistics
from sequifier.rl.attachment import attachment_report, bind_policy
from sequifier.rl.policy import Policy


def phase_actor_path(directory, index, name):
    return PhaseArtifacts.for_phase(directory, index, name).model


def validate_policy_weights(policy: Policy) -> None:
    for name, value in policy.network.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"policy tensor {name!r} must be finite")
    if policy.interface.normalize_real_columns:
        normalization_statistics(policy.interface.selected_columns_statistics)


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
    path: str | Path | ModelArtifact | ResolvedInferenceModel,
    *,
    device: str = "cpu",
    interface_name: str | None = None,
    binding=None,
    model=None,
    metadata_config_path: str | Path | None = None,
) -> Policy:
    if isinstance(path, ResolvedInferenceModel):
        artifact, execution, interface_name = (
            path.artifact,
            path.execution,
            path.interface_name,
        )
        assert artifact is not None and execution is not None
    else:
        artifact = (
            path
            if isinstance(path, ModelArtifact)
            else read_artifact(path, expected_type="sequifier_model")[0]
        )
        execution, _ = model_execution_from_model_config(
            artifact.model_config.values, interface_name=interface_name
        )
    network = restore_execution_network(artifact, execution, device)
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
    resolved = bind_policy(execution, interface_name, binding)
    original_parameters = (
        {name for name, _ in network.named_parameters()}
        if binding is not None
        else None
    )
    policy = Policy.from_network(network, execution, resolved)
    if original_parameters is not None:
        policy.transfer_report = attachment_report(policy, original_parameters)
    if metadata_config_path is not None:
        from sequifier.config.infer_config import validate_policy_metadata

        validate_policy_metadata(execution, interface_name, metadata_config_path)
    validate_policy_weights(policy)
    policy.eval()
    return policy
