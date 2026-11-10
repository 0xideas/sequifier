"""Inference options for policy interfaces in ordinary portable model artifacts."""

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from sequifier.artifacts.model_artifact import ModelArtifact, read_execution_artifact
from sequifier.config.inference_common import InferenceExecutionConfig
from sequifier.config.policy_config import PolicyConfigBase
from sequifier.config.policy_plan import OnlineEnvironmentConfig
from sequifier.helpers import normalize_path


class PolicyRolloutConfig(PolicyConfigBase):
    environment: OnlineEnvironmentConfig
    max_steps: int = Field(gt=0)
    trajectory_path: str = Field(min_length=1)
    seed_index: int = Field(ge=0)
    max_episode_steps: int | None = Field(default=None, gt=0)


class PolicyInferenceConfig(InferenceExecutionConfig):
    model_path: str
    model_type: Literal["policy"]
    data_path: str
    output_path: str | None = None
    output_probabilities: bool = False
    action_selection: Literal["deterministic", "stochastic"] = "deterministic"
    desired_return: float | None = Field(default=None, allow_inf_nan=False)
    desired_return_column: str = "desired_return"
    previous_action_columns: dict[str, str] = Field(default_factory=dict)
    sequence_id_column: str = "sequenceId"
    item_position_column: str = "itemPosition"
    rollout: PolicyRolloutConfig | None = None
    # Optional assertions against the portable artifact contract.
    metadata_config_path: str | None = None
    input_columns: list[str] | None = None
    static_columns: list[str] | None = None
    target_columns: list[str] | None = None
    column_data_types: dict[str, str] | None = None
    target_column_types: dict[str, str] | None = None
    training_objective: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int | None = Field(default=None, ge=0)
    prediction_length: int | None = Field(default=None, gt=0)
    decoder_support: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution(self):
        if self.read_format not in {"csv", "parquet"} or self.write_format not in {
            "csv",
            "parquet",
        }:
            raise ValueError("policy inference formats must be csv or parquet")
        if self.rollout is not None and self.output_probabilities:
            raise ValueError(
                "rollout writes a trajectory including action log probabilities"
            )
        return self


class ResolvedPolicyInferenceConfig(PolicyInferenceConfig):
    """Validated policy request with selected artifact/interface and assertions."""

    _artifacts: dict[str, ModelArtifact] = PrivateAttr(default_factory=dict)


def policy_inference_config(
    values: dict, artifacts: dict[str, ModelArtifact] | None = None
) -> ResolvedPolicyInferenceConfig | None:
    """Dispatch from the persisted interface contract, without training YAML."""
    if values.get("model_type") != "policy":
        return None
    from sequifier.artifacts.model_config import resolved_config_from_model_config
    from sequifier.config.infer_config import (
        _assert_metadata_matches_source,
        _execution_source,
    )
    from sequifier.config.metadata import load_dataset_metadata

    request = PolicyInferenceConfig.model_validate(values)
    artifacts = {} if artifacts is None else artifacts
    model_path = values.get("model_path")
    paths = [model_path] if isinstance(model_path, str) else model_path or []
    selected = values.get("model_interface")
    policy_routes = []
    for path in paths:
        if not isinstance(path, str) or not path.lower().endswith(".pt"):
            continue
        if path not in artifacts:
            artifacts[path] = read_execution_artifact(
                normalize_path(path, values.get("project_root", "."))
            )
        artifact = artifacts[path]
        for name, interface in artifact.model_config.values["interfaces"].items():
            if interface.get("policy") is not None and (
                selected is None or selected == name
            ):
                policy_routes.append(name)
    if len(paths) != 1 or len(policy_routes) != 1:
        raise ValueError(
            "policy inference requires one artifact and one selected policy interface"
        )
    selected = policy_routes[0]
    execution, _ = resolved_config_from_model_config(
        artifacts[paths[0]].model_config.values,
        device=request.device,
        interface_name=selected,
    )
    policy = execution.interface(selected).policy
    if policy is None:
        raise ValueError("selected interface has no policy")
    if (
        request.rollout is not None
        and request.rollout.environment.units != policy.units
    ):
        raise ValueError("rollout environment units differ from policy artifact")
    contract, metadata = _execution_source(execution, execution.dataset(selected))
    for key, actual in contract.items():
        if (
            key in request.model_fields_set
            and getattr(request, key) is not None
            and getattr(request, key) != actual
        ):
            raise ValueError(f"{key} differs from policy artifact")
    if request.metadata_config_path:
        explicit = load_dataset_metadata(
            normalize_path(request.metadata_config_path, request.project_root)
        )
        _assert_metadata_matches_source(
            explicit,
            metadata,
            "policy metadata",
            "storage_layout"
            in artifacts[paths[0]].model_config.values["interfaces"][selected],
        )
    authored = request.model_dump()
    authored["model_path"] = paths[0]
    authored["model_interface"] = selected
    config = ResolvedPolicyInferenceConfig.model_validate(authored)
    config._artifacts = artifacts
    return config
