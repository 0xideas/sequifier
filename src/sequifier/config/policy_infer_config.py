"""Inference options for policy interfaces in ordinary portable model artifacts."""

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from sequifier.artifacts.model_artifact import ModelArtifact, read_execution_artifact
from sequifier.config.policy_config import PolicyConfigBase
from sequifier.config.policy_plan import OnlineEnvironmentConfig
from sequifier.helpers import normalize_path


class PolicyRolloutConfig(PolicyConfigBase):
    environment: OnlineEnvironmentConfig
    trajectory_path: str = Field(min_length=1)
    seed_index: int = Field(ge=0)
    max_episode_steps: int | None = Field(default=None, gt=0)


class PolicyInferenceConfig(PolicyConfigBase):
    _artifacts: dict[str, ModelArtifact] = PrivateAttr(default_factory=dict)

    project_root: str = "."
    model_path: str
    execution_mode: Literal["prediction", "policy"] | None = None
    model_type: Literal["policy", "generative"] = "generative"
    model_interface: str | None = None
    data_path: str
    read_format: Literal["csv", "parquet"] = "csv"
    write_format: Literal["csv", "parquet"] = "csv"
    output_path: str | None = None
    output_probabilities: bool = False
    inference_batch_size: int = Field(default=64, gt=0)
    device: str = "cpu"
    seed: int = 1010
    deterministic: bool = True
    desired_return: float | None = Field(default=None, allow_inf_nan=False)
    desired_return_column: str = "desired_return"
    previous_action_column: str | None = None
    previous_action_columns: dict[str, str] = Field(default_factory=dict)
    sequence_id_column: str = "sequenceId"
    item_position_column: str = "itemPosition"
    rollout: PolicyRolloutConfig | None = None
    autoregressive: bool = False
    generation_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution(self):
        if self.rollout is not None:
            if not self.autoregressive or self.generation_steps is None:
                raise ValueError(
                    "policy rollout requires autoregressive: true and generation_steps"
                )
            if self.output_probabilities:
                raise ValueError(
                    "rollout writes a trajectory including action log probabilities"
                )
        elif self.autoregressive or self.generation_steps is not None:
            raise ValueError("policy generation requires an environment rollout")
        return self


def policy_inference_config(
    values: dict, artifacts: dict[str, ModelArtifact] | None = None
) -> PolicyInferenceConfig | None:
    """Dispatch from the persisted interface contract, without training YAML."""
    if values.get("execution_mode") == "prediction":
        return None
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
    if not policy_routes and values.get("model_type") != "policy":
        return None
    if len(paths) != 1 or len(policy_routes) != 1:
        raise ValueError(
            "policy inference requires one artifact and one selected policy interface"
        )
    authored = dict(values)
    authored["model_path"] = paths[0]
    authored["model_interface"] = policy_routes[0]
    config = PolicyInferenceConfig.model_validate(authored)
    config._artifacts = artifacts
    return config
