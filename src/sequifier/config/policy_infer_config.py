"""Inference options for policy interfaces in ordinary portable model artifacts."""

from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator

from sequifier.config.policy_config import PolicyConfigBase
from sequifier.config.rollout_config import RolloutConfig


class PolicyInferenceConfig(PolicyConfigBase):
    project_root: str = "."
    model_path: str
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
    sequence_id_column: str = "sequenceId"
    rollout: RolloutConfig | None = None
    autoregressive: bool = False
    generation_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution(self):
        if self.rollout is not None:
            if self.rollout.mode != "policy":
                raise ValueError("policy artifacts require rollout.mode: policy")
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


def policy_inference_config(values: dict) -> PolicyInferenceConfig | None:
    """Dispatch from the persisted interface contract, without training YAML."""
    model_path = values.get("model_path")
    paths = [model_path] if isinstance(model_path, str) else model_path or []
    selected = values.get("model_interface")
    policy_routes = []
    for path in paths:
        if not isinstance(path, str) or not path.lower().endswith(".pt"):
            continue
        payload = torch.load(
            Path(values.get("project_root", ".")) / path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("artifact_type") == "sequifier_run_checkpoint":
            payload = payload["model"]
        for name, interface in (
            payload.get("model_config", {}).get("interfaces", {}).items()
        ):
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
    return PolicyInferenceConfig.model_validate(authored)
