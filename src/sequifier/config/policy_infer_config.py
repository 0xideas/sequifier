"""Inference options for policy interfaces in ordinary portable model artifacts."""

from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr, model_validator

from sequifier.config.image_config import MediaPath
from sequifier.config.inference_common import (
    InferenceExecutionConfig,
    ResolvedInferenceModel,
)
from sequifier.config.policy_config import PolicyConfigBase
from sequifier.config.policy_plan import OnlineEnvironmentConfig
from sequifier.config.rollout_config import RolloutSelection

if TYPE_CHECKING:
    from sequifier.trajectory.sources import ResolvedSequenceSource


class PolicyRolloutConfig(RolloutSelection):
    model_config = PolicyConfigBase.model_config
    environment: OnlineEnvironmentConfig
    max_steps: int = Field(gt=0)
    seed_history: Literal["available", "current"] = "available"
    seed_row_index: int | None = Field(default=None, ge=0)
    max_episode_steps: int | None = Field(default=None, gt=0)


class PolicyInferenceConfig(InferenceExecutionConfig):
    model_paths: list[str] = Field(min_length=1, max_length=1)
    task: Literal["policy"]

    training_config_path: str | None = None
    data_path: str | None = None
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None
    output_probabilities: bool = False
    action_selection: Literal["deterministic", "stochastic"] = "deterministic"
    desired_return: float | None = Field(default=None, allow_inf_nan=False)
    desired_return_column: str = "desired_return"
    previous_action_columns: dict[str, str] = Field(default_factory=dict)
    rollout: PolicyRolloutConfig | None = None
    # Optional assertions against the portable artifact contract.
    input_columns: list[str] | None = None
    static_columns: list[str] | None = None
    target_columns: list[str] | None = None
    column_data_types: dict[str, str] | None = None
    target_column_types: dict[str, str] | None = None
    training_objective: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int | None = Field(default=None, ge=0)
    prediction_length: int | None = Field(default=None, gt=0)
    decoder_support_length: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution(self):
        if self.rollout is not None and self.output_probabilities:
            raise ValueError(
                "rollout writes a trajectory including action log probabilities"
            )
        return self


class ResolvedPolicyInferenceConfig(PolicyInferenceConfig):
    """Validated policy request with selected artifact/interface and assertions."""

    _model: ResolvedInferenceModel = PrivateAttr()
    _prepared_source: "ResolvedSequenceSource | None" = PrivateAttr(default=None)
    _indexed_view: object | None = PrivateAttr(default=None)

    def input_paths(self):
        paths = super().input_paths()
        if self.training_config_path:
            paths.append(self.training_config_path)
        if self._prepared_source is not None:
            paths.extend(self._prepared_source.dependencies.identity)
        return paths
