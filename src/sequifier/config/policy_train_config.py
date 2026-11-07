"""Authored and resolved trajectory training contracts for the train command."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    PolicyConfigBase,
    PolicyOptimizationConfig,
)


class TrajectorySourceConfig(PolicyConfigBase):
    paths: list[str] = Field(default_factory=list)
    manifest: str | None = None
    action_semantics: Literal["policy", "executed_predictions"] = "policy"

    @model_validator(mode="after")
    def validate_location(self):
        if bool(self.paths) == (self.manifest is not None):
            raise ValueError("trajectory source requires either paths or one manifest")
        if any(not path for path in self.paths) or self.manifest == "":
            raise ValueError("trajectory paths must be nonempty")
        return self


class ReturnConditionedPhaseConfig(PolicyConfigBase):
    algorithm: Literal["return_conditioned"]
    name: str = Field(default="policy", min_length=1)
    epochs: int = Field(gt=0)
    trajectory_sources: list[str] = Field(min_length=1)
    validation_sources: list[str] | None = Field(default=None, min_length=1)
    policy: str = Field(min_length=1)


class PolicyTrainingPlanConfig(PolicyConfigBase):
    phases: list[ReturnConditionedPhaseConfig] = Field(min_length=1, max_length=1)


class _PolicyTrainConfigBase(PolicyConfigBase):
    project_root: str = "."
    model_name: str = Field(min_length=1)
    device: Literal["cpu"] = "cpu"
    seed: int = Field(default=1010, ge=0)
    trajectory_sources: dict[str, TrajectorySourceConfig] = Field(min_length=1)
    policies: dict[str, CategoricalPolicyConfig] = Field(min_length=1)
    training_plan: PolicyTrainingPlanConfig
    # Epochs and seed belong to the phase and run respectively.
    optimization: PolicyOptimizationConfig = Field(
        default_factory=PolicyOptimizationConfig
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("model_name must be a single path component")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_plan(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        plan = values.get("training_plan")
        if isinstance(plan, dict) and "phases" not in plan:
            values["training_plan"] = {"phases": [plan]}
        optimization = values.get("optimization", {})
        if isinstance(optimization, dict) and {"epochs", "seed"} & optimization.keys():
            raise ValueError("set epochs on the phase and seed on the run")
        return values

    @model_validator(mode="after")
    def validate_references(self):
        phase = self.training_plan.phases[0]
        if phase.policy not in self.policies:
            raise ValueError(f"unknown policy {phase.policy!r}")
        for names in (phase.trajectory_sources, phase.validation_sources or []):
            if len(names) != len(set(names)):
                raise ValueError("trajectory source references must be unique")
            unknown = set(names) - self.trajectory_sources.keys()
            if unknown:
                raise ValueError(f"unknown trajectory sources: {sorted(unknown)}")
        if set(phase.trajectory_sources) & set(phase.validation_sources or []):
            raise ValueError("training and validation sources must be disjoint")
        return self

    @field_validator("actor_path", "report_path", check_fields=False)
    @classmethod
    def validate_output_path(cls, value):
        if value == "":
            raise ValueError("output paths must be nonempty")
        return value


class PolicyTrainConfig(_PolicyTrainConfigBase):
    """Authored paths may be omitted to select the standard output locations."""

    actor_path: str | None = None
    report_path: str | None = None


class ResolvedPolicyTrainConfig(_PolicyTrainConfigBase):
    """Run paths are absolute and the authored phase has been validated."""

    actor_path: str
    report_path: str


def resolve_policy_train_config(config: PolicyTrainConfig) -> ResolvedPolicyTrainConfig:
    root = Path(config.project_root).resolve()

    def absolute(path):
        return str((root / path).resolve())

    values = config.model_dump(mode="json")
    values["project_root"] = str(root)
    for source in values["trajectory_sources"].values():
        source["paths"] = [absolute(path) for path in source["paths"]]
        if source["manifest"] is not None:
            source["manifest"] = absolute(source["manifest"])
    values["actor_path"] = absolute(
        config.actor_path or f"models/{config.model_name}/actor.pt"
    )
    values["report_path"] = absolute(
        config.report_path or f"outputs/training/{config.model_name}/report.json"
    )
    if values["actor_path"] == values["report_path"]:
        raise ValueError("actor_path and report_path must differ")
    return ResolvedPolicyTrainConfig.model_validate(values)
