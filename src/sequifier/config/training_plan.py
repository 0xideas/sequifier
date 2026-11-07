"""Shared authored phase contracts; algorithm-specific budgets remain explicit."""

from typing import Annotated, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.identifiers import identifier as _identifier
from sequifier.typechecking import beartype


class TrainingSourceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    weight: Optional[float] = Field(default=None, gt=0)
    batches_per_selection: Optional[int] = Field(default=None, gt=0)

    @field_validator("source")
    @classmethod
    @beartype
    def validate_source(cls, value):
        parts = value.split(".")
        if len(parts) > 2:
            raise ValueError("Source refs use only dataset or dataset.part")
        for part in parts:
            _identifier(part, "Source reference component")
        return value


class TrainingPhaseSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["supervised"] = "supervised"

    name: str
    epochs: int = Field(gt=0)
    mode: Literal["sequential", "interleaved"]
    selection: Optional[Literal["round_robin", "weighted_random"]] = None
    sources: list[TrainingSourceSpecModel] = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    @beartype
    def validate_name(cls, value):
        return _identifier(value, "Training phase name")

    @model_validator(mode="after")
    @beartype
    def validate_mode_fields(self):
        if self.mode == "sequential":
            if self.selection is not None:
                raise ValueError("Sequential phases cannot configure selection")
            if any(source.batches_per_selection is not None for source in self.sources):
                raise ValueError(
                    "batches_per_selection is invalid in sequential phases"
                )
            if any(source.weight is not None for source in self.sources):
                raise ValueError("weight is invalid in sequential phases")
        else:
            self.selection = self.selection or "round_robin"
            if self.selection == "round_robin" and any(
                source.weight is not None for source in self.sources
            ):
                raise ValueError("weight is only valid for weighted_random selection")
        return self


class ReturnConditionedPhaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    algorithm: Literal["return_conditioned"]
    name: str = Field(default="policy", min_length=1)
    epochs: int = Field(gt=0)
    sources: list[TrainingSourceSpecModel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self):
        if any(
            value.weight is not None or value.batches_per_selection is not None
            for value in self.sources
        ):
            raise ValueError(
                "policy sources do not support supervised source scheduling"
            )
        return self

    @property
    def trajectory_sources(self) -> list[str]:
        return [value.source for value in self.sources]

    validation_sources: list[str] | None = Field(default=None, min_length=1)
    policy: str = Field(min_length=1)


Phase = TypeVar("Phase", TrainingPhaseSpecModel, ReturnConditionedPhaseConfig)


class TrainingPlanModel(BaseModel, Generic[Phase]):
    model_config = ConfigDict(extra="forbid")

    curriculum_training: bool = False
    curriculum_column: Optional[str] = Field(default=None, min_length=1)
    phases: list[Phase] = Field(min_length=1)

    @field_validator("phases", mode="before")
    @classmethod
    def default_algorithm(cls, values):
        return [
            {"algorithm": "supervised", **value} if isinstance(value, dict) else value
            for value in values
        ]

    @model_validator(mode="after")
    @beartype
    def validate_unique_names(self):
        if self.curriculum_column is not None and not self.curriculum_training:
            raise ValueError(
                "curriculum_column may only be set when curriculum_training is true"
            )
        if self.curriculum_training and self.curriculum_column is None:
            raise ValueError(
                "curriculum_column must be set when curriculum_training is true"
            )
        if self.curriculum_training and any(
            phase.algorithm != "supervised" for phase in self.phases
        ):
            raise ValueError(
                "curriculum scheduling currently requires supervised phases"
            )
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Training phase names must be unique")
        return self


SupervisedTrainingPlanModel = TrainingPlanModel[TrainingPhaseSpecModel]
PolicyTrainingPlanModel = TrainingPlanModel[ReturnConditionedPhaseConfig]
AlgorithmPhase = Annotated[
    TrainingPhaseSpecModel | ReturnConditionedPhaseConfig,
    Field(discriminator="algorithm"),
]
