"""Policy alignment and standard model-component configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from sequifier.config.components import (
    BackboneArchitectureConfig,
    ComponentSpec,
    DecoderComponentConfig,
    IngestionComponentConfig,
    ResumeConfig,
)


class PolicyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReturnDefinition(PolicyConfigBase):
    gamma: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    finite_horizon: int | None = Field(default=None, gt=0)


class PolicyModelConfig(PolicyConfigBase):
    context_length: int = Field(default=4, gt=0)
    backbone: BackboneArchitectureConfig
    ingestion: IngestionComponentConfig | None = None
    decoder: DecoderComponentConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_backbone(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "backbone" not in values:
            width = 32
            values["backbone"] = {
                "dim_model": width,
                "num_layers": 2,
                "max_context_length": values.get("context_length", 4),
                "attention": {"n_heads": 1},
                "feed_forward": {"dim": width * 4},
                "positional_encoding_scope": "global",
            }
        elif isinstance(values["backbone"], dict):
            values["backbone"] = dict(values["backbone"])
            values["backbone"].setdefault("positional_encoding_scope", "global")
        return values

    @model_validator(mode="after")
    def validate_backbone(self):
        if self.context_length > self.backbone.max_context_length:
            raise ValueError(
                "policy context_length exceeds backbone max_context_length"
            )
        return self


class CategoricalPolicyConfig(PolicyModelConfig):
    alignment: Literal["current_observation_previous_action"] = (
        "current_observation_previous_action"
    )
    observation_columns: list[str] = Field(min_length=1)
    static_columns: list[str] = Field(default_factory=list)
    # Numeric trajectory fields may also carry already encoded categorical IDs.
    observation_n_classes: dict[
        str, Annotated[int, Field(strict=True, ge=2, le=2**24)]
    ] = Field(default_factory=dict)
    action_column: str = Field(default="action", min_length=1)
    action_values: list[FiniteFloat] = Field(
        default_factory=lambda: [0.0, 1.0], min_length=2
    )
    return_conditioned: bool = True
    returns: ReturnDefinition = Field(default_factory=ReturnDefinition)

    @model_validator(mode="after")
    def validate_names(self):
        names = self.observation_columns + self.static_columns
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "policy observation/static columns must be nonempty and disjoint"
            )
        reserved = {"__policy_previous_action", "__policy_return", "__policy_action"}
        if reserved.intersection(names + [self.action_column]):
            raise ValueError("policy feature names use reserved internal columns")
        if set(self.observation_n_classes) - set(self.observation_columns) or any(
            count < 2 for count in self.observation_n_classes.values()
        ):
            raise ValueError(
                "observation_n_classes requires observation columns with at least two classes"
            )
        if self.decoder is not None and (
            self.decoder.support != 1 or self.decoder.prediction_length != 1
        ):
            raise ValueError(
                "policy decoders require support=1 and prediction_length=1"
            )
        if len(self.action_values) != len(set(self.action_values)):
            raise ValueError("policy action_values must be unique")
        return self


class PolicyOptimizationConfig(PolicyConfigBase):
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=0, ge=0)
    episode_cache_size: int = Field(default=8, gt=0)
    resume: ResumeConfig | None = None
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    learning_rate: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    optimizer: ComponentSpec = Field(default_factory=lambda: ComponentSpec(name="Adam"))
    scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1}
        )
    )
    scheduler_step_on: Literal["epoch"] = "epoch"
    accumulation_steps: Literal[1] = 1
    layer_type_dtypes: None = None


class PolicyTrainingConfig(PolicyOptimizationConfig):
    epochs: int = Field(default=80, gt=0)
    seed: int = Field(default=1010, ge=0)
