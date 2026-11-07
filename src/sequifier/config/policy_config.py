"""Policy alignment and standard model-component configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.config.components import (
    BackboneArchitectureConfig,
    BackboneComponentConfig,
    ComponentSpec,
    DecoderComponentConfig,
    IngestionComponentConfig,
    ResumeConfig,
)
from sequifier.io.values import DomainValue


class PolicyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReturnDefinition(PolicyConfigBase):
    gamma: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    finite_horizon: int | None = Field(default=None, gt=0)


class PolicyModelConfig(PolicyConfigBase):
    context_length: int = Field(default=4, gt=0)
    interface_name: str = Field(default="policy", min_length=1)
    backbone: BackboneArchitectureConfig
    backbone_component: BackboneComponentConfig | None = None
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
        if self.backbone_component is not None:
            if self.backbone_component.architecture != self.backbone:
                raise ValueError(
                    "backbone component architecture must match the policy backbone"
                )
            if self.backbone_component.repository is not None:
                raise ValueError(
                    "policy phases currently use initialize_from instead of backbone repositories"
                )
        return self


class PolicySchemaConfig(PolicyConfigBase):
    alignment: Literal["current_observation_previous_action"] = (
        "current_observation_previous_action"
    )
    observation_columns: list[str] = Field(min_length=1)
    static_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    # Numeric trajectory fields may also carry already encoded categorical IDs.
    observation_n_classes: dict[str, Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=dict
    )
    static_n_classes: dict[str, Annotated[int, Field(strict=True, ge=2)]] = Field(
        default_factory=dict
    )
    # The same label -> global ID convention as preprocessing metadata.
    id_maps: dict[str, dict[str | int, int]] = Field(default_factory=dict)
    categorical_encoding: Literal["global", "legacy"] = "global"
    action_column: str = Field(default="action", min_length=1)
    action_values: list[DomainValue] = Field(
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
        if set(self.static_n_classes) - set(self.static_columns):
            raise ValueError("static_n_classes must name static columns")
        counts = {**self.observation_n_classes, **self.static_n_classes}
        if len(set(self.categorical_columns)) != len(self.categorical_columns) or set(
            self.categorical_columns
        ) - set(names):
            raise ValueError(
                "categorical_columns must name unique observation/static features"
            )
        if set(self.id_maps) - set(counts):
            raise ValueError("id_maps must name declared categorical inputs")
        for column, mapping in self.id_maps.items():
            if (
                not mapping
                or len(set(mapping.values())) != len(mapping)
                or any(
                    value < 0 or value >= counts[column] for value in mapping.values()
                )
            ):
                raise ValueError(f"{column}: invalid categorical ID map")
        if len(self.action_values) != len(set(self.action_values)):
            raise ValueError("policy action_values must be unique")
        if len({isinstance(value, str) for value in self.action_values}) != 1:
            raise ValueError(
                "action_values must use consistently numeric or string labels"
            )
        return self

    @property
    def categorical_n_classes(self) -> dict[str, int]:
        return {**self.observation_n_classes, **self.static_n_classes}


class CategoricalPolicyConfig(PolicySchemaConfig, PolicyModelConfig):
    @model_validator(mode="after")
    def validate_decoder(self):
        if self.decoder is not None and (
            self.decoder.support != 1 or self.decoder.prediction_length != 1
        ):
            raise ValueError(
                "policy decoders require support=1 and prediction_length=1"
            )
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
