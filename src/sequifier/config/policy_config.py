"""Policy alignment and standard model-component configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.io.values import DomainValue

NonemptyStr = Annotated[str, Field(min_length=1)]


class PolicyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReturnDefinition(PolicyConfigBase):
    gamma: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    finite_horizon: int | None = Field(default=None, gt=0)


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
        for counts, columns in (
            (self.observation_n_classes, self.observation_columns),
            (self.static_n_classes, self.static_columns),
        ):
            if counts.keys() - set(columns):
                raise ValueError(
                    "class counts must name declared observation/static columns"
                )
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


class CategoricalPolicyConfig(PolicySchemaConfig):
    """Resolved alignment metadata; architecture belongs to the ordinary model."""

    # Older artifacts embedded model components here. The model bundle owns them.
    model_config = ConfigDict(extra="ignore", frozen=True)
    context_length: int = Field(gt=0)
    interface_name: NonemptyStr
