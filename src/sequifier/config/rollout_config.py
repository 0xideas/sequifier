"""Opt-in configuration for local, single-environment forecast rollouts."""

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.naming import ConfigModel


class InputBinding(ConfigModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: str = Field(min_length=1)
    source: Literal["prediction", "environment"]
    key: str = Field(min_length=1)


class AdapterConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    factory: str
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def protocol_version(self):
        return 1

    @field_validator("factory")
    @classmethod
    def validate_factory(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if not separator or not module or not attribute.isidentifier():
            raise ValueError(
                "environment.factory must be a 'module:factory' import path"
            )
        return value


class RolloutConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(gt=0)
    environment: AdapterConfig
    bindings: list[InputBinding]
    trajectory_path: str = Field(min_length=1)
    # Index over prepared windows in dataset/chunk order, before seed filtering.
    seed_window_index: int = Field(ge=0)
    max_episode_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def reject_duplicate_producers(self):
        names = [binding.input for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("rollout bindings contain duplicate input producers")
        return self

    def validate_bindings(self, inputs: list[str], targets: list[str]) -> None:
        if len(inputs) != len(set(inputs)) or len(targets) != len(set(targets)):
            raise ValueError("rollout input and target columns must be unique")
        bound = {binding.input for binding in self.bindings}
        if bound != set(inputs):
            raise ValueError(
                "rollout requires exactly one producer per changing input; "
                f"missing={sorted(set(inputs) - bound)}, "
                f"unknown/static={sorted(bound - set(inputs))}"
            )
        unknown = {
            binding.key
            for binding in self.bindings
            if binding.source == "prediction" and binding.key not in targets
        }
        if unknown:
            raise ValueError(
                f"rollout bindings reference unknown predictions: {sorted(unknown)}"
            )
