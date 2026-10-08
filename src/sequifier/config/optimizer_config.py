"""Configuration for explicit parameter routing and optimizer groups."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.config.components import ComponentSpec


class OptimizerSelector(BaseModel):
    """Match parameter catalog descriptors; supplied conditions are conjunctive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_groups: list[str] | None = None
    component: Literal["ingestion", "backbone", "decoder"] | None = None
    parameter_kind: Literal["weight", "bias", "other"] | None = None
    ndim: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_selector(self) -> "OptimizerSelector":
        if not any(
            value is not None
            for value in (
                self.semantic_groups,
                self.component,
                self.parameter_kind,
                self.ndim,
            )
        ):
            raise ValueError(
                "An optimizer selector must specify at least one condition."
            )
        if self.semantic_groups is not None and (
            not self.semantic_groups
            or any(not pattern for pattern in self.semantic_groups)
        ):
            raise ValueError("semantic_groups must contain nonempty patterns.")
        return self


class OptimizerGroupSpec(BaseModel):
    """One named route and the optimizer applied to its parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    select: OptimizerSelector | Literal["otherwise"]
    optimizer: ComponentSpec
    learning_rate: float = Field(gt=0)


class OptimizerPlan(BaseModel):
    """Ordered, exhaustive optimizer routes with a final fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: list[OptimizerGroupSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_groups(self) -> "OptimizerPlan":
        identifiers = [group.id for group in self.groups]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Optimizer group IDs must be unique.")
        if self.groups[-1].select != "otherwise" or any(
            group.select == "otherwise" for group in self.groups[:-1]
        ):
            raise ValueError("The final optimizer group must select 'otherwise'.")
        return self
