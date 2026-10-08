"""Configuration for explicit parameter routing and optimizer groups."""

from __future__ import annotations

from typing import Any, Literal

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
    base_learning_rate: float | None = Field(default=None, ge=0)


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


def resolve_plan_scheduler_arguments(
    plan: OptimizerPlan, scheduler: ComponentSpec
) -> dict[str, Any]:
    """Resolve scheduler rate boundaries from the ordered optimizer plan."""

    arguments = dict(scheduler.arguments)
    if scheduler.name in {"OneCycleLR", "CyclicLR"}:
        if arguments.get("cycle_momentum", False) is not False:
            raise ValueError(
                "cycle_momentum must be false with an optimizer plan because "
                "the composite optimizer has no shared momentum setting."
            )
        arguments["cycle_momentum"] = False
    if scheduler.name == "CyclicLR":
        if "base_lr" in arguments or "max_lr" in arguments:
            raise ValueError(
                "global_training.scheduler.base_lr and max_lr must be omitted "
                "with an optimizer plan; set rates in each optimizer group."
            )
        missing = [
            group.id for group in plan.groups if group.base_learning_rate is None
        ]
        if missing:
            raise ValueError(
                "CyclicLR requires base_learning_rate in every optimizer group; "
                f"missing: {missing!r}."
            )
        for group in plan.groups:
            assert group.base_learning_rate is not None
            if group.base_learning_rate >= group.learning_rate:
                raise ValueError(
                    f"Optimizer group {group.id!r} must have base_learning_rate "
                    "less than learning_rate for CyclicLR."
                )
        arguments["base_lr"] = [group.base_learning_rate for group in plan.groups]
        arguments["max_lr"] = [group.learning_rate for group in plan.groups]
    else:
        if any(group.base_learning_rate is not None for group in plan.groups):
            raise ValueError(
                "Optimizer group base_learning_rate is only used with CyclicLR."
            )
        if scheduler.name == "OneCycleLR":
            if "max_lr" in arguments:
                raise ValueError(
                    "global_training.scheduler.max_lr must be omitted with an "
                    "optimizer plan; each group's learning_rate is its peak rate."
                )
            arguments["max_lr"] = [group.learning_rate for group in plan.groups]
    return arguments
