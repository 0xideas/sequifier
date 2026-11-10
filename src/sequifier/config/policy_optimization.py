"""Shared role optimization specs with explicit algorithm scheduling constraints."""

import inspect
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sequifier.config.components import ComponentSpec
from sequifier.config.optimizer_config import (
    OptimizerPlan,
    resolve_plan_scheduler_arguments,
)
from sequifier.optimizers.optimizers import get_optimizer_class, get_scheduler_class


def default_optimizer():
    return ComponentSpec(name="Adam")


def default_scheduler():
    return ComponentSpec(name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1})


class RoleOptimizationConfig(BaseModel):
    """One authored optimizer/scheduler configuration per trainable role."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    optimizer: OptimizerPlan | ComponentSpec = Field(
        default_factory=default_optimizer, union_mode="left_to_right"
    )
    scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    learning_rate: float | None = Field(default=None, gt=0)
    gradient_clip: float | None = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_components(self):
        if isinstance(self.optimizer, OptimizerPlan):
            if self.learning_rate is not None:
                raise ValueError("optimizer plans set learning_rate in each group")
        elif self.learning_rate is None:
            self.learning_rate = 3e-4
        if self.scheduler.name == "OneCycleLR" and {
            "epochs",
            "steps_per_epoch",
        }.intersection(self.scheduler.arguments):
            raise ValueError(
                "OneCycleLR uses the phase budget; configure total_steps instead of scheduler epochs/steps_per_epoch"
            )
        specs = (
            [group.optimizer for group in self.optimizer.groups]
            if isinstance(self.optimizer, OptimizerPlan)
            else [self.optimizer]
        )
        for spec in specs:
            get_optimizer_class(spec.name)
            if {"lr", "params"}.intersection(spec.arguments):
                raise ValueError(
                    "optimizer ownership and learning rates must be configured explicitly"
                )
        arguments = (
            resolve_plan_scheduler_arguments(self.optimizer, self.scheduler)
            if isinstance(self.optimizer, OptimizerPlan)
            else self.scheduler.arguments
        )
        try:
            inspect.signature(get_scheduler_class(self.scheduler.name)).bind(
                object(), **arguments
            )
            for spec in specs:
                inspect.signature(get_optimizer_class(spec.name)).bind(
                    object(), lr=1.0, **spec.arguments
                )
        except TypeError as error:
            raise ValueError(str(error)) from error
        return self

    def build(self, module):
        from sequifier.model.parameter_catalog import ParameterCatalog
        from sequifier.training.optimization import create_optimizer

        optimizer = create_optimizer(
            self.optimizer,
            [p for p in module.parameters() if p.requires_grad],
            self.learning_rate,
            parameter_catalog=ParameterCatalog(module)
            if isinstance(self.optimizer, OptimizerPlan)
            else None,
        )
        arguments = (
            resolve_plan_scheduler_arguments(self.optimizer, self.scheduler)
            if isinstance(self.optimizer, OptimizerPlan)
            else self.scheduler.arguments
        )
        return optimizer, get_scheduler_class(self.scheduler.name)(
            optimizer, **arguments
        )


class OptimizationConfig(RoleOptimizationConfig):
    """Policy roles advance their schedulers once per applied optimizer update."""

    scheduler_step_on: Literal["optimizer_step"] = "optimizer_step"

    def validate_budget(self, step_budget):
        if (
            step_budget is not None
            and self.scheduler.name == "OneCycleLR"
            and self.scheduler.arguments.get("total_steps") != step_budget
        ):
            raise ValueError(
                f"OneCycleLR.total_steps must equal the maximum authored optimizer-step budget ({step_budget})"
            )

    @model_validator(mode="after")
    def validate_policy_scheduler(self):
        if (
            self.scheduler.name
            in {"ReduceLROnPlateau", "SequentialLR", "ChainedScheduler"}
            or "optimizer" in self.scheduler.arguments
        ):
            raise ValueError(
                "policy schedulers must advance independently on optimizer steps"
            )
        return self


class SupervisedOptimizationConfig(RoleOptimizationConfig):
    """Epoch scheduling includes metric-driven ReduceLROnPlateau."""

    scheduler_step_on: Literal["epoch", "batch"] = "epoch"
    scheduler: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            name="StepLR", arguments={"step_size": 1, "gamma": 0.99}
        )
    )

    @model_validator(mode="after")
    def validate_supervised_scheduler(self):
        if (
            self.scheduler.name == "ReduceLROnPlateau"
            and self.scheduler_step_on != "epoch"
        ):
            raise ValueError("ReduceLROnPlateau requires scheduler_step_on: epoch")
        if self.scheduler.name in {"SequentialLR", "ChainedScheduler"}:
            raise ValueError("nested schedulers are not supported")
        if (
            self.scheduler.name == "OneCycleLR"
            and self.scheduler_step_on == "batch"
            and "total_steps" not in self.scheduler.arguments
        ):
            raise ValueError(
                "batch OneCycleLR requires an explicit total_steps optimizer-update budget"
            )
        return self
