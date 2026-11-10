"""Concrete optimizer ownership and scheduler budgets for policy phases."""

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


class OptimizationConfig(BaseModel):
    """One authored optimizer/scheduler configuration per trainable role."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    optimizer: OptimizerPlan | ComponentSpec = Field(
        default_factory=default_optimizer, union_mode="left_to_right"
    )
    scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    learning_rate: float = Field(default=3e-4, gt=0)
    gradient_clip: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_components(self):
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
        if (
            self.scheduler.name
            in {"ReduceLROnPlateau", "SequentialLR", "ChainedScheduler"}
            or "optimizer" in self.scheduler.arguments
        ):
            raise ValueError(
                "policy schedulers must advance independently on optimizer steps"
            )
        return self

    def validate_budget(self, step_budget):
        if (
            step_budget is not None
            and self.scheduler.name == "OneCycleLR"
            and self.scheduler.arguments.get("total_steps") != step_budget
        ):
            raise ValueError(
                f"OneCycleLR.total_steps must equal the maximum authored optimizer-step budget ({step_budget})"
            )

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
