"""Concrete optimizer ownership and scheduler budgets for policy phases."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from sequifier.config.components import ComponentSpec
from sequifier.optimizers.optimizers import get_optimizer_class, get_scheduler_class


def default_optimizer():
    return ComponentSpec(name="Adam")


def default_scheduler():
    return ComponentSpec(name="ConstantLR", arguments={"factor": 1.0, "total_iters": 1})


class OptimizationConfig(BaseModel):
    """One authored optimizer/scheduler configuration per trainable role."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    optimizer: ComponentSpec = Field(default_factory=default_optimizer)
    scheduler: ComponentSpec = Field(default_factory=default_scheduler)
    learning_rate: float = Field(default=3e-4, gt=0)
    gradient_clip: float = Field(default=1.0, gt=0)

    def resolve(self, step_budget):
        return OptimizationPlan(
            self.optimizer,
            self.scheduler,
            self.learning_rate,
            self.gradient_clip,
            step_budget,
        )


@dataclass(frozen=True)
class OptimizationPlan:
    optimizer: ComponentSpec
    scheduler: ComponentSpec
    learning_rate: float
    gradient_clip: float
    step_budget: int | None

    def __post_init__(self):
        if self.optimizer.name not in {"Adam", "AdamW", "SGD", "RMSprop"} or {
            "lr",
            "params",
        }.intersection(self.optimizer.arguments):
            raise ValueError(
                "policy optimizers require a supported optimizer and phase learning rate"
            )
        if (
            self.scheduler.name
            in {"ReduceLROnPlateau", "SequentialLR", "ChainedScheduler"}
            or "optimizer" in self.scheduler.arguments
        ):
            raise ValueError(
                "policy schedulers must advance independently on optimizer steps"
            )
        if (
            self.step_budget is not None
            and self.scheduler.name == "OneCycleLR"
            and self.scheduler.arguments.get("total_steps") != self.step_budget
        ):
            raise ValueError(
                f"OneCycleLR.total_steps must equal the maximum authored optimizer-step budget ({self.step_budget})"
            )

    def build(self, module):
        optimizer = get_optimizer_class(self.optimizer.name)(
            [p for p in module.parameters() if p.requires_grad],
            lr=self.learning_rate,
            **self.optimizer.arguments,
        )
        return optimizer, get_scheduler_class(self.scheduler.name)(
            optimizer, **self.scheduler.arguments
        )
