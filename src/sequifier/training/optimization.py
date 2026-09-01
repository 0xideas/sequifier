"""The complete optimizer transaction for a training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from sequifier.artifacts.run_checkpoint import OptimizationState
from sequifier.integration.callbacks import IntegrationManager
from sequifier.integration.contexts import (
    BackwardCompleted,
    GradientsClipped,
    GradientsUnscaled,
    OptimizerStepCompleted,
    OptimizerStepStarting,
    StepIdentity,
    TrainingAccess,
)
from sequifier.integration.controls import apply_training_directive
from sequifier.model.parameter_catalog import ParameterCatalog
from sequifier.optimizers.optimizers import get_optimizer_class, get_scheduler_class


@dataclass(frozen=True)
class SchedulerPolicy:
    step_on: str


@dataclass(frozen=True)
class GradientPolicy:
    accumulation_steps: int
    clip_norm: float | None


@dataclass(frozen=True)
class UpdatePolicy:
    frozen_parameter_ids: frozenset[int] = frozenset()
    gradient_divisor: int = 1


@dataclass(frozen=True)
class StepResult:
    applied: bool
    overflow: bool
    stop_requested: bool


@dataclass
class OptimizationRuntime:
    optimizer: Optimizer
    scheduler: Any
    scaler: GradScaler
    scheduler_policy: SchedulerPolicy
    gradient_policy: GradientPolicy
    optimizer_step: int = 0
    skip_next_scheduler_step: bool = False

    @classmethod
    def create(
        cls,
        training: Any,
        device: str,
        parameters: Iterable[nn.Parameter] | list[dict[str, Any]],
    ) -> "OptimizationRuntime":
        optimizer_class = get_optimizer_class(training.optimizer.name)
        optimizer = optimizer_class(
            parameters,
            lr=training.learning_rate,
            **training.optimizer.arguments,
        )
        scheduler_class = get_scheduler_class(training.scheduler.name)
        scheduler = scheduler_class(optimizer, **training.scheduler.arguments)
        use_scaler = bool(
            training.layer_type_dtypes
            and "float16" in training.layer_type_dtypes.values()
        )
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=GradScaler(device=device.split(":")[0], enabled=use_scaler),
            scheduler_policy=SchedulerPolicy(training.scheduler_step_on),
            gradient_policy=GradientPolicy(
                accumulation_steps=training.accumulation_steps or 1,
                clip_norm=training.gradient_clip,
            ),
        )

    def access(self, network: nn.Module) -> TrainingAccess:
        return TrainingAccess(
            model=network,
            parameter_catalog=ParameterCatalog(network),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

    def accumulate(
        self,
        loss: Any,
        identity: StepIdentity,
        integrations: IntegrationManager,
        network: nn.Module,
    ) -> None:
        self.scaler.scale(loss).backward()
        if integrations.enabled:
            integrations.emit(
                BackwardCompleted(
                    access=self.access(network),
                    identity=identity,
                    gradients_are_scaled=self.scaler.is_enabled(),
                    optimizer_step_due=False,
                )
            )

    def complete_step(
        self,
        network: nn.Module,
        identity: StepIdentity,
        integrations: IntegrationManager,
        policy: UpdatePolicy,
    ) -> StepResult:
        if policy.gradient_divisor <= 0:
            raise ValueError("gradient_divisor must be positive.")
        access = self.access(network)
        self.scaler.unscale_(self.optimizer)
        parameters = tuple(network.parameters())
        for parameter in parameters:
            if id(parameter) in policy.frozen_parameter_ids:
                parameter.grad = None
            elif policy.gradient_divisor != 1 and parameter.grad is not None:
                parameter.grad.div_(float(policy.gradient_divisor))
        event = GradientsUnscaled(access=access, identity=identity)
        if integrations.enabled:
            integrations.emit(event)
        directive = integrations.directive(event)
        if directive is not None:
            apply_training_directive(
                self.optimizer, directive, scheduler=self.scheduler
            )
        clip_norm = self.gradient_policy.clip_norm
        if directive is not None:
            if directive.disable_gradient_clipping:
                clip_norm = None
            elif directive.gradient_clip_norm is not None:
                clip_norm = directive.gradient_clip_norm
        if clip_norm is not None:
            active_parameters = [p for p in parameters if p.grad is not None]
            total_norm = nn.utils.clip_grad_norm_(active_parameters, clip_norm)
            if integrations.enabled:
                integrations.emit(
                    GradientsClipped(
                        access=access,
                        identity=identity,
                        max_norm=float(clip_norm),
                        total_norm=total_norm,
                    )
                )
        skip = bool(directive is not None and directive.skip_optimizer_step)
        if integrations.enabled:
            integrations.emit(
                OptimizerStepStarting(
                    access=access,
                    identity=identity,
                    skip_optimizer_step=skip,
                    reason=None if directive is None else directive.reason,
                )
            )
        previous_scale = self.scaler.get_scale()
        applied = False
        if not skip:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            applied = (
                not self.scaler.is_enabled()
                or self.scaler.get_scale() >= previous_scale
            )
        else:
            self.scaler.update()
        overflow = not skip and not applied
        if applied:
            self.optimizer_step += 1
            if self.scheduler_policy.step_on == "batch":
                if directive is None or not directive.skip_scheduler_step:
                    self.step_scheduler()
            elif directive is not None and directive.skip_scheduler_step:
                self.skip_next_scheduler_step = True
            if integrations.enabled:
                integrations.emit(
                    OptimizerStepCompleted(
                        access=access,
                        identity=StepIdentity(
                            epoch=identity.epoch,
                            batch=identity.batch,
                            global_batch_step=identity.global_batch_step,
                            optimizer_step=self.optimizer_step,
                            accumulation_index=identity.accumulation_index,
                            accumulation_steps=identity.accumulation_steps,
                            rank=identity.rank,
                            world_size=identity.world_size,
                        ),
                    )
                )
        self.optimizer.zero_grad(set_to_none=True)
        return StepResult(
            applied=applied,
            overflow=overflow,
            stop_requested=bool(directive is not None and directive.stop_after_step),
        )

    def step_scheduler(self) -> bool:
        if self.skip_next_scheduler_step:
            self.skip_next_scheduler_step = False
            return False
        if (
            hasattr(self.scheduler, "total_steps")
            and self.scheduler.last_epoch >= self.scheduler.total_steps
        ):
            return False
        self.scheduler.step()
        return True

    def state_dict(
        self, optimizer_state: dict[str, Any] | None = None
    ) -> OptimizationState:
        return OptimizationState(
            optimizer=optimizer_state or self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            optimizer_step=self.optimizer_step,
        )

    def load_non_optimizer_state(self, state: OptimizationState) -> None:
        self.scheduler.load_state_dict(state.scheduler)
        self.scaler.load_state_dict(state.scaler)
        self.optimizer_step = int(state.optimizer_step)
