from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from sequifier.model.parameter_catalog import ParameterCatalog


@dataclass(frozen=True)
class StepIdentity:
    epoch: int
    batch: int
    global_batch_step: int
    optimizer_step: int
    accumulation_index: int
    accumulation_steps: int
    rank: int
    world_size: int


@dataclass
class TrainingAccess:
    model: nn.Module
    parameter_catalog: ParameterCatalog
    optimizer: Optimizer
    scheduler: Any
    scaler: GradScaler


@dataclass(frozen=True)
class TrainingEvent:
    access: TrainingAccess
    identity: StepIdentity | None = None


@dataclass(frozen=True)
class ModelReady(TrainingEvent):
    pass


@dataclass(frozen=True)
class BatchPrepared(TrainingEvent):
    inputs: dict[str, Tensor] = field(default_factory=dict)
    targets: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardCompleted(TrainingEvent):
    outputs: dict[str, Tensor] = field(default_factory=dict)
    captures: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class LossComputed(TrainingEvent):
    loss: Tensor | None = None
    backward_loss: Tensor | None = None


@dataclass(frozen=True)
class BackwardCompleted(TrainingEvent):
    gradients_are_scaled: bool = False
    optimizer_step_due: bool = False


@dataclass(frozen=True)
class GradientsUnscaled(TrainingEvent):
    reduced_summary: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GradientsClipped(TrainingEvent):
    max_norm: float = 0.0
    total_norm: Tensor | float | None = None


@dataclass(frozen=True)
class OptimizerStepStarting(TrainingEvent):
    skip_optimizer_step: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class OptimizerStepCompleted(TrainingEvent):
    pass


@dataclass(frozen=True)
class ValidationCompleted(TrainingEvent):
    total_loss: float = float("nan")
    target_losses: dict[str, float] = field(default_factory=dict)
    evaluation_kind: str = "validation"


@dataclass(frozen=True)
class CheckpointSaving(TrainingEvent):
    path: Path | None = None


@dataclass(frozen=True)
class CheckpointSaved(TrainingEvent):
    path: Path | None = None


@dataclass(frozen=True)
class RunCompleted(TrainingEvent):
    completion_reason: str = "normal_completion"
