from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from sequifier.integration.callbacks import IntegrationManager
from sequifier.training.engine import TrainingEngine
from sequifier.typechecking import beartype


@dataclass
class TrainingSession:
    config: Any
    model: nn.Module
    engine: TrainingEngine
    train_loader: DataLoader
    validation_loader: DataLoader
    integrations: IntegrationManager

    @beartype
    def restore_integration_state(self, checkpoint: dict[str, Any] | None) -> None:
        if checkpoint is not None:
            self.integrations.load_state_dict(checkpoint.get("integration_state"))

    @beartype
    def run(self, *, ddp_model: nn.Module | None = None) -> None:
        self.engine.run(self.train_loader, self.validation_loader, ddp_model)
