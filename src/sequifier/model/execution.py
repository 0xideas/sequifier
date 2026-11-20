"""Model execution facts, independent of dataset ownership and learning phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sequifier.config.model_interface import (
    ModelSpecModel,
    ResolvedInterfaceStorage,
    ResolvedModelInterface,
)
from sequifier.model.execution_options import ModelExecutionOptions

if TYPE_CHECKING:
    from sequifier.config.train_config import ResolvedSequifierConfig


@dataclass
class ModelExecution:
    model: ModelSpecModel
    interfaces: dict[str, ResolvedModelInterface]
    options: ModelExecutionOptions
    device: str = "cpu"
    project_root: str = "."
    model_name: str = "loaded-model"
    seed: int = 0
    embedding_layer_names: list[str] = field(
        default_factory=lambda: ["backbone.final_norm"]
    )

    interface_storage: dict[str, ResolvedInterfaceStorage] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, config: ResolvedSequifierConfig | ModelExecution
    ) -> ModelExecution:
        return config if isinstance(config, ModelExecution) else config.execution

    @property
    def interface_names(self) -> tuple[str, ...]:
        return tuple(self.interfaces)

    def interface(self, name: str | None = None) -> ResolvedModelInterface:
        if name is None:
            if len(self.interfaces) != 1:
                raise ValueError("A model interface selection is required")
            name = next(iter(self.interfaces))
        return self.interfaces[name]

    def validate(self) -> ModelExecution:
        from sequifier.config.train_config import interface_build_view
        from sequifier.model.decoders import resolve_decoding_plan
        from sequifier.model.ingestion_compiler import resolve_ingestion_plan

        for interface in self.interfaces.values():
            view = interface_build_view(self, interface)
            resolve_ingestion_plan(view)
            resolve_decoding_plan(view)
        return self
