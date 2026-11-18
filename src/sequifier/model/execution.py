"""Model execution facts, independent of dataset ownership and learning phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequifier.config.train_config import (
        ModelSpecModel,
        ResolvedModelInterface,
        ResolvedSequifierConfig,
        TrainingRuntimeSettings,
    )


@dataclass
class ModelExecution:
    model: ModelSpecModel
    interfaces: dict[str, ResolvedModelInterface]
    runtime_settings: TrainingRuntimeSettings
    device: str = "cpu"
    project_root: str = "."
    model_name: str = "loaded-model"
    seed: int = 0
    embedding_layer_names: list[str] = field(
        default_factory=lambda: ["backbone.final_norm"]
    )

    @classmethod
    def from_config(
        cls, config: ResolvedSequifierConfig | ModelExecution
    ) -> ModelExecution:
        if isinstance(config, cls):
            return config
        return cls(
            model=config.model,
            interfaces=config.interfaces,
            runtime_settings=config.runtime_settings,
            device=config.device,
            project_root=config.project_root,
            model_name=config.model_name,
            seed=config.seed,
            embedding_layer_names=config.embedding_layer_names,
        )

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
