"""Model execution facts, independent of dataset ownership and learning phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sequifier.config.components import BackboneComponentConfig
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
    backbone: BackboneComponentConfig
    interfaces: dict[str, ResolvedModelInterface]
    options: ModelExecutionOptions
    embedding_layer_names: list[str] = field(
        default_factory=lambda: ["backbone.final_norm"]
    )

    interface_storage: dict[str, ResolvedInterfaceStorage] = field(default_factory=dict)

    def __post_init__(self):
        if not self.interfaces or any(
            name != route.name for name, route in self.interfaces.items()
        ):
            raise ValueError("Model execution requires interfaces keyed by their names")

    @property
    def model(self) -> ModelSpecModel:
        """Project the effective architecture for callers that consume a model spec."""
        return ModelSpecModel.model_validate(
            {"backbone": self.backbone, "interfaces": self.interfaces}
        )

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
        from sequifier.model.decoders import resolve_decoding_plan
        from sequifier.model.ingestion_compiler import resolve_ingestion_plan

        for interface in self.interfaces.values():
            resolve_ingestion_plan(
                interface,
                self.backbone.architecture,
                causal_attention=self.options.attention == "causal",
            )
            resolve_decoding_plan(interface)
        return self
