from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sequifier.model.embedding import embedding_layer_trace_site
from sequifier.model.tracing import CaptureRequest


class EmbeddingNetwork(nn.Module):
    """Stateless view that concatenates explicitly selected network activations."""

    def __init__(
        self, network: Any, interface_name: str, layer_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.network = network
        self.interface_name = interface_name
        self.layer_names = layer_names

    def forward(
        self, features: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        sites = tuple(embedding_layer_trace_site(name) for name in self.layer_names)
        traced = self.network.trace(
            features,
            metadata,
            CaptureRequest(sites=sites),
            interface_name=self.interface_name,
        )
        route = self.network.resolve_interface(self.interface_name)
        activations = [traced.captures[site] for site in sites]
        normalized = []
        for activation in activations:
            if activation.ndim != 3:
                raise ValueError(
                    "Embedding trace sites must have batch/time/channel axes."
                )
            normalized.append(activation[:, -route.prediction_length :])
        return torch.cat(normalized, dim=-1).transpose(0, 1)


class EmbeddingModelExporter:
    def build(self, network: Any, interface_name: str, config: Any) -> EmbeddingNetwork:
        return EmbeddingNetwork(
            network, interface_name, tuple(config.embedding_layer_names)
        )
