from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from sequifier.model.embedding import parse_embedding_layer_name


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
        selectors = tuple(parse_embedding_layer_name(name) for name in self.layer_names)
        route = self.network.resolve_interface(self.interface_name)
        valid = metadata["attention_valid_mask"]
        hidden = route.ingest(features, metadata)
        hidden = hidden.masked_fill(~valid[:, :, None], 0.0)
        attention = self.network._build_attention_mask(valid, hidden.dtype)
        final, captures = self.network.backbone.forward_with_activations(
            hidden,
            attention,
            tuple(
                selector.index
                for selector in selectors
                if selector.source == "backbone_layer"
            ),
            any(selector.source == "backbone_final_norm" for selector in selectors),
        )
        decoder_input = None
        decoder_captures = {}
        for selector in selectors:
            if selector.source == "decoder_hidden_block":
                if decoder_input is None:
                    decoder_input = route.decoder_input(
                        final.masked_fill(~valid[:, :, None], 0.0)
                    )
                branch = route.decoder.branches[selector.branch]
                if selector.branch not in decoder_captures:
                    indices = tuple(
                        s.index
                        for s in selectors
                        if s.source == "decoder_hidden_block"
                        and s.branch == selector.branch
                    )
                    decoder_captures[selector.branch] = (
                        branch.project_hidden_with_activations(decoder_input, indices)
                    )
        activations = []
        for selector in selectors:
            if selector.source == "backbone_layer":
                value = captures[selector.index]
            elif selector.source == "backbone_final_norm":
                value = captures["final_norm"]
            else:
                value = decoder_captures[selector.branch][selector.index]
            activations.append(value[:, -route.prediction_length :])
        return torch.cat(activations, dim=-1).float().transpose(0, 1)


class EmbeddingModelExporter:
    def build(self, network: Any, interface_name: str, config: Any) -> EmbeddingNetwork:
        return EmbeddingNetwork(
            network, interface_name, tuple(config.embedding_layer_names)
        )
