"""Reusable encoder stack with explicit capacity and stable layer ownership."""

from typing import Any

from torch import Tensor, nn

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.layers import RMSNorm, SequifierEncoderLayer
from sequifier.model.tracing import TraceContext


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        architecture: Any,
        max_context_length: int,
        *,
        defer_layers: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.dim_model = architecture.dim_model
        self.max_context_length = max_context_length
        if not defer_layers:
            self._construct_layers()

    def _construct_layers(self):
        architecture = self.architecture
        layers = [
            SequifierEncoderLayer(architecture, self.max_context_length)
            for _ in range(architecture.num_layers)
        ]
        for group in architecture.shared_layer_groups:
            shared_layer = layers[group[0]]
            for layer_index in group[1:]:
                layers[layer_index] = shared_layer
        self.layers = nn.ModuleList(layers)

        if architecture.normalization.norm_first:
            normalization_type = architecture.normalization.type
            norm_class = RMSNorm if normalization_type == "rmsnorm" else nn.LayerNorm
            norm_eps = 1e-6 if normalization_type == "rmsnorm" else 1e-3
            self.final_norm = norm_class(self.dim_model, eps=norm_eps)
        else:
            self.final_norm = nn.Identity()

    def _run_layers(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        layer_indices: tuple[int, ...] = (),
        capture_final_norm: bool = False,
        trace: TraceContext | None = None,
        site_prefix: str = "backbone",
        axes: tuple[str, ...] = ("batch", "time", "channel"),
    ):
        activations: dict[int | str, Tensor] = {}
        selected_indices = set(layer_indices)
        for index, layer in enumerate(self.layers):
            if trace is not None:
                x = trace.emit(
                    f"{site_prefix}.layer.{index}.input",
                    x,
                    axes=axes,
                    width=self.dim_model,
                )
            x = layer(
                x,
                src_mask=attention_mask,
                trace=trace,
                site_prefix=f"{site_prefix}.layer.{index}",
            )
            if trace is not None:
                x = trace.emit(
                    f"{site_prefix}.layer.{index}.output",
                    x,
                    axes=axes,
                    width=self.dim_model,
                )
            if index in selected_indices:
                activations[index] = x
        x = self.final_norm(cast_floating_to_module_dtype(x, self.final_norm))
        if trace is not None:
            x = trace.emit(
                f"{site_prefix}.final_norm",
                x,
                axes=axes,
                width=self.dim_model,
            )
        if capture_final_norm:
            activations["final_norm"] = x
        return x, activations

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self._run_layers(x, attention_mask)[0]
