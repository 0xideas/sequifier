import math
from typing import Any

import torch
from torch import Tensor, nn

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.encoder_stack import TransformerEncoderStack
from sequifier.model.tracing import TraceContext
from sequifier.typechecking import beartype, conditional_beartype


class TransformerBackbone(TransformerEncoderStack):
    """Dataset-independent temporal transformer.

    The input and output contract is batch-first
    ``[batch, time, input_dim]``. ``input_dim`` normally equals ``dim_model``;
    ``range_concat`` reserves the final model channel for its coordinate.
    Attention policy and padding are supplied by the caller as one
    broadcastable attention mask.
    """

    @beartype
    def __init__(self, architecture: Any):
        super().__init__(
            architecture, architecture.max_context_length, defer_layers=True
        )
        self.architecture = architecture
        self.dim_model = architecture.dim_model
        self.max_context_length = architecture.max_context_length
        self.position_encoding_type = architecture.position_encoding.type
        self.positional_encoding_scope = architecture.positional_encoding_scope
        self.input_dim = self.dim_model - int(
            self.position_encoding_type == "range_concat"
        )
        self.position_dropout = nn.Dropout(architecture.dropout)

        self.position_embedding: nn.Module | None = None
        self.range_projection: nn.Module | None = None
        if (
            self.position_encoding_type == "learned"
            and self.positional_encoding_scope == "global"
        ):
            self.position_embedding = nn.Embedding(
                self.max_context_length, self.dim_model
            )
        elif self.position_encoding_type == "range":
            self.range_projection = nn.Linear(self.dim_model + 1, self.dim_model)

        sinusoidal = self._sinusoidal_positions(
            self.max_context_length,
            self.dim_model,
            architecture.position_encoding.theta,
        )
        self.register_buffer("sinusoidal_positions", sinusoidal, persistent=False)

        self._construct_layers()

    @staticmethod
    @conditional_beartype
    def _sinusoidal_positions(length: int, dim: int, theta: float) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(theta) / dim)
        )
        values = torch.zeros(length, dim, dtype=torch.float32)
        values[:, 0::2] = torch.sin(positions * frequencies)
        if dim > 1:
            values[:, 1::2] = torch.cos(
                positions * frequencies[: values[:, 1::2].shape[1]]
            )
        return values

    @conditional_beartype
    def _add_temporal_position(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[1]
        if (
            not torch.onnx.is_in_onnx_export()
            and sequence_length > self.max_context_length
        ):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds backbone "
                f"max_context_length {self.max_context_length}."
            )

        if (
            self.position_encoding_type == "learned"
            and self.positional_encoding_scope == "global"
        ):
            positions = torch.arange(sequence_length, device=x.device)
            if self.position_embedding is None:
                raise RuntimeError("Learned position embedding was not initialized.")
            position_values = self.position_embedding(positions).to(dtype=x.dtype)
            return self.position_dropout(x + position_values.unsqueeze(0))

        if self.position_encoding_type in {"range", "range_concat"}:
            if sequence_length == 1:
                positions = torch.zeros(1, device=x.device, dtype=x.dtype)
            else:
                position_dtype = (
                    torch.float32 if torch.onnx.is_in_onnx_export() else x.dtype
                )
                positions = torch.linspace(
                    -1.0,
                    1.0,
                    sequence_length,
                    device=x.device,
                    dtype=position_dtype,
                )
                positions = positions.to(dtype=x.dtype)
            positions = positions.view(1, sequence_length, 1).expand(x.shape[0], -1, -1)
            positioned = torch.cat((x, positions), dim=-1)
            if self.position_encoding_type == "range_concat":
                return positioned
            if self.range_projection is None:
                raise RuntimeError("Range position projection was not initialized.")
            positioned = self.range_projection(
                cast_floating_to_module_dtype(positioned, self.range_projection)
            )
            return self.position_dropout(positioned)

        if self.position_encoding_type == "sinusoidal":
            positions = self.sinusoidal_positions[:sequence_length].to(
                device=x.device, dtype=x.dtype
            )
            return self.position_dropout(x + positions.unsqueeze(0))

        return x

    @conditional_beartype
    def _forward_with_activations(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        layer_indices: tuple[int, ...],
        capture_final_norm: bool,
        trace: TraceContext | None = None,
    ) -> tuple[Tensor, dict[int | str, Tensor]]:
        if x.ndim != 3:
            raise ValueError(
                "TransformerBackbone expects [batch, time, dim_model], got "
                f"shape {tuple(x.shape)}."
            )
        if not torch.onnx.is_in_onnx_export() and x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Backbone input width must be {self.input_dim}, got {x.shape[-1]}."
            )

        x = self._add_temporal_position(x)
        if trace is not None:
            x = trace.emit(
                "backbone.positioned",
                x,
                axes=("batch", "time", "channel"),
                width=self.dim_model,
            )
        return self._run_layers(
            x, attention_mask, layer_indices, capture_final_norm, trace
        )

    @conditional_beartype
    def forward_with_activations(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        layer_indices: tuple[int, ...],
        capture_final_norm: bool,
        trace: TraceContext | None = None,
    ) -> tuple[Tensor, dict[int | str, Tensor]]:
        """Return the final state and selected batch-first layer activations."""
        return self._forward_with_activations(
            x,
            attention_mask,
            layer_indices,
            capture_final_norm,
            trace,
        )

    @conditional_beartype
    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        *,
        trace: TraceContext | None = None,
    ) -> Tensor:
        output, _ = self._forward_with_activations(x, attention_mask, (), False, trace)
        return output
