from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.tracing import (
    CaptureRequest,
    ForwardContext,
    InterventionBinding,
    TraceContext,
    TraceSite,
    active_trace_context,
    trace_sites,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: dict[str, Tensor]
    prediction_positions: slice | Tensor


@dataclass(frozen=True)
class DecodeRequest:
    target_columns: tuple[str, ...] | None = None
    positions: slice | Tensor | None = None
    apply_output_transform: bool = False
    apply_final_norm: bool = False


@dataclass(frozen=True)
class TracedModelOutput:
    output: ModelOutput
    captures: dict[str, Tensor] = field(default_factory=dict)


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        *,
        ingestion: nn.Module,
        ingestion_adapter: nn.Module,
        backbone: nn.Module,
        decoder: nn.Module,
        attention_mask_policy: Tensor,
        decoding_support: int,
        prediction_length: int,
        target_columns: tuple[str, ...],
        target_column_types: dict[str, str],
    ) -> None:
        super().__init__()
        if decoding_support <= 0:
            raise ValueError("decoding_support must be positive.")
        self.ingestion = ingestion
        self.ingestion_adapter = ingestion_adapter
        self.backbone = backbone
        self.decoder = decoder
        self.decoding_support = decoding_support
        self.prediction_length = prediction_length
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.register_buffer(
            "attention_mask_policy", attention_mask_policy, persistent=False
        )

    @property
    def dim_model(self) -> int:
        return int(self.backbone.dim_model)

    @property
    def context_length(self) -> int:
        return int(self.attention_mask_policy.shape[-1])

    @property
    def trace_catalog(self) -> tuple[TraceSite, ...]:
        branch_counts = {
            name: tuple(branch.hidden_dims)
            for name, branch in self.decoder.branches.items()
        }
        first_attention = self.backbone.layers[0].attn
        return trace_sites(
            num_layers=len(self.backbone.layers),
            model_width=self.dim_model,
            attention_width=int(first_attention.head_dim),
            decoder_input_width=self.decoding_support * self.dim_model,
            decoder_branches=branch_counts,
            target_branches=dict(self.decoder.target_to_branch),
        )

    def _build_attention_mask(self, valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        batch_size, context_length = valid_mask.shape
        if context_length != self.context_length:
            raise ValueError(
                f"valid_mask sequence length ({context_length}) must match "
                f"model sequence length ({self.context_length})."
            )
        base_mask = self.attention_mask_policy.to(
            device=valid_mask.device, dtype=dtype
        ).view(1, 1, context_length, context_length)
        padding_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_length,
            device=valid_mask.device,
            dtype=dtype,
        )
        padding_mask = padding_mask.masked_fill(
            (~valid_mask.bool())[:, None, None, :], torch.finfo(dtype).min
        )
        return base_mask + padding_mask

    def encode(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        *,
        trace: TraceContext | None = None,
    ) -> Tensor:
        valid_mask = metadata["attention_valid_mask"].bool()
        hidden = self.ingestion(features, metadata)
        hidden = self.ingestion_adapter(
            cast_floating_to_module_dtype(hidden, self.ingestion_adapter)
        )
        if hidden.ndim != 3:
            raise ValueError(
                "Ingestion must produce [batch, time, channel], got "
                f"{tuple(hidden.shape)}."
            )
        if valid_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Invalid attention_valid_mask shape {tuple(valid_mask.shape)} "
                f"for ingestion output {tuple(hidden.shape)}."
            )
        if trace is not None:
            hidden = trace.emit(
                "ingestion.output",
                hidden,
                axes=("batch", "time", "channel"),
                width=hidden.shape[-1],
            )
        hidden = hidden.masked_fill(~valid_mask[:, :, None], 0.0)
        attention_mask = self._build_attention_mask(valid_mask, hidden.dtype)
        hidden = self.backbone(hidden, attention_mask, trace=trace)
        return hidden.masked_fill(~valid_mask[:, :, None], 0.0)

    def decoder_input(self, representation: Tensor) -> Tensor:
        if representation.ndim != 3:
            raise ValueError(
                "Representation must have [batch, time, channel] layout, got "
                f"{tuple(representation.shape)}."
            )
        if representation.shape[-1] != self.dim_model:
            raise ValueError(
                f"Representation width must be {self.dim_model}, got "
                f"{representation.shape[-1]}."
            )
        if self.decoding_support == 1:
            return representation
        if self.decoding_support > representation.shape[1]:
            raise ValueError(
                f"decoding_support {self.decoding_support} exceeds sequence length "
                f"{representation.shape[1]}."
            )
        windows = representation.unfold(1, self.decoding_support, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows.reshape(
            representation.shape[0],
            representation.shape[1] - self.decoding_support + 1,
            self.decoding_support * representation.shape[2],
        )

    def decode_representation(
        self,
        representation: Tensor,
        request: DecodeRequest = DecodeRequest(),
        *,
        trace: TraceContext | None = None,
    ) -> dict[str, Tensor]:
        if request.apply_final_norm:
            representation = self.backbone.final_norm(
                cast_floating_to_module_dtype(representation, self.backbone.final_norm)
            )
        decoder_input = self.decoder_input(representation)
        if request.positions is not None:
            decoder_input = decoder_input[:, request.positions]
        if trace is not None:
            decoder_input = trace.emit(
                "decoder.input",
                decoder_input,
                axes=("batch", "time", "channel"),
                width=decoder_input.shape[-1],
            )
        decoded = self.decoder(decoder_input, trace=trace)
        targets = request.target_columns or self.target_columns
        unknown = set(targets).difference(self.target_columns)
        if unknown:
            raise ValueError(f"Unknown decoder target columns: {sorted(unknown)!r}.")
        outputs = {target: decoded[target] for target in targets}
        if request.apply_output_transform:
            outputs = {
                target: (
                    torch.log_softmax(output.float(), dim=-1)
                    if self.target_column_types[target] == "categorical"
                    else output
                )
                for target, output in outputs.items()
            }
        return outputs

    def forward(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        *,
        trace: TraceContext | None = None,
    ) -> ModelOutput:
        if trace is None:
            trace = active_trace_context()
        if trace is not None and not trace.forward_context.metadata:
            trace.forward_context.metadata.update(metadata)
        representation = self.encode(features, metadata, trace=trace)
        logits = self.decode_representation(representation, trace=trace)
        start = max(0, next(iter(logits.values())).shape[1] - self.prediction_length)
        return ModelOutput(logits=logits, prediction_positions=slice(start, None))

    def trace(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        request: CaptureRequest,
        *,
        interventions: tuple[InterventionBinding, ...] = (),
    ) -> TracedModelOutput:
        available = {site.name for site in self.trace_catalog}
        requested = set(request.sites).union(binding.site for binding in interventions)
        unknown = requested.difference(available)
        if unknown:
            raise ValueError(f"Unknown trace sites: {sorted(unknown)!r}.")
        context = TraceContext(
            request,
            interventions=interventions,
            forward_context=ForwardContext(training=self.training, metadata=metadata),
        )
        output = self(features, metadata, trace=context)
        return TracedModelOutput(output=output, captures=dict(context.captures))


class LegacyOutputAdapter(nn.Module):
    def __init__(self, network: TransformerNetwork, *, apply_output_transform: bool):
        super().__init__()
        self.network = network
        self.apply_output_transform = apply_output_transform

    def forward(
        self, features: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        output = self.network(features, metadata)
        logits = {
            target: tensor[:, output.prediction_positions].transpose(0, 1)
            for target, tensor in output.logits.items()
        }
        if not self.apply_output_transform:
            return logits
        return {
            target: (
                torch.log_softmax(value.float(), dim=-1)
                if self.network.target_column_types[target] == "categorical"
                else value
            )
            for target, value in logits.items()
        }


class ModelInterfaceModule(nn.Module):
    """One named ingestion/adapter/decoder route around the shared backbone."""

    def __init__(
        self,
        *,
        ingestion: nn.Module,
        ingestion_adapter: nn.Module,
        decoder: nn.Module,
        decoding_support: int,
        prediction_length: int,
        target_columns: tuple[str, ...],
        target_column_types: dict[str, str],
    ) -> None:
        super().__init__()
        if decoding_support <= 0:
            raise ValueError("decoding_support must be positive.")
        self.ingestion = ingestion
        self.ingestion_adapter = ingestion_adapter
        self.decoder = decoder
        self.decoding_support = decoding_support
        self.prediction_length = prediction_length
        self.target_columns = target_columns
        self.target_column_types = target_column_types

    def ingest(
        self, features: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        hidden = self.ingestion(features, metadata)
        return self.ingestion_adapter(
            cast_floating_to_module_dtype(hidden, self.ingestion_adapter)
        )

    def decoder_input(self, representation: Tensor) -> Tensor:
        if representation.ndim != 3:
            raise ValueError(
                "Representation must have [batch, time, channel] layout, got "
                f"{tuple(representation.shape)}."
            )
        if self.decoding_support == 1:
            return representation
        if self.decoding_support > representation.shape[1]:
            raise ValueError(
                f"decoding_support {self.decoding_support} exceeds sequence length "
                f"{representation.shape[1]}."
            )
        windows = representation.unfold(1, self.decoding_support, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows.reshape(
            representation.shape[0],
            representation.shape[1] - self.decoding_support + 1,
            self.decoding_support * representation.shape[2],
        )

    def decode(
        self,
        representation: Tensor,
        request: DecodeRequest = DecodeRequest(),
        *,
        trace: TraceContext | None = None,
    ) -> dict[str, Tensor]:
        decoder_input = self.decoder_input(representation)
        if request.positions is not None:
            decoder_input = decoder_input[:, request.positions]
        if trace is not None:
            decoder_input = trace.emit(
                "decoder.input",
                decoder_input,
                axes=("batch", "time", "channel"),
                width=decoder_input.shape[-1],
            )
        decoded = self.decoder(decoder_input, trace=trace)
        targets = request.target_columns or self.target_columns
        unknown = set(targets).difference(self.target_columns)
        if unknown:
            raise ValueError(f"Unknown decoder target columns: {sorted(unknown)!r}.")
        outputs = {target: decoded[target] for target in targets}
        if request.apply_output_transform:
            outputs = {
                target: (
                    torch.log_softmax(output.float(), dim=-1)
                    if self.target_column_types[target] == "categorical"
                    else output
                )
                for target, output in outputs.items()
            }
        return outputs


class ComposableTransformerNetwork(nn.Module):
    """A shared backbone with explicitly selected named model interfaces."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        interfaces: dict[str, ModelInterfaceModule],
        attention_mask_policy: Tensor,
    ) -> None:
        super().__init__()
        if not interfaces:
            raise ValueError("At least one model interface is required.")
        self.backbone = backbone
        self.interfaces = nn.ModuleDict(interfaces)
        self.register_buffer(
            "attention_mask_policy", attention_mask_policy, persistent=False
        )

    @property
    def dim_model(self) -> int:
        return int(self.backbone.dim_model)

    @property
    def context_length(self) -> int:
        return int(self.attention_mask_policy.shape[-1])

    def resolve_interface(self, interface_name: str | None) -> ModelInterfaceModule:
        if interface_name is None:
            if len(self.interfaces) != 1:
                raise ValueError(
                    "interface_name is required when the model has multiple interfaces"
                )
            interface_name = next(iter(self.interfaces))
        if interface_name not in self.interfaces:
            raise ValueError(f"Unknown model interface {interface_name!r}.")
        return self.interfaces[interface_name]

    def trace_catalog_for(
        self, interface_name: str | None = None
    ) -> tuple[TraceSite, ...]:
        route = self.resolve_interface(interface_name)
        branch_counts = {
            name: tuple(branch.hidden_dims)
            for name, branch in route.decoder.branches.items()
        }
        first_attention = self.backbone.layers[0].attn
        return trace_sites(
            num_layers=len(self.backbone.layers),
            model_width=self.dim_model,
            attention_width=int(first_attention.head_dim),
            decoder_input_width=route.decoding_support * self.dim_model,
            decoder_branches=branch_counts,
            target_branches=dict(route.decoder.target_to_branch),
        )

    @property
    def trace_catalog(self) -> tuple[TraceSite, ...]:
        return self.trace_catalog_for()

    def _build_attention_mask(self, valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        batch_size, context_length = valid_mask.shape
        if context_length != self.context_length:
            raise ValueError(
                f"valid_mask sequence length ({context_length}) must match "
                f"model sequence length ({self.context_length})."
            )
        base_mask = self.attention_mask_policy.to(
            device=valid_mask.device, dtype=dtype
        ).view(1, 1, context_length, context_length)
        padding_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_length,
            device=valid_mask.device,
            dtype=dtype,
        )
        return base_mask + padding_mask.masked_fill(
            (~valid_mask.bool())[:, None, None, :], torch.finfo(dtype).min
        )

    def encode(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        *,
        interface_name: str | None = None,
        trace: TraceContext | None = None,
    ) -> Tensor:
        route = self.resolve_interface(interface_name)
        valid_mask = metadata["attention_valid_mask"].bool()
        hidden = route.ingest(features, metadata)
        if hidden.ndim != 3:
            raise ValueError(
                "Ingestion must produce [batch, time, channel], got "
                f"{tuple(hidden.shape)}."
            )
        if valid_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Invalid attention_valid_mask shape {tuple(valid_mask.shape)} "
                f"for ingestion output {tuple(hidden.shape)}."
            )
        if trace is not None:
            hidden = trace.emit(
                "ingestion.output",
                hidden,
                axes=("batch", "time", "channel"),
                width=hidden.shape[-1],
            )
        hidden = hidden.masked_fill(~valid_mask[:, :, None], 0.0)
        attention_mask = self._build_attention_mask(valid_mask, hidden.dtype)
        hidden = self.backbone(hidden, attention_mask, trace=trace)
        return hidden.masked_fill(~valid_mask[:, :, None], 0.0)

    def decode_representation(
        self,
        representation: Tensor,
        request: DecodeRequest = DecodeRequest(),
        *,
        interface_name: str | None = None,
        trace: TraceContext | None = None,
    ) -> dict[str, Tensor]:
        route = self.resolve_interface(interface_name)
        if representation.shape[-1] != self.dim_model:
            raise ValueError(
                f"Representation width must be {self.dim_model}, got "
                f"{representation.shape[-1]}."
            )
        if request.apply_final_norm:
            representation = self.backbone.final_norm(
                cast_floating_to_module_dtype(representation, self.backbone.final_norm)
            )
        return route.decode(representation, request, trace=trace)

    def forward(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        *,
        interface_name: str | None = None,
        trace: TraceContext | None = None,
    ) -> ModelOutput:
        if trace is None:
            trace = active_trace_context()
        if trace is not None and not trace.forward_context.metadata:
            trace.forward_context.metadata.update(metadata)
        route = self.resolve_interface(interface_name)
        representation = self.encode(
            features, metadata, interface_name=interface_name, trace=trace
        )
        logits = route.decode(representation, trace=trace)
        start = max(0, next(iter(logits.values())).shape[1] - route.prediction_length)
        return ModelOutput(logits=logits, prediction_positions=slice(start, None))

    def trace(
        self,
        features: dict[str, Tensor],
        metadata: dict[str, Tensor],
        request: CaptureRequest,
        *,
        interface_name: str | None = None,
        interventions: tuple[InterventionBinding, ...] = (),
    ) -> TracedModelOutput:
        available = {site.name for site in self.trace_catalog_for(interface_name)}
        requested = set(request.sites).union(binding.site for binding in interventions)
        unknown = requested.difference(available)
        if unknown:
            raise ValueError(f"Unknown trace sites: {sorted(unknown)!r}.")
        context = TraceContext(
            request,
            interventions=interventions,
            forward_context=ForwardContext(training=self.training, metadata=metadata),
        )
        output = self(features, metadata, interface_name=interface_name, trace=context)
        return TracedModelOutput(output=output, captures=dict(context.captures))
