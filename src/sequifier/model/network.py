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
from sequifier.typechecking import beartype, conditional_beartype


@dataclass(frozen=True)
class ModelOutput:
    logits: dict[str, Tensor]
    prediction_positions: slice | Tensor


@dataclass(frozen=True)
class EncodeRequest:
    interface_name: str | None = None


@dataclass(frozen=True)
class EncodedOutput:
    representation: Tensor


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


class ModelInterfaceModule(nn.Module):
    """One named ingestion/adapter/decoder route around the shared backbone."""

    @beartype
    def __init__(
        self,
        *,
        ingestion: nn.Module,
        ingestion_adapter: nn.Module,
        decoder: nn.Module,
        decoder_input_width: int,
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
        self.decoder_input_width = decoder_input_width
        self.decoding_support = decoding_support
        self.prediction_length = prediction_length
        self.target_columns = target_columns
        self.target_column_types = target_column_types

    @conditional_beartype
    def ingest(
        self, features: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        hidden = self.ingestion(features, metadata)
        return self.ingestion_adapter(
            cast_floating_to_module_dtype(hidden, self.ingestion_adapter)
        )

    @conditional_beartype
    def decoder_input(self, representation: Tensor) -> Tensor:
        if representation.ndim != 3:
            raise ValueError(
                "Representation must have [batch, time, channel] layout, got "
                f"{tuple(representation.shape)}."
            )
        if self.decoding_support == 1:
            return representation
        if (
            not torch.onnx.is_in_onnx_export()
            and self.decoding_support > representation.shape[1]
        ):
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

    @conditional_beartype
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
                width=self.decoder_input_width,
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

    @beartype
    def __init__(
        self,
        *,
        backbone: nn.Module,
        interfaces: dict[str, ModelInterfaceModule],
        attention_mask_policy: Tensor,
        context_length: int,
    ) -> None:
        super().__init__()
        if not interfaces:
            raise ValueError("At least one model interface is required.")
        self.backbone = backbone
        self.interfaces = nn.ModuleDict(interfaces)
        self._context_length = context_length
        self.register_buffer(
            "attention_mask_policy", attention_mask_policy, persistent=False
        )

    @property
    @conditional_beartype
    def dim_model(self) -> int:
        return int(self.backbone.dim_model)

    @property
    @conditional_beartype
    def context_length(self) -> int:
        return self._context_length

    @conditional_beartype
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

    @conditional_beartype
    def regularization_loss(self, interface_name: str | None = None) -> Tensor:
        """Return decoder regularization through an explicit model contract."""

        route = self.resolve_interface(interface_name)
        regularization = getattr(route.decoder, "regularization_loss", None)
        if not callable(regularization):
            parameter = next(route.decoder.parameters(), None)
            device = parameter.device if parameter is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return regularization()

    def structural_metadata(self) -> dict[str, object]:
        return {
            "state_dict_prefixes": ("backbone.", "interfaces."),
            "interfaces": tuple(self.interfaces),
            "context_length": self.context_length,
            "dim_model": self.dim_model,
        }

    @conditional_beartype
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
            decoder_input_width=route.decoder_input_width,
            decoder_branches=branch_counts,
            target_branches=dict(route.decoder.target_to_branch),
        )

    @property
    @conditional_beartype
    def trace_catalog(self) -> tuple[TraceSite, ...]:
        catalogs = [self.trace_catalog_for(name) for name in self.interfaces]
        by_name: dict[str, TraceSite] = {}
        for catalog in catalogs:
            for site in catalog:
                previous = by_name.get(site.name)
                if previous is not None and previous != site:
                    raise ValueError(
                        f"Trace site {site.name!r} has incompatible interface shapes."
                    )
                by_name[site.name] = site
        return tuple(by_name.values())

    @conditional_beartype
    def _build_attention_mask(self, valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        batch_size, context_length = valid_mask.shape
        if not torch.onnx.is_in_onnx_export() and context_length != self.context_length:
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

    @conditional_beartype
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
        if not torch.onnx.is_in_onnx_export() and valid_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"Invalid attention_valid_mask shape {tuple(valid_mask.shape)} "
                f"for ingestion output {tuple(hidden.shape)}."
            )
        if trace is not None:
            hidden = trace.emit(
                "ingestion.output",
                hidden,
                axes=("batch", "time", "channel"),
                width=self.backbone.input_dim,
            )
        hidden = hidden.masked_fill(~valid_mask[:, :, None], 0.0)
        attention_mask = self._build_attention_mask(valid_mask, hidden.dtype)
        hidden = self.backbone(hidden, attention_mask, trace=trace)
        return hidden.masked_fill(~valid_mask[:, :, None], 0.0)

    @conditional_beartype
    def decode_representation(
        self,
        representation: Tensor,
        request: DecodeRequest = DecodeRequest(),
        *,
        interface_name: str | None = None,
        trace: TraceContext | None = None,
    ) -> dict[str, Tensor]:
        route = self.resolve_interface(interface_name)
        if (
            not torch.onnx.is_in_onnx_export()
            and representation.shape[-1] != self.dim_model
        ):
            raise ValueError(
                f"Representation width must be {self.dim_model}, got "
                f"{representation.shape[-1]}."
            )
        if request.apply_final_norm:
            representation = self.backbone.final_norm(
                cast_floating_to_module_dtype(representation, self.backbone.final_norm)
            )
        return route.decode(representation, request, trace=trace)

    @conditional_beartype
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
        return ModelOutput(
            logits=logits,
            prediction_positions=slice(-route.prediction_length, None),
        )

    @conditional_beartype
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
