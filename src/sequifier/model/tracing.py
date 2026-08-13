"""Explicit logical-site tracing and differentiable interventions."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class TraceSite:
    name: str
    axes: tuple[str, ...]
    width: int | None = None


@dataclass(frozen=True)
class CaptureRequest:
    sites: tuple[str, ...]
    detach: bool = True
    retain_grad: bool = False
    clone: bool = False
    positions: slice | Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.sites) != len(set(self.sites)):
            raise ValueError("CaptureRequest sites must be unique.")
        if self.detach and self.retain_grad:
            raise ValueError("retain_grad requires detach=False.")


@dataclass(frozen=True)
class ForwardContext:
    training: bool
    metadata: dict[str, Tensor] = field(default_factory=dict)
    create_graph: bool = False
    retain_graph: bool = False


class Intervention(Protocol):
    def transform(
        self, site: TraceSite, tensor: Tensor, context: ForwardContext
    ) -> Tensor: ...


@dataclass(frozen=True)
class InterventionBinding:
    site: str
    intervention: Intervention


class TraceContext:
    """A per-forward collector; create a new instance for each execution."""

    def __init__(
        self,
        request: CaptureRequest | None = None,
        *,
        interventions: tuple[InterventionBinding, ...] = (),
        forward_context: ForwardContext | None = None,
    ) -> None:
        self.request = request
        self.interventions = interventions
        self.forward_context = forward_context or ForwardContext(training=False)
        self.captures: dict[str, Tensor] = {}
        self._capture_sites = set(request.sites if request is not None else ())
        self._interventions_by_site: dict[str, list[Intervention]] = {}
        for binding in interventions:
            self._interventions_by_site.setdefault(binding.site, []).append(
                binding.intervention
            )

    def requires(self, site_name: str) -> bool:
        return (
            site_name in self._capture_sites or site_name in self._interventions_by_site
        )

    @property
    def enabled(self) -> bool:
        return bool(self._capture_sites or self._interventions_by_site)

    def emit(
        self,
        name: str,
        tensor: Tensor,
        *,
        axes: tuple[str, ...],
        width: int | None = None,
    ) -> Tensor:
        site = TraceSite(name=name, axes=axes, width=width)
        transformed = tensor
        for intervention in self._interventions_by_site.get(name, ()):
            candidate = intervention.transform(site, transformed, self.forward_context)
            if not isinstance(candidate, Tensor):
                raise TypeError(f"Intervention at {name!r} must return a Tensor.")
            if candidate.shape != transformed.shape:
                raise ValueError(
                    f"Intervention at {name!r} changed shape from "
                    f"{tuple(transformed.shape)} to {tuple(candidate.shape)}."
                )
            transformed = candidate

        if name in self._capture_sites:
            captured = transformed
            request = self.request
            if request is None:
                raise RuntimeError("Trace capture requested without a CaptureRequest.")
            if request.positions is not None and "time" in axes:
                time_axis = axes.index("time")
                selection: list[Any] = [slice(None)] * captured.ndim
                selection[time_axis] = request.positions
                captured = captured[tuple(selection)]
            if request.retain_grad and captured.requires_grad:
                captured.retain_grad()
            if request.detach:
                captured = captured.detach()
            if request.clone:
                captured = captured.clone()
            self.captures[name] = captured
        return transformed


_ACTIVE_TRACE: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "sequifier_active_trace", default=None
)


def active_trace_context() -> TraceContext | None:
    return _ACTIVE_TRACE.get()


@contextlib.contextmanager
def activate_trace_context(context: TraceContext | None):
    """Activate a preconstructed trace context for one eager forward."""

    if context is None:
        yield None
        return
    token = _ACTIVE_TRACE.set(context)
    try:
        yield context
    finally:
        _ACTIVE_TRACE.reset(token)


@contextlib.contextmanager
def analysis_execution(
    model: Any,
    *,
    create_graph: bool = False,
    retain_graph: bool = False,
    trace: CaptureRequest | None = None,
    interventions: tuple[InterventionBinding, ...] = (),
):
    """Enable eager, graph-preserving execution for higher-order analysis.

    The returned :class:`TraceContext` owns captures. ``create_graph`` and
    ``retain_graph`` are recorded for the caller; pass the same flags to the
    eventual autograd operation because PyTorch applies them during backward,
    not during the forward pass.
    """

    if hasattr(model, "_orig_mod"):
        raise ValueError("analysis_execution requires an eager, uncompiled model.")
    available = {site.name for site in getattr(model, "trace_catalog", ())}
    requested = set(trace.sites if trace is not None else ()).union(
        binding.site for binding in interventions
    )
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(f"Unknown analysis trace sites: {sorted(unknown)!r}.")
    context = TraceContext(
        trace,
        interventions=interventions,
        forward_context=ForwardContext(
            training=bool(model.training),
            create_graph=create_graph,
            retain_graph=retain_graph,
        ),
    )
    with torch.enable_grad():
        with activate_trace_context(context):
            yield context


def functional_state(model: Any) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return explicit parameter and buffer mappings for ``torch.func`` calls."""

    parameters = dict(model.named_parameters(remove_duplicate=True))
    buffers = dict(model.named_buffers(remove_duplicate=True))
    return parameters, buffers


def functional_forward(
    model: Any,
    parameters: dict[str, Tensor],
    buffers: dict[str, Tensor],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute the eager network with caller-supplied functional state."""

    if hasattr(model, "_orig_mod"):
        raise ValueError("functional_forward requires an eager, uncompiled model.")
    return torch.func.functional_call(model, (parameters, buffers), args, kwargs)


def trace_sites(
    *,
    num_layers: int,
    model_width: int,
    attention_width: int,
    decoder_input_width: int,
    decoder_branches: dict[str, tuple[int, ...]],
    target_branches: dict[str, str],
) -> tuple[TraceSite, ...]:
    """Return the logical trace catalog for one concrete network."""

    batch_time_channel = ("batch", "time", "channel")
    sites = [
        TraceSite("ingestion.output", batch_time_channel),
        TraceSite("backbone.positioned", batch_time_channel, model_width),
    ]
    for index in range(num_layers):
        prefix = f"backbone.layer.{index}"
        sites.extend(
            [
                TraceSite(f"{prefix}.input", batch_time_channel, model_width),
                TraceSite(
                    f"{prefix}.attention.norm_input",
                    batch_time_channel,
                    model_width,
                ),
                TraceSite(
                    f"{prefix}.attention.q",
                    ("batch", "head", "time", "channel"),
                    attention_width,
                ),
                TraceSite(
                    f"{prefix}.attention.k",
                    ("batch", "head", "time", "channel"),
                    attention_width,
                ),
                TraceSite(
                    f"{prefix}.attention.v",
                    ("batch", "head", "time", "channel"),
                    attention_width,
                ),
                TraceSite(
                    f"{prefix}.attention.scores",
                    ("batch", "head", "time", "key_time"),
                ),
                TraceSite(
                    f"{prefix}.attention.weights",
                    ("batch", "head", "time", "key_time"),
                ),
                TraceSite(
                    f"{prefix}.attention.update", batch_time_channel, model_width
                ),
                TraceSite(
                    f"{prefix}.attention.output", batch_time_channel, model_width
                ),
                TraceSite(f"{prefix}.mlp.norm_input", batch_time_channel, model_width),
                TraceSite(f"{prefix}.mlp.pre_activation", batch_time_channel),
                TraceSite(f"{prefix}.mlp.activation", batch_time_channel),
                TraceSite(f"{prefix}.mlp.update", batch_time_channel, model_width),
                TraceSite(f"{prefix}.output", batch_time_channel, model_width),
            ]
        )
    sites.extend(
        [
            TraceSite("backbone.final_norm", batch_time_channel, model_width),
            TraceSite("decoder.input", batch_time_channel, decoder_input_width),
        ]
    )
    for branch, block_widths in decoder_branches.items():
        sites.extend(
            TraceSite(
                f"decoder.branch.{branch}.block.{index}",
                batch_time_channel,
                width,
            )
            for index, width in enumerate(block_widths)
        )
    sites.extend(
        TraceSite(f"decoder.branch.{branch}.logits.{target}", batch_time_channel)
        for target, branch in target_branches.items()
    )
    return tuple(sites)
