"""Construction, dispatch, state, and coordination for integrations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.distributed as dist

from sequifier.integration.contexts import (
    BatchPrepared,
    GradientsUnscaled,
    TrainingEvent,
)
from sequifier.integration.controls import TrainingDirective
from sequifier.integration.specifications import (
    ExecutionRequirements,
    IntegrationSpec,
    StatePolicy,
)
from sequifier.model.tracing import (
    CaptureRequest,
    ForwardContext,
    InterventionBinding,
    TraceContext,
)


@runtime_checkable
class TrainingObserver(Protocol):
    integration_id: str

    def handle(self, event: TrainingEvent) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class TrainingController(Protocol):
    integration_id: str

    def on_gradients_unscaled(
        self, event: GradientsUnscaled
    ) -> TrainingDirective | None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class _RegisteredIntegration:
    instance: Any
    factory: str | None
    state_policy: StatePolicy
    rank_policy: str


def _load_factory(path: str) -> Any:
    module_name, attribute_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ImportError(f"Integration factory {path!r} does not exist.") from error


class IntegrationManager:
    """Manage trusted observers and the single optional training controller."""

    def __init__(
        self,
        *,
        specs: tuple[IntegrationSpec, ...] = (),
        instances: tuple[Any, ...] = (),
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
    ) -> None:
        if distributed and instances:
            raise ValueError(
                "Direct integration instances are only supported for "
                "non-distributed training; use IntegrationSpec for workers."
            )
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        registered: list[_RegisteredIntegration] = []

        for spec in specs:
            if spec.rank_policy == "rank_zero" and rank != 0:
                continue
            factory = _load_factory(spec.factory)
            instance = factory(dict(spec.config))
            actual_id = getattr(instance, "integration_id", None)
            if actual_id != spec.integration_id:
                raise ValueError(
                    f"Factory {spec.factory!r} returned integration_id "
                    f"{actual_id!r}, expected {spec.integration_id!r}."
                )
            registered.append(
                _RegisteredIntegration(
                    instance, spec.factory, spec.state_policy, spec.rank_policy
                )
            )
        registered.extend(
            _RegisteredIntegration(instance, None, "if_present", "all")
            for instance in instances
        )

        ids: set[str] = set()
        self._observers: list[_RegisteredIntegration] = []
        self._controller: _RegisteredIntegration | None = None
        for item in registered:
            integration_id = getattr(item.instance, "integration_id", None)
            if not isinstance(integration_id, str) or not integration_id:
                raise ValueError(
                    "Every integration must define a non-empty integration_id."
                )
            if integration_id in ids:
                raise ValueError(f"Duplicate integration_id: {integration_id!r}.")
            ids.add(integration_id)
            if callable(getattr(item.instance, "handle", None)):
                self._observers.append(item)
            if callable(getattr(item.instance, "on_gradients_unscaled", None)):
                if self._controller is not None:
                    raise ValueError(
                        "At most one TrainingController may be configured."
                    )
                self._controller = item
            if item not in self._observers and item is not self._controller:
                raise TypeError(
                    f"Integration {integration_id!r} is neither an observer "
                    "nor a controller."
                )
        self._controller_configured = self._controller is not None
        if self.distributed and specs:
            configured_by_rank: list[bool | None] = [None] * self.world_size
            dist.all_gather_object(configured_by_rank, self._controller is not None)
            self._controller_configured = any(
                configured is True for configured in configured_by_rank
            )

    @property
    def enabled(self) -> bool:
        return bool(self._observers or self._controller)

    @property
    def controller(self) -> Any | None:
        return None if self._controller is None else self._controller.instance

    @property
    def control_enabled(self) -> bool:
        return self._controller_configured

    def emit(self, event: TrainingEvent) -> None:
        for item in self._observers:
            item.instance.handle(event)

    def forward_trace(self, event: BatchPrepared) -> TraceContext | None:
        """Combine declared per-batch trace/intervention requests."""

        requests: list[CaptureRequest] = []
        interventions: list[InterventionBinding] = []
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            request_provider = getattr(item.instance, "capture_request", None)
            if callable(request_provider):
                request = request_provider(event)
                if request is not None and not isinstance(request, CaptureRequest):
                    raise TypeError(
                        "capture_request must return CaptureRequest or None."
                    )
                if request is not None:
                    requests.append(request)
            intervention_provider = getattr(item.instance, "interventions", None)
            if callable(intervention_provider):
                bindings = intervention_provider(event)
                if not isinstance(bindings, tuple) or not all(
                    isinstance(binding, InterventionBinding) for binding in bindings
                ):
                    raise TypeError(
                        "interventions must return tuple[InterventionBinding, ...]."
                    )
                interventions.extend(bindings)
        if not requests and not interventions:
            return None

        positions = requests[0].positions if requests else None

        def positions_equal(left: Any, right: Any) -> bool:
            if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
                return (
                    isinstance(left, torch.Tensor)
                    and isinstance(right, torch.Tensor)
                    and torch.equal(left, right)
                )
            return left == right

        if any(
            not positions_equal(request.positions, positions)
            for request in requests[1:]
        ):
            positions = None
        request = CaptureRequest(
            sites=tuple(
                dict.fromkeys(site for item in requests for site in item.sites)
            ),
            detach=all(item.detach for item in requests),
            retain_grad=any(item.retain_grad for item in requests),
            clone=any(item.clone for item in requests),
            positions=positions,
        )
        trace_catalog = getattr(event.access.model, "trace_catalog", ())
        available = {site.name for site in trace_catalog}
        requested = set(request.sites).union(binding.site for binding in interventions)
        unknown = requested.difference(available)
        if unknown:
            raise ValueError(f"Unknown training trace sites: {sorted(unknown)!r}.")
        return TraceContext(
            request,
            interventions=tuple(interventions),
            forward_context=ForwardContext(training=True, metadata=event.metadata),
        )

    def validate_execution(
        self, *, torch_compile: str, data_parallelism: str | None
    ) -> None:
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            requirements = getattr(
                item.instance, "execution_requirements", ExecutionRequirements()
            )
            if not isinstance(requirements, ExecutionRequirements):
                raise TypeError(
                    f"Integration {item.instance.integration_id!r} has invalid "
                    "execution_requirements."
                )
            needs_eager = (
                requirements.activation_tracing
                or requirements.interventions
                or requirements.higher_order_gradients
                or callable(getattr(item.instance, "capture_request", None))
                or callable(getattr(item.instance, "interventions", None))
            )
            if needs_eager and torch_compile != "none":
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requires eager "
                    "execution; set training_spec.torch_compile to 'none'."
                )
            if self.distributed and needs_eager and item.rank_policy == "rank_zero":
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requests "
                    "rank-zero-only tracing or interventions during distributed "
                    "training; execution-affecting integrations must run on all ranks."
                )
            if data_parallelism == "FSDP" and (
                requirements.interventions
                or requirements.higher_order_gradients
                or requirements.full_parameters
            ):
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requests an "
                    "execution mode unsupported during FSDP training."
                )

    def directive(self, event: GradientsUnscaled) -> TrainingDirective | None:
        if not self._controller_configured:
            return None
        directive: TrainingDirective | None = None
        controller_event = event
        if self._controller is not None:
            summary_provider = getattr(
                self._controller.instance, "local_gradient_summary", None
            )
            if callable(summary_provider):
                local_summary = summary_provider(event)
                if not isinstance(local_summary, dict) or not all(
                    isinstance(name, str) and isinstance(value, (int, float))
                    for name, value in local_summary.items()
                ):
                    raise TypeError(
                        "local_gradient_summary must return dict[str, float]."
                    )
                reduced_summary = {
                    name: float(value) for name, value in local_summary.items()
                }
                if self.distributed:
                    if self._controller.rank_policy != "all_reduce_summary":
                        raise ValueError(
                            "A distributed controller with local_gradient_summary "
                            "must use rank_policy='all_reduce_summary'."
                        )
                    parameter = next(event.access.model.parameters())
                    names = sorted(reduced_summary)
                    names_by_rank: list[list[str] | None] = [None] * self.world_size
                    dist.all_gather_object(names_by_rank, names)
                    if any(rank_names != names for rank_names in names_by_rank):
                        raise ValueError(
                            "Controller summary keys must match on every rank."
                        )
                    values = torch.tensor(
                        [reduced_summary[name] for name in names],
                        dtype=torch.float64,
                        device=parameter.device,
                    )
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    reduction = getattr(
                        self._controller.instance, "gradient_reduction", "mean"
                    )
                    if reduction == "mean":
                        values /= self.world_size
                    elif reduction != "sum":
                        raise ValueError(
                            "gradient_reduction must be either 'mean' or 'sum'."
                        )
                    reduced_summary = {
                        name: float(value)
                        for name, value in zip(names, values.cpu().tolist())
                    }
                controller_event = replace(event, reduced_summary=reduced_summary)
        if self._controller is not None and (not self.distributed or self.rank == 0):
            directive = self._controller.instance.on_gradients_unscaled(
                controller_event
            )
            if directive is not None and not isinstance(directive, TrainingDirective):
                raise TypeError(
                    "TrainingController must return TrainingDirective or None."
                )

        if self.distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "Distributed integration control requires an initialized "
                    "process group."
                )
            payload: list[TrainingDirective | None] = [directive]
            dist.broadcast_object_list(payload, src=0)
            directive = payload[0]
        return directive

    def state_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            integration_id = item.instance.integration_id
            state_dict = getattr(item.instance, "state_dict", None)
            result[integration_id] = {
                "factory": item.factory,
                "state": state_dict() if callable(state_dict) else {},
            }
        return result

    def checkpoint_state_dict(self) -> dict[str, dict[str, Any]]:
        """Collect per-rank state without assuming observer state is identical."""

        local_state = self.state_dict()
        if not self.distributed:
            return local_state
        states: list[dict[str, Any] | None] | None = (
            cast(list[dict[str, Any] | None], [None] * self.world_size)
            if self.rank == 0
            else None
        )
        dist.gather_object(local_state, object_gather_list=states, dst=0)
        if self.rank != 0:
            return {}
        if states is None:
            raise RuntimeError("Rank-zero integration state gather failed.")

        integration_ids = {
            integration_id
            for state in states
            if state is not None
            for integration_id in state
        }
        collected: dict[str, dict[str, Any]] = {}
        for integration_id in integration_ids:
            rank_entries = {
                str(rank): state[integration_id]
                for rank, state in enumerate(states)
                if state is not None and integration_id in state
            }
            first = next(iter(rank_entries.values()))
            collected[integration_id] = {
                "factory": first.get("factory"),
                "state_by_rank": {
                    rank: entry.get("state", {}) for rank, entry in rank_entries.items()
                },
            }
        return collected

    def load_state_dict(self, saved: dict[str, Any] | None) -> None:
        saved = saved or {}
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        configured_ids = {item.instance.integration_id for item in items}
        for item in items:
            integration_id = item.instance.integration_id
            entry = saved.get(integration_id)
            if entry is None:
                if item.state_policy == "required":
                    raise ValueError(
                        f"Checkpoint has no required state for {integration_id!r}."
                    )
                continue
            if item.state_policy == "fresh":
                continue
            if isinstance(entry, dict) and "state_by_rank" in entry:
                state_by_rank = entry["state_by_rank"]
                if not isinstance(state_by_rank, dict):
                    raise ValueError(
                        f"Checkpoint per-rank state for {integration_id!r} is invalid."
                    )
                state = state_by_rank.get(str(self.rank), state_by_rank.get("0"))
            else:
                state = entry.get("state") if isinstance(entry, dict) else None
            if not isinstance(state, dict):
                raise ValueError(
                    f"Checkpoint integration state for {integration_id!r} is invalid."
                )
            load_state_dict = getattr(item.instance, "load_state_dict", None)
            if callable(load_state_dict):
                load_state_dict(state)
        unknown = set(saved).difference(configured_ids)
        # Missing integrations are intentionally allowed: a resumed run may omit
        # an old optional recorder while retaining its state in the checkpoint.
        _ = unknown
