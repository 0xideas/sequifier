from __future__ import annotations

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
from sequifier.integration.factories import load_factory
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
from sequifier.typechecking import beartype


@runtime_checkable
class TrainingObserver(Protocol):
    integration_id: str

    @beartype
    def handle(self, event: TrainingEvent) -> None: ...

    @beartype
    def state_dict(self) -> dict[str, Any]: ...

    @beartype
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class TrainingController(Protocol):
    integration_id: str

    @beartype
    def on_gradients_unscaled(
        self, event: GradientsUnscaled
    ) -> TrainingDirective | None: ...

    @beartype
    def state_dict(self) -> dict[str, Any]: ...

    @beartype
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class _RegisteredIntegration:
    instance: Any
    factory: str | None
    state_policy: StatePolicy
    rank_policy: str

    def __post_init__(self):
        instance = self.instance
        try:
            self.handle = instance.handle
        except AttributeError:
            self.handle = None
        try:
            self.on_gradients_unscaled = instance.on_gradients_unscaled
        except AttributeError:
            self.on_gradients_unscaled = None
        try:
            self.capture_request = instance.capture_request
        except AttributeError:
            self.capture_request = None
        try:
            self.interventions = instance.interventions
        except AttributeError:
            self.interventions = None
        try:
            self.local_gradient_summary = instance.local_gradient_summary
        except AttributeError:
            self.local_gradient_summary = None
        try:
            self.state_dict = instance.state_dict
        except AttributeError:
            self.state_dict = None
        try:
            self.load_state_dict = instance.load_state_dict
        except AttributeError:
            self.load_state_dict = None


@beartype
def _load_factory(path: str) -> Any:
    try:
        return load_factory(path)
    except AttributeError as error:
        raise ImportError(f"Integration factory {path!r} does not exist.") from error


class IntegrationManager:
    @beartype
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
            try:
                actual_id = instance.integration_id
            except AttributeError:
                actual_id = None
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
            try:
                integration_id = item.instance.integration_id
            except AttributeError:
                integration_id = None
            if not isinstance(integration_id, str) or not integration_id:
                raise ValueError(
                    "Every integration must define a non-empty integration_id."
                )
            if integration_id in ids:
                raise ValueError(f"Duplicate integration_id: {integration_id!r}.")
            ids.add(integration_id)
            if callable(item.handle):
                self._observers.append(item)
            if callable(item.on_gradients_unscaled):
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
    @beartype
    def enabled(self) -> bool:
        return bool(self._observers or self._controller)

    @property
    @beartype
    def controller(self) -> Any | None:
        return None if self._controller is None else self._controller.instance

    @property
    @beartype
    def control_enabled(self) -> bool:
        return self._controller_configured

    @beartype
    def emit(self, event: TrainingEvent) -> None:
        for item in self._observers:
            handle = item.handle
            assert callable(handle)
            handle(event)

    @beartype
    def forward_trace(self, event: BatchPrepared) -> TraceContext | None:
        requests: list[CaptureRequest] = []
        interventions: list[InterventionBinding] = []
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            request_provider = item.capture_request
            if callable(request_provider):
                request = request_provider(event)
                if request is not None and not isinstance(request, CaptureRequest):
                    raise TypeError(
                        "capture_request must return CaptureRequest or None."
                    )
                if request is not None:
                    requests.append(request)
            intervention_provider = item.interventions
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

        @beartype
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
        trace_catalog: Any
        try:
            trace_catalog = event.access.model.trace_catalog
        except AttributeError:
            trace_catalog = ()
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

    @beartype
    def validate_execution(
        self, *, torch_compile: str, data_parallelism: str | None
    ) -> None:
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            try:
                requirements = item.instance.execution_requirements
            except AttributeError:
                requirements = ExecutionRequirements()
            if not isinstance(requirements, ExecutionRequirements):
                raise TypeError(
                    f"Integration {item.instance.integration_id!r} has invalid "
                    "execution_requirements."
                )
            needs_eager = (
                requirements.activation_tracing
                or requirements.interventions
                or requirements.higher_order_gradients
                or callable(item.capture_request)
                or callable(item.interventions)
            )
            if needs_eager and torch_compile != "none":
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requires eager "
                    "execution; set run.execution.compilation.torch_compile to 'none'."
                )
            if self.distributed and needs_eager and item.rank_policy == "rank_zero":
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requests "
                    "rank-zero-only tracing or interventions during distributed "
                    "training; execution-affecting integrations must run on all ranks."
                )
            if data_parallelism == "fsdp" and (
                requirements.interventions
                or requirements.higher_order_gradients
                or requirements.full_parameters
            ):
                raise ValueError(
                    f"Integration {item.instance.integration_id!r} requests an "
                    "execution mode unsupported during FSDP training."
                )

    @beartype
    def directive(self, event: GradientsUnscaled) -> TrainingDirective | None:
        if not self._controller_configured:
            return None
        directive: TrainingDirective | None = None
        controller_event = event
        if self._controller is not None:
            summary_provider = self._controller.local_gradient_summary
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
                    try:
                        reduction = self._controller.instance.gradient_reduction
                    except AttributeError:
                        reduction = "mean"
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
            decide = self._controller.on_gradients_unscaled
            assert callable(decide)
            decision = decide(controller_event)
            if decision is not None and not isinstance(decision, TrainingDirective):
                raise TypeError(
                    "TrainingController must return TrainingDirective or None."
                )
            directive = decision

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

    @beartype
    def state_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        items = [*self._observers]
        if self._controller is not None and self._controller not in items:
            items.append(self._controller)
        for item in items:
            integration_id = item.instance.integration_id
            state_dict = item.state_dict
            result[integration_id] = {
                "factory": item.factory,
                "state": state_dict() if callable(state_dict) else {},
            }
        return result

    @beartype
    def checkpoint_state_dict(self) -> dict[str, dict[str, Any]]:
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

    @beartype
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
            load_state_dict = item.load_state_dict
            if callable(load_state_dict):
                load_state_dict(state)
        unknown = set(saved).difference(configured_ids)
        _ = unknown
