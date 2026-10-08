"""Route catalog parameters into a standard, checkpointable optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Iterable

import torch
from torch import nn

from sequifier.config.optimizer_config import OptimizerPlan, OptimizerSelector
from sequifier.model.parameter_catalog import ParameterCatalog, ParameterDescriptor
from sequifier.optimizers.optimizers import get_optimizer_class


@dataclass(frozen=True)
class RoutedOptimizerGroup:
    id: str
    optimizer_name: str
    parameters: tuple[nn.Parameter, ...]
    learning_rate: float
    arguments: dict[str, Any]


def _matches(descriptor: ParameterDescriptor, selector: OptimizerSelector) -> bool:
    return (
        (
            selector.semantic_groups is None
            or any(
                fnmatchcase(descriptor.semantic_group, pattern)
                for pattern in selector.semantic_groups
            )
        )
        and (selector.component is None or descriptor.component == selector.component)
        and (
            selector.parameter_kind is None
            or descriptor.parameter_kind == selector.parameter_kind
        )
        and (selector.ndim is None or len(descriptor.shape) == selector.ndim)
    )


def route_optimizer_parameters(
    plan: OptimizerPlan,
    catalog: ParameterCatalog,
    parameters: Iterable[nn.Parameter],
) -> tuple[RoutedOptimizerGroup, ...]:
    """Assign every supplied trainable parameter once, rejecting ambiguous routes."""

    supplied = {
        id(parameter): parameter for parameter in parameters if parameter.requires_grad
    }
    grouped: dict[str, list[nn.Parameter]] = {group.id: [] for group in plan.groups}
    seen: set[int] = set()
    explicit = plan.groups[:-1]
    fallback = plan.groups[-1]

    for descriptor in catalog.descriptors():
        parameter = catalog.parameter(descriptor.parameter_id)
        identity = id(parameter)
        if identity not in supplied or identity in seen:
            continue
        seen.add(identity)
        matches = [
            group.id
            for group in explicit
            if isinstance(group.select, OptimizerSelector)
            and _matches(descriptor, group.select)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Parameter {descriptor.canonical_name!r} matches multiple optimizer "
                f"groups: {matches!r}."
            )
        grouped[matches[0] if matches else fallback.id].append(parameter)

    if seen != set(supplied):
        raise ValueError(
            "The parameter catalog did not cover all trainable optimizer parameters."
        )
    empty = [group.id for group in plan.groups if not grouped[group.id]]
    if empty:
        raise ValueError(f"Optimizer groups have no trainable parameters: {empty!r}.")

    return tuple(
        RoutedOptimizerGroup(
            id=group.id,
            optimizer_name=group.optimizer.name,
            parameters=tuple(grouped[group.id]),
            learning_rate=group.learning_rate,
            arguments=dict(group.optimizer.arguments),
        )
        for group in plan.groups
    )


class CompositeOptimizer(torch.optim.Optimizer):
    """Expose several disjoint optimizers as one ordinary PyTorch optimizer.

    Child groups and state are shared with this facade. This lets schedulers,
    gradient scaling, integrations, and PyTorch state-dict helpers operate on
    one optimizer with the usual ``state`` and ``param_groups`` layout.
    """

    def __init__(self, groups: tuple[RoutedOptimizerGroup, ...]):
        if not groups:
            raise ValueError("A composite optimizer requires at least one group.")
        super().__init__([{"params": group.parameters} for group in groups], {})
        children: list[torch.optim.Optimizer] = []
        exposed_groups: list[dict[str, Any]] = []
        for group in groups:
            optimizer_class = get_optimizer_class(group.optimizer_name)
            if optimizer_class is torch.optim.LBFGS:
                raise ValueError(
                    "LBFGS requires closure-based steps and cannot be routed."
                )
            child = optimizer_class(
                [{"params": group.parameters, "group_id": group.id}],
                lr=group.learning_rate,
                **group.arguments,
            )
            self.state.update(child.state)
            child.state = self.state
            children.append(child)
            exposed_groups.append(child.param_groups[0])
        self._children = children
        self.param_groups = exposed_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for child in self._children:
            child.step()
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_ids = [
            group.get("group_id") for group in state_dict.get("param_groups", ())
        ]
        current_ids = [group["group_id"] for group in self.param_groups]
        if saved_ids != current_ids:
            raise ValueError(
                "Checkpoint optimizer groups do not match the current plan: "
                f"{saved_ids!r} != {current_ids!r}."
            )
        super().load_state_dict(state_dict)
        for child, group in zip(self._children, self.param_groups):
            child.param_groups = [group]
            child.state = self.state
