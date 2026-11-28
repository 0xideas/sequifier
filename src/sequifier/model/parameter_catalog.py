from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Set
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from sequifier.artifacts.state_dict import canonical_parameter_name
from sequifier.model.parameter_groups import semantic_batches
from sequifier.typechecking import beartype

if TYPE_CHECKING:
    from sequifier.config.parameter_selection import ParameterSelector

ParameterComponent = Literal["ingestion", "backbone", "decoder"]
ParameterKind = Literal["weight", "bias", "other"]


def _unwrapped(name: str) -> str:
    parts = name.split(".")
    while len(parts) > 1 and (parts[0] == "network" or parts[0].isdigit()):
        parts.pop(0)
    return ".".join(parts)


def _module_paths(alias: str) -> frozenset[str]:
    """Ancestor module paths of a single alias, with and without role wrappers."""
    paths = set()
    for name in (alias, _unwrapped(alias)):
        while "." in name:
            name = name.rsplit(".", 1)[0]
            paths.add(name)
    return frozenset(paths)


@dataclass(frozen=True)
class ParameterDescriptor:
    parameter_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    component: ParameterComponent
    semantic_group: str
    parameter_kind: ParameterKind
    shape: tuple[int, ...]
    dtype: torch.dtype
    shared_parameter_id: str | None
    depth_parameter: bool = False
    branch_path: tuple[str, ...] = ()
    semantic_aliases: tuple[str, ...] = ()

    @cached_property
    def module_paths(self) -> tuple[frozenset[str], ...]:
        return tuple(_module_paths(alias) for alias in self.aliases)

    def in_scope(self, selector: ParameterSelector) -> bool:
        if selector.exclude_modules:
            excluded = [
                not paths.isdisjoint(selector.exclude_modules)
                for paths in self.module_paths
            ]
            if any(excluded):
                if not all(excluded):
                    raise ValueError(
                        f"conflicting selector exclusions for shared parameter {self.parameter_id}"
                    )
                return False
        return (
            (selector.component is None or selector.component == self.component)
            and (
                selector.parameter_kind is None
                or selector.parameter_kind == self.parameter_kind
            )
            and (selector.ndim is None or selector.ndim == len(self.shape))
            and (
                not selector.branch_path
                or tuple(selector.branch_path)
                == self.branch_path[: len(selector.branch_path)]
            )
            and (
                not selector.module
                or any(selector.module in paths for paths in self.module_paths)
            )
            and (
                not selector.interface
                or any(
                    f"interfaces.{selector.interface}" in paths
                    for paths in self.module_paths
                )
            )
        )

    def matching_patterns(self, patterns: Iterable[str]) -> set[str]:
        return {
            pattern
            for pattern in patterns
            if any(fnmatchcase(group, pattern) for group in self.semantic_aliases)
        }


class ParameterCatalog:
    @beartype
    def __init__(self, model: nn.Module):
        self.model = model
        from sequifier.model.depth_ingestion import DepthTransformerIngestion

        roles = {}
        for batch in semantic_batches(model):
            for parameter in batch.parameters:
                if parameter is not None:
                    roles.setdefault(id(parameter), {}).setdefault(
                        batch.group, batch.kind
                    )

        depth_parameters = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, DepthTransformerIngestion)
            for parameter in module.parameters()
        }
        aliases_by_identity: dict[int, list[str]] = {}
        parameter_by_identity: dict[int, nn.Parameter] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            identity = id(parameter)
            parameter_by_identity[identity] = parameter
            canonical = canonical_parameter_name(name)
            if canonical not in aliases_by_identity.setdefault(identity, []):
                aliases_by_identity[identity].append(canonical)

        descriptors: list[ParameterDescriptor] = []
        parameters: dict[str, nn.Parameter] = {}
        aliases: dict[str, str] = {}
        for identity, names in aliases_by_identity.items():
            parameter = parameter_by_identity[identity]
            canonical_name = names[0]
            component = self._component(canonical_name)
            parameter_id = canonical_name
            groups = roles[identity]
            primary = next(iter(groups))
            descriptor = ParameterDescriptor(
                parameter_id=parameter_id,
                canonical_name=canonical_name,
                aliases=tuple(names),
                component=component,
                semantic_group=primary,
                semantic_aliases=tuple(sorted(groups)),
                parameter_kind=self._kind(canonical_name)
                if primary == "free_parameter"
                else groups[primary],
                # A learned scale has scalar semantics in saved selectors.
                shape=()
                if ".prediction_scales.log_scales." in canonical_name
                else tuple(parameter.shape),
                dtype=parameter.dtype,
                shared_parameter_id=parameter_id if len(names) > 1 else None,
                depth_parameter=bool(identity in depth_parameters),
                branch_path=tuple(
                    part
                    for i, part in enumerate(canonical_name.split("."))
                    if i and canonical_name.split(".")[i - 1] == "branches"
                ),
            )
            descriptors.append(descriptor)
            parameters[parameter_id] = parameter
            aliases.update({name: parameter_id for name in names})

        self._descriptors = tuple(descriptors)
        self._parameters = parameters
        self._aliases = aliases
        self._ids = {id(parameter): key for key, parameter in parameters.items()}
        module_names = (
            canonical_parameter_name(name)
            for name, _ in model.named_modules(remove_duplicate=False)
        )
        self._modules = {
            spelling for name in module_names for spelling in (name, _unwrapped(name))
        }

    @staticmethod
    @beartype
    def _component(name: str) -> ParameterComponent:
        parts = _unwrapped(name).split(".")
        root = parts[0]
        route_component = parts[2] if root == "interfaces" and len(parts) > 2 else root
        if route_component in {
            "ingestion",
            "ingestion_adapter",
            "static_condition_encoder",
            "preconditioning_image_encoder",
            "prefix_projection",
        }:
            return "ingestion"
        if route_component in {"decoder", "prediction_scales"}:
            return "decoder"
        return "backbone"

    @staticmethod
    @beartype
    def _kind(name: str) -> ParameterKind:
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "weight":
            return "weight"
        if leaf == "bias":
            return "bias"
        return "other"

    @beartype
    def descriptors(self) -> tuple[ParameterDescriptor, ...]:
        return self._descriptors

    @beartype
    def parameter(self, parameter_id: str) -> nn.Parameter:
        resolved = self._aliases.get(parameter_id, parameter_id)
        try:
            return self._parameters[resolved]
        except KeyError as error:
            raise KeyError(
                f"Unknown parameter_id or alias: {parameter_id!r}."
            ) from error

    @beartype
    def select(
        self,
        *,
        component: str | None = None,
        semantic_group: str | None = None,
    ) -> tuple[ParameterDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in self._descriptors
            if (component is None or descriptor.component == component)
            and (
                semantic_group is None
                or semantic_group
                in (descriptor.semantic_aliases or (descriptor.semantic_group,))
            )
        )

    def parameter_ids(self, parameters: Iterable[nn.Parameter]) -> set[str]:
        try:
            return {self._ids[id(parameter)] for parameter in parameters}
        except KeyError as error:
            raise ValueError(
                "Parameter is outside the model parameter catalog"
            ) from error

    def matching_ids(self, selector: ParameterSelector, eligible: Set[str]) -> set[str]:
        """Resolve a selector and validate its globs within the eligible parameters."""
        paths = set(selector.exclude_modules)
        if selector.module is not None:
            paths.add(selector.module)
        if unknown := paths - self._modules:
            raise ValueError(f"unknown selector modules: {sorted(unknown)}")

        included_patterns = set()
        excluded_patterns = set()
        selected = set()
        for descriptor in self._descriptors:
            if descriptor.parameter_id not in eligible or not descriptor.in_scope(
                selector
            ):
                continue
            if selector.semantic_groups:
                matched = descriptor.matching_patterns(selector.semantic_groups)
                included_patterns.update(matched)
                if not matched:
                    continue
            excluded = descriptor.matching_patterns(selector.exclude_semantic_groups)
            excluded_patterns.update(excluded)
            if not excluded:
                selected.add(descriptor.parameter_id)
        for field, patterns, matched in (
            ("semantic_groups", selector.semantic_groups, included_patterns),
            (
                "exclude_semantic_groups",
                selector.exclude_semantic_groups,
                excluded_patterns,
            ),
        ):
            if missing := set(patterns or ()) - matched:
                raise ValueError(
                    f"{field} patterns {sorted(missing)} matched no parameters in the selector's eligible scope"
                )
        return selected

    @beartype
    def fingerprint(self) -> str:
        payload = [
            {
                "id": descriptor.parameter_id,
                "aliases": descriptor.aliases,
                "shape": descriptor.shape,
                "dtype": str(descriptor.dtype),
                "group": descriptor.semantic_group,
            }
            for descriptor in self._descriptors
        ]
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@beartype
def optimizer_group_id(descriptor: ParameterDescriptor) -> str:
    group = descriptor.semantic_group
    if group.startswith("decoder."):
        return group
    if descriptor.component == "decoder":
        if group.startswith("embedding."):
            return "decoder.embedding"
        if group.startswith(("attention.", "feed_forward.", "normalization")):
            return f"decoder.{group}"
    if group == "normalization":
        return f"{descriptor.component}.normalization"
    if group.startswith(("attention.", "feed_forward.")):
        return (
            f"ingestion.{group}" if descriptor.depth_parameter else f"backbone.{group}"
        )
    if group.startswith("embedding."):
        return group
    if descriptor.component == "ingestion":
        return "ingestion"
    return f"{descriptor.component}.{group}"


@beartype
def semantic_optimizer_groups(
    catalog: ParameterCatalog,
    *,
    parameters: set[str] | None = None,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[nn.Parameter]] = {}
    for descriptor in catalog.descriptors():
        if parameters is not None and descriptor.parameter_id not in parameters:
            continue
        parameter = catalog.parameter(descriptor.parameter_id)
        if not parameter.requires_grad:
            continue
        grouped.setdefault(optimizer_group_id(descriptor), []).append(parameter)

    base_options = dict(options or {})
    return [
        {"params": grouped[group_id], "group_id": group_id, **base_options}
        for group_id in sorted(grouped)
    ]
