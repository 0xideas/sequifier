from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from sequifier.model.parameter_groups import semantic_parameter_groups
from sequifier.typechecking import beartype

ParameterComponent = Literal["ingestion", "backbone", "decoder"]
ParameterKind = Literal["weight", "bias", "other"]


@beartype
def _canonical_name(name: str) -> str:
    return name.replace("_orig_mod.", "")


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


class ParameterCatalog:
    @beartype
    def __init__(self, model: nn.Module):
        self.model = model
        grouped = semantic_parameter_groups(model)
        semantic_by_identity = {
            id(parameter): group
            for group, parameters in grouped.items()
            for parameter in parameters
        }

        aliases_by_identity: dict[int, list[str]] = {}
        parameter_by_identity: dict[int, nn.Parameter] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            identity = id(parameter)
            parameter_by_identity[identity] = parameter
            canonical = _canonical_name(name)
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
            descriptor = ParameterDescriptor(
                parameter_id=parameter_id,
                canonical_name=canonical_name,
                aliases=tuple(names),
                component=component,
                semantic_group=str(
                    semantic_by_identity.get(identity, "free_parameter")
                ),
                parameter_kind=self._kind(canonical_name),
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                shared_parameter_id=parameter_id if len(names) > 1 else None,
            )
            descriptors.append(descriptor)
            parameters[parameter_id] = parameter
            aliases.update({name: parameter_id for name in names})

        self._descriptors = tuple(descriptors)
        self._parameters = parameters
        self._aliases = aliases

    @staticmethod
    @beartype
    def _component(name: str) -> ParameterComponent:
        root = name.split(".", 1)[0]
        if root in {"ingestion", "ingestion_adapter"}:
            return "ingestion"
        if root == "decoder":
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
            and (semantic_group is None or descriptor.semantic_group == semantic_group)
        )

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
    if group.startswith(("attention.", "feed_forward.", "normalization")):
        return f"backbone.{group}"
    if group.startswith("embedding."):
        return group
    if descriptor.component == "ingestion":
        return "ingestion"
    return f"{descriptor.component}.{group}"


@beartype
def semantic_optimizer_groups(
    catalog: ParameterCatalog,
    *,
    parameters: set[int] | None = None,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[nn.Parameter]] = {}
    seen: set[int] = set()
    for descriptor in catalog.descriptors():
        parameter = catalog.parameter(descriptor.parameter_id)
        identity = id(parameter)
        if identity in seen or (parameters is not None and identity not in parameters):
            continue
        if not parameter.requires_grad:
            continue
        seen.add(identity)
        grouped.setdefault(optimizer_group_id(descriptor), []).append(parameter)

    base_options = dict(options or {})
    return [
        {"params": grouped[group_id], "group_id": group_id, **base_options}
        for group_id in sorted(grouped)
    ]
