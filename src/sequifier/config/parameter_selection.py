"""One catalog selector language for trainability and optimizer routing."""

from fnmatch import fnmatchcase
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sequifier.config.naming import ConfigModel


class ParameterSelector(ConfigModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_groups: list[str] | None = None
    exclude_modules: list[str] = Field(default_factory=list)
    exclude_semantic_groups: list[str] = Field(default_factory=list)
    component: Literal["ingestion", "backbone", "decoder"] | None = None
    parameter_kind: Literal["weight", "bias", "other"] | None = None
    ndim: int | None = Field(default=None, ge=0)
    interface: str | None = None
    branch_path: list[str] | None = None
    module: str | None = None

    @model_validator(mode="after")
    def conditions(self):
        if all(getattr(self, key) in (None, []) for key in type(self).model_fields):
            raise ValueError("a parameter selector requires at least one condition")
        for key in (
            "semantic_groups",
            "branch_path",
            "exclude_modules",
            "exclude_semantic_groups",
        ):
            value = getattr(self, key)
            if value is not None and (
                (not value and key in {"semantic_groups", "branch_path"})
                or any(not x for x in value)
            ):
                raise ValueError(f"{key} requires nonempty entries")
        for key in ("module", "interface"):
            value = getattr(self, key)
            if value is not None and (
                not value or any(not x for x in value.split("."))
            ):
                raise ValueError(f"{key} must be nonempty")
        return self

    def matches(self, descriptor):
        names = list(descriptor.aliases)
        for alias in descriptor.aliases:
            parts = alias.split(".")
            while len(parts) > 1 and (parts[0] == "network" or parts[0].isdigit()):
                parts.pop(0)
            normalized = ".".join(parts)
            if normalized not in names:
                names.append(normalized)

        def in_module(name, module):
            return name == module or name.startswith(module + ".")

        if self.exclude_modules and descriptor.shared_parameter_id:
            excluded = [
                any(in_module(name, path) for path in self.exclude_modules)
                for name in descriptor.aliases
            ]
            if any(excluded) and not all(excluded):
                raise ValueError(
                    f"conflicting selector exclusions for shared parameter {descriptor.parameter_id}"
                )
        return (
            (
                self.semantic_groups is None
                or any(
                    fnmatchcase(group, p)
                    for group in (
                        descriptor.semantic_aliases or (descriptor.semantic_group,)
                    )
                    for p in self.semantic_groups
                )
            )
            and not any(
                fnmatchcase(group, p)
                for group in (
                    descriptor.semantic_aliases or (descriptor.semantic_group,)
                )
                for p in self.exclude_semantic_groups
            )
            and not any(in_module(n, m) for n in names for m in self.exclude_modules)
            and (self.component is None or descriptor.component == self.component)
            and (
                self.parameter_kind is None
                or descriptor.parameter_kind == self.parameter_kind
            )
            and (self.ndim is None or len(descriptor.shape) == self.ndim)
            and (
                self.branch_path is None
                or tuple(self.branch_path)
                == descriptor.branch_path[: len(self.branch_path)]
            )
            and (
                self.interface is None
                or any(
                    in_module(n, f"interfaces.{self.interface}")
                    or in_module(n, f"network.interfaces.{self.interface}")
                    for n in names
                )
            )
            and (self.module is None or any(in_module(n, self.module) for n in names))
        )

    def validate_semantic_patterns(self, descriptors):
        """Require every literal/glob to match within this selector's scope.

        Validate inclusions before subtraction, so overlapping or repeated patterns
        remain valid. Exclusions must match the positively selected scope. Never
        normalize, autocorrect, or silently discard an unmatched pattern.
        """
        structural = self.model_copy(
            update={"semantic_groups": None, "exclude_semantic_groups": []}
        )
        scoped = [
            descriptor for descriptor in descriptors if structural.matches(descriptor)
        ]
        positive = self.model_copy(update={"exclude_semantic_groups": []})
        included = [descriptor for descriptor in scoped if positive.matches(descriptor)]
        for field, candidates in (
            ("semantic_groups", scoped),
            ("exclude_semantic_groups", included),
        ):
            for pattern in getattr(self, field) or []:
                if not any(
                    fnmatchcase(group, pattern)
                    for descriptor in candidates
                    for group in (
                        descriptor.semantic_aliases or (descriptor.semantic_group,)
                    )
                ):
                    raise ValueError(
                        f"{field} pattern {pattern!r} matched no parameters "
                        "in the selector's eligible scope"
                    )


class RoleTrainability(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    freeze: list[ParameterSelector] | None = None
    freeze_except: list[ParameterSelector] | None = None

    @model_validator(mode="after")
    def exclusive(self):
        if self.freeze is not None and self.freeze_except is not None:
            raise ValueError("freeze and freeze_except are mutually exclusive")
        return self

    def resolve(self, module, *, eligible=None):
        from sequifier.model.parameter_catalog import ParameterCatalog

        catalog = ParameterCatalog(module)
        candidates = {
            d.parameter_id
            for d in catalog.descriptors()
            if eligible is None or id(catalog.parameter(d.parameter_id)) in eligible
        }
        known_modules = set()
        for name, _ in module.named_modules(remove_duplicate=False):
            known_modules.add(name)
            parts = name.split(".")
            while len(parts) > 1 and (parts[0] == "network" or parts[0].isdigit()):
                parts.pop(0)
            known_modules.add(".".join(parts))
        selected = set()
        for selector in (
            self.freeze if self.freeze is not None else self.freeze_except or []
        ):
            paths = selector.exclude_modules + (
                [selector.module] if selector.module else []
            )
            if unknown := set(paths) - known_modules:
                raise ValueError(f"unknown selector modules: {sorted(unknown)}")
            eligible_descriptors = [
                descriptor
                for descriptor in catalog.descriptors()
                if descriptor.parameter_id in candidates
            ]
            selector.validate_semantic_patterns(eligible_descriptors)
            matched = {
                descriptor.parameter_id
                for descriptor in eligible_descriptors
                if selector.matches(descriptor)
            }
            if not matched:
                raise ValueError(
                    f"trainability selector matched no parameters: {selector.model_dump(exclude_none=True)}"
                )
            selected.update(matched)
        if self.freeze_except is not None:
            selected = candidates - selected
        return frozenset(id(catalog.parameter(key)) for key in selected)


class RolePolicies(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    roles: dict[str, RoleTrainability] = Field(default_factory=dict)


class TrainabilityConfig(RolePolicies):
    by_dataset: dict[str, RolePolicies] = Field(default_factory=dict)

    def policy(self, role, dataset=None):
        override = self.by_dataset.get(dataset) if dataset is not None else None
        return (
            override.roles.get(role)
            if override and role in override.roles
            else self.roles.get(role)
        ) or RoleTrainability()

    def validate_roles(self, roles, *, datasets=None):
        for policy in [self, *self.by_dataset.values()]:
            if unknown := policy.roles.keys() - set(roles):
                raise ValueError(f"unsupported trainability roles: {sorted(unknown)}")
        if self.by_dataset and datasets is None:
            raise ValueError(
                "source-dependent trainability requires supervised updates"
            )
        if datasets is not None and (unknown := self.by_dataset.keys() - set(datasets)):
            raise ValueError(
                f"trainability overrides require training datasets: {sorted(unknown)}"
            )
