"""One catalog selector language for trainability and optimizer routing."""

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
        if all(value in (None, []) for _, value in self):
            raise ValueError("a parameter selector requires at least one condition")
        for key, value in (
            ("semantic_groups", self.semantic_groups),
            ("branch_path", self.branch_path),
            ("exclude_modules", self.exclude_modules),
            ("exclude_semantic_groups", self.exclude_semantic_groups),
        ):
            if value is not None and (
                (not value and key in {"semantic_groups", "branch_path"})
                or any(not x for x in value)
            ):
                raise ValueError(f"{key} requires nonempty entries")
        for key, value in (("module", self.module), ("interface", self.interface)):
            if value is not None and (
                not value or any(not x for x in value.split("."))
            ):
                raise ValueError(f"{key} must be nonempty")
        return self


class RoleTrainability(ConfigModel):
    freeze: list[ParameterSelector] | None = None
    freeze_except: list[ParameterSelector] | None = None

    @model_validator(mode="after")
    def exclusive(self):
        if self.freeze is not None and self.freeze_except is not None:
            raise ValueError("freeze and freeze_except are mutually exclusive")
        return self

    def frozen_ids(self, catalog, eligible):
        selected = set()
        for selector in (
            self.freeze if self.freeze is not None else self.freeze_except or ()
        ):
            matched = catalog.matching_ids(selector, eligible)
            if not matched:
                raise ValueError(
                    f"trainability selector matched no parameters: {selector.model_dump(exclude_none=True)}"
                )
            selected.update(matched)
        return frozenset(
            eligible - selected if self.freeze_except is not None else selected
        )


class RolePolicies(ConfigModel):
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
