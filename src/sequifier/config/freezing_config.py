"""Typed component fields for semantic layer freezing."""

from typing import Optional

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from sequifier.config.layer_groups import LayerGroup
from sequifier.config.naming import ConfigModel
from sequifier.typechecking import beartype


class LayerFreezingConfigFields(ConfigModel):
    """Mutually exclusive semantic layer-freezing selections."""

    model_config = ConfigDict(extra="forbid")

    freeze_groups: Optional[list[LayerGroup]] = None
    freeze_except_groups: Optional[list[LayerGroup]] = None

    @field_validator("freeze_groups", "freeze_except_groups")
    @classmethod
    @beartype
    def validate_unique_groups(
        cls, value: Optional[list[LayerGroup]]
    ) -> Optional[list[LayerGroup]]:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("layer freezing groups cannot contain duplicates")
        return value

    @model_validator(mode="after")
    @beartype
    def validate_mutually_exclusive(self):
        if self.freeze_groups is not None and self.freeze_except_groups is not None:
            raise ValueError(
                "freeze_groups and freeze_except_groups are mutually exclusive; "
                "only one may be non-null"
            )
        return self

    @model_serializer(mode="wrap")
    @beartype
    def serialize_freezing_fields(self, serializer):
        """Omit inactive fields from canonical dataset freezing policies."""

        values = serializer(self)
        if self.freeze_groups is None:
            values.pop("freeze_groups", None)
        if self.freeze_except_groups is None:
            values.pop("freeze_except_groups", None)
        return values

    @property
    @beartype
    def has_freezing_policy(self) -> bool:
        return self.freeze_groups is not None or self.freeze_except_groups is not None


class IngestionFreezingConfig(LayerFreezingConfigFields):
    """Recursive policies following the declared composite branch tree."""

    branches: dict[str, "IngestionFreezingConfig"] = Field(default_factory=dict)

    @property
    def has_freezing_policy(self) -> bool:
        return super().has_freezing_policy or any(
            child.has_freezing_policy for child in self.branches.values()
        )

    @model_serializer(mode="wrap")
    def serialize_ingestion_policy(self, serializer):
        values = serializer(self)
        if self.freeze_groups is None:
            values.pop("freeze_groups", None)
        if self.freeze_except_groups is None:
            values.pop("freeze_except_groups", None)
        if not self.branches:
            values.pop("branches", None)
        return values
