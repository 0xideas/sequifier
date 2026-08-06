"""Typed component fields for semantic layer freezing."""

from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_serializer,
    model_validator,
)

from sequifier.config.layer_groups import LayerGroup


class LayerFreezingConfigFields(BaseModel):
    """Mutually exclusive semantic layer-freezing selections."""

    model_config = ConfigDict(extra="forbid")

    freezing: Optional[list[LayerGroup]] = None
    freezing_except: Optional[list[LayerGroup]] = None

    @field_validator("freezing", "freezing_except")
    @classmethod
    def validate_unique_groups(
        cls, value: Optional[list[LayerGroup]]
    ) -> Optional[list[LayerGroup]]:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("layer freezing groups cannot contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.freezing is not None and self.freezing_except is not None:
            raise ValueError(
                "freezing and freezing_except are mutually exclusive; "
                "only one may be non-null"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_freezing_fields(self, serializer):
        """Omit inactive fields to preserve legacy resolved serialization."""

        values = serializer(self)
        if self.freezing is None:
            values.pop("freezing", None)
        if self.freezing_except is None:
            values.pop("freezing_except", None)
        return values

    @property
    def has_freezing_policy(self) -> bool:
        return self.freezing is not None or self.freezing_except is not None
