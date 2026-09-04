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
from sequifier.typechecking import beartype


class LayerFreezingConfigFields(BaseModel):
    """Mutually exclusive semantic layer-freezing selections."""

    model_config = ConfigDict(extra="forbid")

    freeze: Optional[list[LayerGroup]] = None
    freezing_except: Optional[list[LayerGroup]] = None

    @field_validator("freeze", "freezing_except")
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
        if self.freeze is not None and self.freezing_except is not None:
            raise ValueError(
                "freeze and freezing_except are mutually exclusive; "
                "only one may be non-null"
            )
        return self

    @model_serializer(mode="wrap")
    @beartype
    def serialize_freezing_fields(self, serializer):
        """Omit inactive fields from canonical dataset freezing policies."""

        values = serializer(self)
        if self.freeze is None:
            values.pop("freeze", None)
        if self.freezing_except is None:
            values.pop("freezing_except", None)
        return values

    @property
    @beartype
    def has_freezing_policy(self) -> bool:
        return self.freeze is not None or self.freezing_except is not None
