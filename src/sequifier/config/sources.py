"""Named dataset and part references shared by training algorithms."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sequifier.config.identifiers import identifier as _identifier
from sequifier.typechecking import beartype


class TrainingSourceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    weight: Optional[float] = Field(default=None, gt=0)
    batches_per_selection: Optional[int] = Field(default=None, gt=0)

    @field_validator("source")
    @classmethod
    @beartype
    def validate_source(cls, value):
        parts = value.split(".")
        if len(parts) > 2:
            raise ValueError("Source refs use only dataset or dataset.part")
        for part in parts:
            _identifier(part, "Source reference component")
        return value
