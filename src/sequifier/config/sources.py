"""Named dataset and part references shared by training algorithms."""

from typing import Optional

from pydantic import ConfigDict, Field, field_validator

from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.naming import ConfigModel
from sequifier.typechecking import beartype


def phase_sources(phase):
    """All authored references, including held-out and preference sources."""
    refs = list(getattr(phase, "sources", []))
    refs.extend(getattr(phase.evaluation, "sources", None) or [])
    if (preference := getattr(phase, "preferences", None)) is not None:
        refs.append(preference)
    return refs


def validate_source(config, source, usage):
    name, _, part = source.partition(".")
    if name not in config.dataset_training or (
        part and part not in config.dataset_training[name].parts
    ):
        raise ValueError(f"unknown {usage} source {source!r}")


class TrainingSourceSpecModel(ConfigModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    selection_weight: Optional[float] = Field(default=None, gt=0)
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
