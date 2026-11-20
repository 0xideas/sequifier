"""Named dataset and part references shared by training algorithms."""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import Field, field_validator

from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.naming import ConfigModel
from sequifier.typechecking import beartype


def validate_source(config, source, usage):
    name, _, part = source.partition(".")
    if name not in config.dataset_training or (
        part and part not in config.dataset_training[name].parts
    ):
        raise ValueError(f"unknown {usage} source {source!r}")


class TrainingSourceSpecModel(ConfigModel):
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


SourceKind = Literal["sequence", "preference", "none"]


@dataclass(frozen=True)
class PhaseSources:
    kind: SourceKind
    training: tuple[TrainingSourceSpecModel, ...] = ()
    evaluation: tuple[TrainingSourceSpecModel, ...] = ()

    @classmethod
    def from_phase(cls, phase):
        kind = phase.source_kind
        if kind == "none":
            return cls(kind)
        training = (
            (phase.preferences,) if kind == "preference" else tuple(phase.sources)
        )
        evaluation = tuple(phase.evaluation.sources or ()) if phase.evaluation else ()
        return cls(kind, training, evaluation)

    def __iter__(self):
        yield from self.training
        yield from self.evaluation
