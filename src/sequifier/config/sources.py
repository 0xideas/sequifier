"""Named dataset and part references shared by training algorithms."""

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import Field, field_validator

from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.naming import ConfigModel
from sequifier.typechecking import beartype


class SourceReference(ConfigModel):
    source: str

    @property
    def dataset(self) -> str:
        return self.source.partition(".")[0]

    @property
    def part(self) -> str:
        return self.source.partition(".")[2]

    def validate_location(self, datasets, usage):
        if self.dataset not in datasets or (
            not self.part
            or (
                not datasets[self.dataset].uses_metadata_views
                and self.part not in datasets[self.dataset].views
            )
        ):
            raise ValueError(f"unknown {usage} source {self.source!r}")

    @field_validator("source")
    @classmethod
    @beartype
    def validate_source(cls, value):
        parts = value.split(".")
        if len(parts) != 2:
            raise ValueError("Source references must select dataset.view")
        for part in parts:
            _identifier(part, "Source reference component")
        return value


class TrainingSourceSpecModel(SourceReference):
    selection_weight: Optional[float] = Field(default=None, gt=0)
    batches_per_selection: Optional[int] = Field(default=None, gt=0)


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
        training = tuple(phase.sources)
        evaluation = tuple(phase.evaluation.sources or ()) if phase.evaluation else ()
        return cls(kind, training, evaluation)

    def __iter__(self):
        yield from self.training
        yield from self.evaluation
