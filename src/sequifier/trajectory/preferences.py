"""Versioned reward-free preference manifests over contiguous decision spans."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreferenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    start: int = Field(default=0, ge=0, strict=True)
    end: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def bounds(self):
        if self.end <= self.start:
            raise ValueError(
                "preference spans require start < end (zero-based, end-exclusive)"
            )
        return self


class PreferencePair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pair_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    group_id: str | None = None
    chosen: PreferenceSpan
    rejected: PreferenceSpan


class PreferenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["sequifier.preference-pairs"]
    version: Literal[1]
    pairs: list[PreferencePair] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [pair.pair_id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("preference pair IDs must be unique")
        return self


def load_preferences(path):
    return PreferenceManifest.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
