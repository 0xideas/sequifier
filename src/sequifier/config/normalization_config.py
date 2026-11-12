"""Frozen interface normalization facts, independent of learning algorithms."""

import math
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sequifier.config.naming import ConfigModel


class FrozenNormalizationConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["identity", "statistics"]
    statistics: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_statistics(self):
        if self.source == "identity" and self.statistics:
            raise ValueError("identity normalization cannot include statistics")
        for column, values in self.statistics.items():
            if (
                set(values) != {"mean", "std"}
                or not all(math.isfinite(value) for value in values.values())
                or values["std"] < 0
            ):
                raise ValueError(
                    f"{column}: statistics require finite mean and nonnegative std"
                )
        return self
