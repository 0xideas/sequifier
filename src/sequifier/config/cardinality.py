"""Categorical fitting settings shared by preprocessing and raw policy training."""

from typing import Optional

from pydantic import ConfigDict, Field, model_validator

from sequifier.config.naming import ConfigModel


class CardinalityHashingModel(ConfigModel):
    """Hash non-retained categorical values into a bounded set of buckets."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_buckets: int = Field(
        ge=1,
    )
    seed: int = 0


class CardinalityLimitModel(ConfigModel):
    """Bound the vocabulary produced for one categorical column."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    min_count: Optional[int] = Field(
        default=None,
        ge=1,
    )
    max_categories: Optional[int] = Field(
        default=None,
        ge=1,
    )
    hashing: Optional[CardinalityHashingModel] = None

    @model_validator(mode="after")
    def validate_options(self) -> "CardinalityLimitModel":
        if self.min_count is not None and self.max_categories is not None:
            raise ValueError(
                "max_categories and min_count are mutually exclusive cardinality limits"
            )
        if (
            self.min_count is None
            and self.max_categories is None
            and self.hashing is None
        ):
            raise ValueError(
                "cardinality configuration requires max_categories, min_count, or hashing"
            )
        return self
