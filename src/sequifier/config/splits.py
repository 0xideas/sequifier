"""Named preprocessing partitions in temporal order."""

from datetime import date, datetime
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt

from sequifier.config.identifiers import identifier
from sequifier.config.naming import ConfigModel


class WindowSplit(ConfigModel):
    window_length: StrictInt = Field(gt=0)
    fraction: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    cutoff: int | str | date | datetime | None = None
    prediction_aligned: bool = False


class WindowStrideSettings(ConfigModel):
    window_stride: int | None = Field(default=None, gt=0)

    split_method: Literal["between_sequence", "within_sequence", "value_cutoff"] = (
        "within_sequence"
    )
    split_column: str | None = None
    splits: dict[str, WindowSplit] = Field(min_length=1)
    max_target_offset: int = Field(default=1, ge=0)
    prediction_length: int | None = Field(default=None, gt=0)
    prediction_alignment_steps: int = Field(default=1, gt=0)
    target_offset: int | None = Field(default=None, ge=0)

    @property
    def window_lengths(self):
        return [split.window_length for split in self.splits.values()]

    @property
    def prediction_aligned_splits(self):
        return [
            index
            for index, split in enumerate(self.splits.values())
            if split.prediction_aligned
        ]

    @property
    def resolved_window_stride(self) -> int:
        return (
            self.window_lengths[0] if self.window_stride is None else self.window_stride
        )

    def validate_window_placement(self):
        widths = self.window_lengths
        if any(width <= self.max_target_offset for width in widths):
            raise ValueError("Each split window_length must exceed max_target_offset")
        if any(
            width != widths[0] + self.prediction_alignment_steps - 1
            for width in widths[2:]
        ):
            raise ValueError(
                "All splits after the first two must have window_length equal to "
                "the training width + (prediction_alignment_steps - 1)"
            )
        aligned = self.prediction_aligned_splits
        if aligned:
            if self.prediction_length is None or self.target_offset is None:
                raise ValueError(
                    "An aligned split requires prediction_length and target_offset"
                )
            if self.target_offset != self.max_target_offset:
                raise ValueError(
                    "Aligned placement requires target_offset == max_target_offset"
                )
            if any(
                self.prediction_alignment_steps * self.prediction_length
                > widths[i] - self.max_target_offset
                for i in aligned
            ):
                raise ValueError(
                    "Aligned window width cannot hold all prediction steps "
                    "after max_target_offset"
                )
        elif self.prediction_length is not None or self.target_offset is not None:
            raise ValueError(
                "prediction_length and target_offset require at least one aligned split"
            )

    @property
    def split_ratios(self):
        if self.split_method == "value_cutoff":
            return None
        return [
            split.fraction
            for split in self.splits.values()
            if split.fraction is not None
        ]

    @property
    def split_values(self):
        if self.split_method != "value_cutoff":
            return None
        return [split.cutoff for split in list(self.splits.values())[:-1]]

    def validate_split_method(self):
        for name in self.splits:
            identifier(name, "Split")
        if next(iter(self.splits)) != "train":
            raise ValueError("The first split must be train (used for feature fitting)")
        entries = list(self.splits.values())
        if self.split_method == "value_cutoff":
            if (
                any(s.fraction is not None for s in entries)
                or any(s.cutoff is None for s in entries[:-1])
                or entries[-1].cutoff is not None
            ):
                raise ValueError(
                    "Cutoff partitions require a cutoff on every split except the last, and no fractions"
                )
        elif any(
            s.cutoff is not None or s.fraction is None for s in entries
        ) or not np.isclose(
            sum(s.fraction for s in entries if s.fraction is not None),
            1.0,
        ):
            raise ValueError(
                "Ratio partitions require fractions summing to one and no cutoffs"
            )
        if self.split_method == "value_cutoff":
            if not self.split_column:
                raise ValueError(
                    "split_column must be set when split_method is 'value_cutoff'"
                )
            if len(entries) < 2:
                raise ValueError("A cutoff partition requires at least two splits")
            if self.split_column in {"sequenceId", "itemPosition"}:
                raise ValueError("split_column cannot be sequenceId or itemPosition")
            if self.split_column == "__sequifier_split_value":
                raise ValueError(
                    "split_column cannot use reserved name '__sequifier_split_value'"
                )
        elif self.split_column is not None:
            raise ValueError(
                "split_column is only valid when split_method is 'value_cutoff'"
            )
