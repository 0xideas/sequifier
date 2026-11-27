"""Window coordinates shared by dense and indexed sequence storage."""

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise

import numpy as np

from sequifier.typechecking import beartype


@dataclass(frozen=True)
class PredictionAlignment:
    """Preprocessing contract for split-end-aligned prediction groups."""

    splits: tuple[int, ...]
    prediction_length: int
    target_offset: int
    steps: int = 1

    @classmethod
    def from_settings(cls, settings):
        if not settings.prediction_aligned_splits:
            return None
        assert (
            settings.prediction_length is not None
            and settings.target_offset is not None
        )
        return cls(
            tuple(settings.prediction_aligned_splits),
            settings.prediction_length,
            settings.target_offset,
            settings.prediction_alignment_steps,
        )

    @classmethod
    def for_split(cls, settings, index):
        if index not in settings.prediction_aligned_splits:
            return None
        return cls.from_settings(settings)

    def required_width(self, context_length, offset):
        return context_length + offset + (self.steps - 1) * self.prediction_length

    def validate_width(self, width, view, source):
        required = self.required_width(view.context_length, view.target_offset)
        if width < required:
            raise ValueError(
                f"{source}: prediction alignment requires window width at least {required}"
            )

    def validate_decoder(self, view, prediction_length, source):
        if (view.target_offset, prediction_length) != (
            self.target_offset,
            self.prediction_length,
        ):
            raise ValueError(
                f"{source}: prediction alignment requires target_offset={self.target_offset} "
                f"and prediction_length={self.prediction_length}"
            )

    @property
    def spacing(self) -> int:
        return self.steps * self.prediction_length


def split_intervals(length: int, ratios: Sequence[float]) -> list[tuple[int, int]]:
    """Partition every item, including the final item when ratios round down."""
    cuts = (min(length, int(fraction * length)) for fraction in accumulate(ratios[:-1]))
    return list(pairwise((0, *cuts, length)))


def cutoff_intervals(values, cutoffs):
    if np.any(values[1:] < values[:-1]):
        raise ValueError(
            "split_column values must be non-decreasing within each sequence"
        )
    cuts = [0, *np.searchsorted(values, cutoffs, side="left").tolist(), len(values)]
    return list(pairwise(cuts))


@dataclass(frozen=True)
class SplitWindowPlan:
    widths: tuple[int, ...]
    stride: int
    ratios: tuple[float, ...]
    alignment: PredictionAlignment | None
    arrivals: bool = False

    @classmethod
    def from_settings(cls, settings, *, arrivals=False):
        return cls(
            tuple(settings.window_lengths),
            settings.resolved_window_stride,
            tuple(settings.split_ratios or ()),
            PredictionAlignment.from_settings(settings),
            arrivals,
        )

    def validate_training_share(self, row_counts: Sequence[float]):
        """Estimate window shares using each split's configured spacing."""
        weights = [
            count
            / (
                self.alignment.spacing
                if self.alignment and i in self.alignment.splits
                else self.stride
            )
            for i, count in enumerate(row_counts)
        ]
        total = sum(weights)
        if not total or weights[0] / total >= 0.5:
            return
        details = ", ".join(
            f"split{i}: {weight / total:.1%}" for i, weight in enumerate(weights)
        )
        message = (
            "Split 0 is estimated to have less than 50% of the windows written to disk: "
            f"{details}. Set ALLOW_LARGE_SPLITS=1 to allow this."
        )
        if os.environ.get("ALLOW_LARGE_SPLITS") == "1":
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            raise ValueError(message)

    def placements(self, first, count, *, assigned=None, bounds=None):
        """Place windows in an assigned split, cutoff intervals, or ratio-based intervals."""
        if assigned is not None:
            intervals = [(assigned, (0, count))]
        else:
            intervals = enumerate(
                split_intervals(count, self.ratios) if bounds is None else bounds
            )
        for split, (low, high) in intervals:
            if low < high:
                yield (
                    split,
                    WindowPlacement.for_split(
                        first,
                        first + low,
                        first + high,
                        self.widths[split],
                        self.stride,
                        self.alignment
                        if self.alignment and split in self.alignment.splits
                        else None,
                        arrivals=self.arrivals,
                    ),
                )


def window_count(
    span: int, width: int, stride: int, alignment: PredictionAlignment | None = None
) -> int:
    if span <= 0:
        return 0
    if alignment is not None:
        return (span + alignment.spacing - 1) // alignment.spacing
    return 1 + (max(0, span - width) + stride - 1) // stride


@beartype
def get_subsequence_starts(
    in_context_length: int, window_length: int, stride_for_split: int
) -> np.ndarray:
    return WindowPlacement.for_split(
        0, 0, in_context_length, window_length, stride_for_split
    ).starts


@dataclass(frozen=True)
class WindowPlacement:
    width: int
    context_start: int
    low: int
    high: int
    starts: np.ndarray
    padding: np.ndarray

    @classmethod
    def for_split(
        cls,
        first,
        low,
        high,
        width,
        stride,
        alignment=None,
        *,
        begin=0,
        end=None,
        arrivals=False,
    ):
        """Place all or a slice of the windows in a nonempty target interval."""
        if high > np.iinfo(np.int64).max:
            raise ValueError("splitEndItemPosition falls outside signed Int64.")
        context_start = (
            max(first, low - width + 1) if arrivals and alignment is None else low
        )
        count = window_count(high - context_start, width, stride, alignment)
        end = count if end is None else end
        if not 0 <= begin <= end <= count:
            raise ValueError(f"Window slice [{begin}, {end}) is outside [0, {count})")
        indices = np.arange(begin, end, dtype=np.int64)
        if alignment is None:
            first = context_start
            distance = high - first - width
            if count <= 1:
                offsets = np.full(len(indices), distance, dtype=np.int64)
            else:
                # Match linspace's evenly spaced starts and exact final endpoint.
                offsets = np.floor(indices * (distance / (count - 1))).astype(np.int64)
                if len(indices) and end == count:
                    offsets[-1] = distance
        else:
            offsets = high - first - width - (count - 1 - indices) * alignment.spacing
        if len(offsets) and first + int(offsets[0]) < np.iinfo(np.int64).min:
            raise ValueError(
                "startItemPosition falls outside signed Int64 after applying left padding."
            )
        return cls(width, first, low, high, first + offsets, np.maximum(0, -offsets))

    def __iter__(self):
        return zip(map(int, self.starts), map(int, self.padding))

    def gather(self, values, first):
        """Gather dense feature windows from one contiguous sequence."""
        values = values[self.context_start - first : self.high - first]
        pad = int(self.padding[0])
        if pad:
            values = np.pad(values, (pad, 0), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(values, self.width)
        return np.ascontiguousarray(windows[self.starts - self.context_start + pad])

    def curriculum(self, columns, first, sequence):
        if not columns:
            return None
        offsets = self.starts - self.context_start
        starts = np.maximum(offsets, 0)
        stops = offsets + self.width
        samples = []
        for values in columns.values():
            values = values[self.context_start - first : self.high - first]
            changes = np.concatenate(
                ([0], np.cumsum(values[1:] != values[:-1], dtype=np.int64))
            )
            invalid = np.flatnonzero(changes[stops - 1] != changes[starts])
            if len(invalid):
                raise ValueError(
                    "Curriculum columns must be identical within each generated "
                    f"subsequence; sequenceId={sequence}, subsequenceId={int(invalid[0])}"
                )
            samples.append(values[starts].astype(np.int64))
        return np.column_stack(samples)
