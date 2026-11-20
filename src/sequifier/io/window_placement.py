"""Window coordinates shared by dense and indexed sequence storage."""

import math
from dataclasses import dataclass

import numpy as np

from sequifier.typechecking import beartype


@dataclass(frozen=True)
class PredictionAlignment:
    """Preprocessing contract for split-end-aligned prediction groups."""

    splits: tuple[int, ...]
    prediction_length: int
    target_offset: int
    steps: int = 1

    def starts(
        self, split_start: int, split_stop: int, window_length: int
    ) -> np.ndarray:
        """Return chronological, possibly negative, starts relative to a sequence."""
        spacing = self.steps * self.prediction_length
        count = (split_stop - split_start + spacing - 1) // spacing
        return np.asarray(
            [
                split_stop - window_length - i * spacing
                for i in range(count - 1, -1, -1)
            ],
            dtype=np.int64,
        )


@beartype
def get_subsequence_starts(
    in_context_length: int,
    window_length: int,
    stride_for_split: int,
) -> np.ndarray:
    """Return distributed window starts, including the final available start."""
    last_available_start = in_context_length - window_length
    num_subsequences = math.ceil(last_available_start / stride_for_split) + 1
    starts = np.linspace(0, last_available_start, num_subsequences, dtype=int)
    return np.unique(starts)


@dataclass(frozen=True)
class WindowPlacement:
    width: int
    context_start: int
    low: int
    high: int
    starts: np.ndarray
    padding: np.ndarray

    @classmethod
    def for_split(cls, first, low, high, width, stride, alignment=None):
        """Place windows in a nonempty target interval with optional earlier context."""
        if high > np.iinfo(np.int64).max:
            raise ValueError("splitEndItemPosition falls outside signed Int64.")
        if alignment is None:
            first = low
            span = high - first
            offsets = get_subsequence_starts(max(span, width), width, stride) - max(
                0, width - span
            )
        else:
            offsets = alignment.starts(low - first, high - first, width)
        if first + int(offsets[0]) < np.iinfo(np.int64).min:
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
