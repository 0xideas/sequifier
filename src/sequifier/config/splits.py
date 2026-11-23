"""Named preprocessing partitions in temporal order."""

from abc import abstractmethod
from collections.abc import Sequence
from datetime import date, datetime

from pydantic import Field, StrictInt

from sequifier.config.naming import ConfigModel


class WindowSplit(ConfigModel):
    window_length: StrictInt = Field(gt=0)
    fraction: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    cutoff: int | str | date | datetime | None = None
    prediction_aligned: bool = False


class WindowStrideSettings(ConfigModel):
    window_stride: int | None = Field(default=None, gt=0)

    @property
    @abstractmethod
    def window_lengths(self) -> Sequence[int]: ...

    @property
    def resolved_window_stride(self) -> int:
        return (
            self.window_lengths[0] if self.window_stride is None else self.window_stride
        )
