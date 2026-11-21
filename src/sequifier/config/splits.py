"""Named preprocessing partitions in temporal order."""

from datetime import date, datetime

from pydantic import Field, StrictInt

from sequifier.config.naming import ConfigModel


class WindowSplit(ConfigModel):
    window_length: StrictInt = Field(gt=0)
    fraction: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    cutoff: int | str | date | datetime | None = None
    prediction_aligned: bool = False
