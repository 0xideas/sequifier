"""Configuration contract for history carried across temporal splits."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitContextConfig(BaseModel):
    """Describe whether later splits retain preceding sequence history."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["isolated", "preceding"] = "isolated"
    target_offset: Optional[int] = Field(default=None, ge=0)
    prediction_length: Optional[int] = Field(default=None, gt=0)
    contract_version: int = Field(default=1, ge=1, le=1)

    @model_validator(mode="after")
    def validate_contract(self) -> "SplitContextConfig":
        configured = (
            self.target_offset is not None or self.prediction_length is not None
        )
        if self.mode == "preceding" and (
            self.target_offset is None or self.prediction_length is None
        ):
            raise ValueError(
                "split_context preceding mode requires target_offset and "
                "prediction_length"
            )
        if self.mode == "isolated" and configured:
            raise ValueError(
                "split_context target_offset and prediction_length are only valid "
                "in preceding mode"
            )
        return self

    def halo_length(self, window_length: int, max_target_offset: int) -> int:
        """Return rows preceding a split needed by the declared prediction view."""
        if self.mode == "isolated":
            return 0
        assert self.target_offset is not None
        assert self.prediction_length is not None
        halo = (
            window_length
            - max_target_offset
            + self.target_offset
            - self.prediction_length
        )
        if halo < 0:
            raise ValueError(
                "split_context produces a negative history length; check "
                "target_offset and prediction_length"
            )
        return halo
