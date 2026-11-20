"""Model-owned options shared by training models and portable artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from sequifier.config.naming import ConfigModel

DEFAULT_MAX_CONCAT_TENSORS = 12


class ModelExecutionOptions(ConfigModel):
    context_length: int = Field(gt=0)
    training_objective: str
    target_offset: int = Field(default=1, ge=0)
    forward_attention: Literal["causal", "bidirectional"] | None = None
    layer_type_dtypes: dict[str, str] | None = None
    max_concat_tensors: int = Field(default=DEFAULT_MAX_CONCAT_TENSORS, gt=0)

    @field_validator("training_objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        from sequifier.objectives import get_objective_class

        get_objective_class(value)
        return value

    @field_validator("layer_type_dtypes")
    @classmethod
    def validate_layer_dtypes(cls, value: dict[str, str] | None):
        if value is None:
            return value
        allowed_keys = {"embedding", "linear", "conv", "norm", "decoder"}
        allowed_types = {
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "float8_e4m3fn",
            "float8_e5m2",
        }
        if invalid := set(value) - allowed_keys:
            raise ValueError(f"Invalid layer_type_dtypes keys: {sorted(invalid)}")
        if invalid := set(value.values()) - allowed_types:
            raise ValueError(f"Invalid layer_type_dtypes values: {sorted(invalid)}")
        return value

    @property
    def attention(self) -> Literal["causal", "bidirectional"]:
        return self.forward_attention or (
            "bidirectional" if self.training_objective == "bert" else "causal"
        )
