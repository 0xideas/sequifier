"""Model-owned options shared by training models and portable artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from sequifier.config.naming import ConfigModel

DEFAULT_MAX_CONCAT_TENSORS = 12


class _SharedExecutionOptions(ConfigModel):
    target_offset: int = Field(default=1, ge=0)
    forward_attention: Literal["causal", "bidirectional"] | None = None
    layer_type_dtypes: dict[str, str] | None = None
    max_concat_tensors: int = Field(default=DEFAULT_MAX_CONCAT_TENSORS, gt=0)

    @field_validator("training_objective", check_fields=False)
    @classmethod
    def validate_objective(cls, value: str | None) -> str | None:
        if value is not None:
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


class ExecutionOptions(_SharedExecutionOptions):
    context_length: int | None = Field(default=None, gt=0)
    training_objective: str | None = None


class ModelExecutionOptions(_SharedExecutionOptions):
    context_length: int = Field(gt=0)
    training_objective: str

    @property
    def attention(self) -> Literal["causal", "bidirectional"]:
        return self.forward_attention or (
            "bidirectional" if self.training_objective == "bert" else "causal"
        )

    @classmethod
    def from_training(cls, settings: ExecutionOptions) -> ModelExecutionOptions:
        context_length = settings.context_length
        training_objective = settings.training_objective
        if context_length is None or training_objective is None:
            raise ValueError(
                "Model execution requires context_length and training_objective"
            )
        return cls(
            context_length=context_length,
            training_objective=training_objective,
            target_offset=settings.target_offset,
            forward_attention=settings.forward_attention,
            layer_type_dtypes=settings.layer_type_dtypes,
            max_concat_tensors=settings.max_concat_tensors,
        )
