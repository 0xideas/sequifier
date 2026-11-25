"""Public inference requests, compiled to the model and storage resolver's inputs."""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from sequifier.config.column_names import (
    validate_column_names,
    validate_policy_column_types,
)
from sequifier.config.command_ownership import InferenceRunSettings
from sequifier.config.image_config import MediaPath
from sequifier.config.naming import ConfigModel
from sequifier.config.policy_infer_config import PolicyRolloutConfig
from sequifier.config.rollout_config import RolloutConfig


class ArtifactSelection(ConfigModel):
    paths: list[str] = Field(min_length=1)
    interface: str | None = None


class TrainingRoute(ConfigModel):
    config_path: str
    dataset: str | None = None
    view: str | None = None


class InferenceInput(ConfigModel):
    training_route: TrainingRoute | None = None
    format: Literal["csv", "parquet", "pt", "sequence"] | None = None
    path: str | None = None
    split: str | None = None
    layout: Literal["auto", "raw", "indexed"] = "auto"
    sequence_view: Literal["decisions", "observations"] = "decisions"
    metadata_config_path: str | None = None
    static_data_path: str | None = None
    mask_column: str | None = None
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None

    @field_validator("mask_column")
    @classmethod
    def validate_mask(cls, value):
        if value:
            validate_column_names(
                [value], "input.mask_column", allowed=("__observation_mask",)
            )
        return value

    @model_validator(mode="after")
    def select_split(self):
        if (
            self.format == "sequence" or self.sequence_view != "decisions"
        ) and self.layout != "raw":
            raise ValueError("Sequence manifest input requires input.layout: raw")
        if self.split is not None and (
            self.path or self.layout == "raw" or not self.has_split_metadata()
        ):
            raise ValueError(
                "input.split selects a named split from the supplied preprocessing metadata"
            )
        return self

    def has_split_metadata(self):
        return self.training_route is None and bool(self.metadata_config_path)


class PredictionInput(InferenceInput):
    inline_metadata: dict | None = None
    static_features_path: str | None = None
    window_stride: int | None = Field(default=None, gt=0)
    fresh_prediction_positions: Literal["right_edge", "observed"] | None = None

    def has_split_metadata(self):
        return self.training_route is None and (
            bool(self.metadata_config_path) or self.inline_metadata is not None
        )


class PolicyInput(InferenceInput):
    previous_action_columns: dict[str, str] = Field(default_factory=dict)
    desired_return_column: str = "desired_return"

    @model_validator(mode="after")
    def validate_columns(self):
        validate_column_names(
            self.previous_action_columns, "previous_action_columns actions"
        )
        validate_column_names(
            self.previous_action_columns.values(), "previous_action_columns inputs"
        )
        validate_column_names([self.desired_return_column], "desired_return_column")
        return self


class ContractAssertions(ConfigModel):
    input_columns: list[str] | None = None
    static_columns: list[str] | None = None
    target_columns: list[str] | None = None
    column_data_types: dict[str, str] | None = None
    target_column_types: dict[str, str] | None = None
    training_objective: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int | None = Field(default=None, ge=0)
    prediction_length: int | None = Field(default=None, gt=0)
    decoder_support_length: int | None = Field(default=None, gt=0)

    @field_validator(
        "input_columns",
        "target_columns",
        "static_columns",
        "column_data_types",
        "target_column_types",
    )
    @classmethod
    def validate_columns(cls, values, info):
        validate_column_names(
            values or (),
            info.field_name,
            allowed=("__policy_return", "__policy_previous_action"),
            prefixes=("__executed_",),
        )
        return values


class ForecastAssertions(ContractAssertions):
    prediction_alignment_steps: int | None = Field(default=None, gt=0)


class TableOutput(ConfigModel):
    path: str | None = None
    format: Literal["csv", "parquet"] = "csv"


class PolicyOutput(TableOutput):
    output_probabilities: bool = False


class ForecastOutput(PolicyOutput):
    decode_categories: bool = True


class InferenceRequest(ConfigModel):
    artifact: ArtifactSelection
    run: InferenceRunSettings = Field(default_factory=InferenceRunSettings)

    def resolver_values(self):
        values = self.model_dump(exclude_unset=True, exclude_none=True)
        artifact = values.pop("artifact")
        values["model_paths"] = artifact["paths"]
        if "interface" in artifact:
            values["model_interface"] = artifact["interface"]
        values.update(values.pop("assertions", {}))
        inputs = values.pop("input")
        route = inputs.pop("training_route", {})
        for old, new in (
            ("config_path", "training_config_path"),
            ("dataset", "dataset"),
            ("view", "part"),
        ):
            if old in route:
                values[new] = route[old]
        for old, new in (
            ("path", "data_path"),
            ("format", "read_format"),
            ("layout", "input_layout"),
        ):
            if old in inputs:
                inputs[new] = inputs.pop(old)
        values.update(inputs)
        output = values.pop("output", {})
        if "path" in output:
            values["output_path"] = output.pop("path")
        if "format" in output:
            values["write_format"] = output.pop("format")
        values.update(output)
        if "generation_steps" in values:
            values["autoregressive_generation"] = True
        return values


class ForecastRequest(InferenceRequest):
    task: Literal["forecast"]
    input: PredictionInput
    assertions: ForecastAssertions = Field(default_factory=ForecastAssertions)
    output: ForecastOutput | None = None
    rollout: RolloutConfig | None = None
    generation_steps: int | None = Field(default=None, gt=0)
    sample_from_distribution_columns: list[str] | None = None
    infer_with_dropout: bool = False

    @model_validator(mode="after")
    def select_output(self):
        if self.rollout and (self.output or self.generation_steps is not None):
            raise ValueError("A forecast rollout produces its own trajectory")
        return self


class EmbeddingRequest(InferenceRequest):
    task: Literal["embedding"]
    input: PredictionInput
    assertions: ForecastAssertions = Field(default_factory=ForecastAssertions)
    output: TableOutput | None = None
    infer_with_dropout: bool = False


class PolicyRequest(InferenceRequest):
    task: Literal["policy"]
    assertions: ContractAssertions = Field(default_factory=ContractAssertions)
    input: PolicyInput
    action_selection: Literal["deterministic", "stochastic"] = "deterministic"
    desired_return: float | None = Field(default=None, allow_inf_nan=False)
    output: PolicyOutput | None = None
    rollout: PolicyRolloutConfig | None = None

    @model_validator(mode="after")
    def select_output(self):
        self.assertions.column_data_types = validate_policy_column_types(
            self.assertions.column_data_types
        )
        if self.rollout and self.output:
            raise ValueError("A policy rollout produces its own trajectory")
        if len(self.artifact.paths) != 1:
            raise ValueError("Policy inference requires one artifact")
        return self


REQUESTS = TypeAdapter(
    Annotated[
        ForecastRequest | EmbeddingRequest | PolicyRequest, Field(discriminator="task")
    ]
)


def inference_overrides(values):
    """Project CLI conveniences onto the same paths used by YAML and search."""
    overrides = dict(values)
    paths = {
        "model_paths": ("artifact", "paths"),
        "model_interface": ("artifact", "interface"),
        "data_path": ("input", "path"),
        "input_layout": ("input", "layout"),
        "metadata_config_path": ("input", "metadata_config_path"),
        "dataset": ("input", "training_route", "dataset"),
        "view": ("input", "training_route", "view"),
        "input_columns": ("assertions", "input_columns"),
    }
    if overrides.get("data_path") is not None:
        overrides.setdefault("input", {})["split"] = None
    for key, path in paths.items():
        if key in overrides:
            value = overrides.pop(key)
            node = overrides
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
    return overrides
