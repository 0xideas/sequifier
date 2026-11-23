"""Public inference requests, compiled to the model and storage resolver's inputs."""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

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
    format: Literal["csv", "parquet", "pt"] | None = None
    path: str | None = None
    split: str | None = None
    layout: Literal["auto", "raw", "indexed"] = "auto"
    metadata_config_path: str | None = None
    static_data_path: str | None = None
    mask_column: str | None = None
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None

    @model_validator(mode="after")
    def select_split(self):
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
    sequence_id_column: str = "sequenceId"
    item_position_column: str = "itemPosition"
    desired_return_column: str = "desired_return"


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
