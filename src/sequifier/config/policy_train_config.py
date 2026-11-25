"""Authored and resolved trajectory training contracts for the train command."""

from typing import Literal

from pydantic import Field, model_validator

from sequifier.config.column_names import validate_column_names
from sequifier.config.policy_config import NonemptyStr, PolicyConfigBase


class PolicyTableColumns(PolicyConfigBase):
    """Canonical feature names for tables without a model input schema."""

    observations: list[NonemptyStr] = Field(default_factory=list)
    static_inputs: list[NonemptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_distinct_columns(self):
        validate_column_names(
            self.observations,
            "source.columns.observations",
            prefixes=("__image:", "__image_mask:", "depth_valid_mask:"),
        )
        validate_column_names(
            self.static_inputs,
            "source.columns.static_inputs",
            allowed=("__preconditioning_image",),
        )
        names = self.observations + self.static_inputs
        if len(names) != len(set(names)):
            raise ValueError("observation and static columns must be distinct")
        return self


class SequenceSourceConfig(PolicyConfigBase):
    units: dict[str, str] | None = None
    paths: list[NonemptyStr] = Field(default_factory=list)
    manifest_path: NonemptyStr | None = None
    format: Literal["auto", "trajectory", "table", "prepared"] = "auto"
    columns: PolicyTableColumns = Field(default_factory=PolicyTableColumns)
    # A file boundary alone says nothing about task completion.
    episode_end: Literal["unknown", "terminated", "truncated"] = "unknown"
    episode_namespace: str = ""
    metadata_config_path: str | None = None
    encoding: Literal["raw", "model"] = "raw"
    static_data_path: str | None = None
    split: NonemptyStr | None = None
    action_semantics: Literal["policy", "executed_predictions"] = "policy"

    @model_validator(mode="after")
    def validate_location(self):
        metadata_split = (
            self.metadata_config_path is not None and self.split is not None
        )
        if sum((bool(self.paths), self.manifest_path is not None, metadata_split)) != 1:
            raise ValueError("trajectory source requires either paths or one manifest")
        if (
            self.split is not None
            or self.encoding == "model"
            or self.format == "prepared"
        ) and self.metadata_config_path is None:
            raise ValueError(
                "prepared/model-encoded sources and named splits require metadata_config_path"
            )
        return self


class PartitionView(PolicyConfigBase):
    """Physical sources partitioned together and the subset exposed by one view."""

    sources: list[str]
    part: str
    split: Literal["train", "validation"]
    validation_fraction: float


class PolicyDataSettings(PolicyConfigBase):
    project_root: str
    sources: dict[str, SequenceSourceConfig]
    preferences: dict[str, str]
    partitions: dict[str, PartitionView] = Field(default_factory=dict)
    seed: int
    episode_cache_size: int
