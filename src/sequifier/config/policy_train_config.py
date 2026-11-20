"""Authored and resolved trajectory training contracts for the train command."""

from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from sequifier.config.policy_config import NonemptyStr, PolicyConfigBase


class PolicyTableColumns(PolicyConfigBase):
    """Canonical feature names for tables without a model input schema."""

    observations: list[NonemptyStr] = Field(default_factory=list)
    static_inputs: list[NonemptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_distinct_columns(self):
        names = self.observations + self.static_inputs
        if len(names) != len(set(names)):
            raise ValueError("observation and static columns must be distinct")
        return self


class SequenceSourceConfig(PolicyConfigBase):
    _project_root: str = PrivateAttr(default=".")
    _resolved_split_index: int | None = PrivateAttr(default=None)
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
    split_index: int | None = Field(default=None, ge=0)
    action_semantics: Literal["policy", "executed_predictions"] = "policy"

    @property
    def project_root(self) -> str:
        """Run-owned media root, excluded from the authored and serialized schema."""
        return self._project_root

    @property
    def selected_split_index(self) -> int | None:
        """Keep the split selection available after expanding it into paths."""
        return (
            self.split_index
            if self.split_index is not None
            else self._resolved_split_index
        )

    @model_validator(mode="after")
    def validate_location(self):
        metadata_split = (
            self.metadata_config_path is not None and self.split_index is not None
        )
        if sum((bool(self.paths), self.manifest_path is not None, metadata_split)) != 1:
            raise ValueError("trajectory source requires either paths or one manifest")
        if (
            self.split_index is not None
            or self.encoding == "model"
            or self.format == "prepared"
        ) and self.metadata_config_path is None:
            raise ValueError(
                "prepared/model-encoded sources and split_index require metadata_config_path"
            )
        return self


class PolicyDataSettings(PolicyConfigBase):
    sources: dict[str, SequenceSourceConfig]
    preferences: dict[str, str]
    source_validation_fractions: dict[str, float]
    seed: int
    episode_cache_size: int


def resolve_sequence_source(
    source: SequenceSourceConfig, root: Path, *, metadata=None
) -> SequenceSourceConfig:
    values = source.model_dump()
    values["paths"] = [str((root / path).absolute()) for path in source.paths]
    for key in ("manifest_path", "metadata_config_path", "static_data_path"):
        if values[key] is not None:
            values[key] = str((root / values[key]).absolute())
    if source.metadata_config_path is not None and (
        source.static_data_path is None or source.split_index is not None
    ):
        from sequifier.config.metadata import load_dataset_metadata

        if metadata is None:
            metadata = load_dataset_metadata(values["metadata_config_path"])
        prepared = (
            source.format == "prepared"
            or source.split_index is not None
            or any(Path(path).suffix.lower() == ".pt" for path in source.paths)
        )
        if (
            prepared
            and source.static_data_path is None
            and metadata.static_features_path
        ):
            values["static_data_path"] = str(
                (root / metadata.static_features_path).absolute()
            )
        if source.split_index is not None:
            if source.split_index >= len(metadata.split_paths):
                raise ValueError(
                    "source split_index is outside the prepared metadata splits"
                )
            path = (root / metadata.split_paths[source.split_index]).absolute()
            values.update(
                paths=[str(path)],
                format="prepared",
                split_index=None,
            )
    resolved = SequenceSourceConfig.model_validate(values)
    resolved._project_root = str(root)
    resolved._resolved_split_index = source.selected_split_index
    return resolved
