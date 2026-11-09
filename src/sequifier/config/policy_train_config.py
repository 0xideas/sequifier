"""Authored and resolved trajectory training contracts for the train command."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from sequifier.config.policy_config import NonemptyStr, PolicyConfigBase


class PolicyTableColumns(PolicyConfigBase):
    """Explicit row timing: current observation/action, then resulting reward."""

    sequence_id: NonemptyStr = "sequenceId"
    item_position: NonemptyStr = "itemPosition"
    observations: dict[str, NonemptyStr] = Field(default_factory=dict)
    static_inputs: dict[str, NonemptyStr] = Field(default_factory=dict)
    action: NonemptyStr | None = None
    reward: NonemptyStr | None = "reward"
    terminated: NonemptyStr | None = "terminated"
    truncated: NonemptyStr | None = "truncated"
    previous_action: NonemptyStr | None = None
    successors: dict[str, NonemptyStr] = Field(default_factory=dict)
    bootstrap_allowed: NonemptyStr | None = None
    group: NonemptyStr | None = None
    episode_id: NonemptyStr | None = None


class SequenceSourceConfig(PolicyConfigBase):
    units: dict[str, str] | None = None
    paths: list[NonemptyStr] = Field(default_factory=list)
    manifest: NonemptyStr | None = None
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

    @model_validator(mode="after")
    def validate_location(self):
        metadata_split = (
            self.metadata_config_path is not None and self.split_index is not None
        )
        if sum((bool(self.paths), self.manifest is not None, metadata_split)) != 1:
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


def resolve_sequence_source(
    source: SequenceSourceConfig, root: Path
) -> SequenceSourceConfig:
    values = source.model_dump()
    values["paths"] = [str((root / path).resolve()) for path in source.paths]
    for key in ("manifest", "metadata_config_path", "static_data_path"):
        if values[key] is not None:
            values[key] = str((root / values[key]).resolve())
    if source.metadata_config_path is not None and (
        source.static_data_path is None or source.split_index is not None
    ):
        from sequifier.config.metadata import load_dataset_metadata

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
                (root / metadata.static_features_path).resolve()
            )
        if source.split_index is not None:
            if source.split_index >= len(metadata.split_paths):
                raise ValueError(
                    "source split_index is outside the prepared metadata splits"
                )
            path = (root / metadata.split_paths[source.split_index]).resolve()
            pattern = "*.pt" if any(path.glob("*.pt")) else "*.parquet"
            values.update(
                paths=sorted(str(item) for item in path.glob(pattern))
                if path.is_dir()
                else [str(path)],
                format="prepared",
                split_index=None,
            )
    return SequenceSourceConfig.model_validate(values)
