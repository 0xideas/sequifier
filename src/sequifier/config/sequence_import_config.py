"""Explicit sequence import with the same source and authored task contracts."""

import json
from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import Field, field_validator, model_validator

from sequifier.config.cardinality import CardinalityLimitModel
from sequifier.config.command_ownership import (
    CommandRunSettings,
    SeededPreprocessingRun,
)
from sequifier.config.policy_config import (
    PolicyBinding,
    PolicyConfigBase,
    ResolvedPolicyBinding,
)
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.splits import SequencePreprocessingSettings

CommandRun = TypeVar("CommandRun", bound=CommandRunSettings)


def select_policy_input(values, path, *, source_format=None):
    """Replace the configured source location while preserving feature bindings."""
    values.pop("preprocessing_data_path", None)
    source = values.setdefault("source", {})
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("preprocessing_data_path requires a nonempty path")
    run = values.get("run", {})
    root = (
        run.project_root
        if isinstance(run, CommandRunSettings)
        else run.get("project_root", ".")
    )
    location = Path(root) / path
    if (
        source_format is None
        and location.is_file()
        and location.suffix.lower() not in {".csv", ".parquet", ".pt"}
    ):
        with location.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(
                f"Expected a sequence manifest, indexed view or trajectory in {location}"
            )
        source_format = document.get("format")
    for key in ("paths", "split", "manifest_path"):
        source.pop(key, None)
    if source_format in {"sequifier.sequence-data", "sequifier.trajectory-source"}:
        source["manifest_path"] = str(path)
        if source_format == "sequifier.sequence-data":
            # Committed decisions carry domain values and explicit boundaries.
            for key in ("encoding", "episode_end", "format"):
                source.pop(key, None)
    else:
        source["paths"] = [str(path)]
        if source_format == "sequifier.sequence-view":
            from sequifier.io.indexed_sequences import IndexedView

            source["format"] = "prepared"
            source.setdefault(
                "metadata_config_path", str(IndexedView.load(location).metadata_path)
            )


class EpisodeSourceConfig(PolicyConfigBase, Generic[CommandRun]):
    run: CommandRun
    source: SequenceSourceConfig
    policy_binding: PolicyBinding
    initialize_from: str | None = None
    model_interface: str | None = None
    reprocess: bool = False

    @model_validator(mode="after")
    def validate_interface_source(self):
        self.policy_binding.validate_source_units(self.source.units)
        if self.model_interface is not None and not self.initialize_from:
            raise ValueError("model_interface requires initialize_from")
        return self

    @property
    def project_root(self):
        return self.run.project_root

    def resolved_source(self):
        from sequifier.trajectory.sources import resolve_source

        return resolve_source(self.source, self.project_root)

    def resolved_policy(self, source=None) -> ResolvedPolicyBinding:
        if self.initialize_from:
            from sequifier.artifacts.model_artifact import read_execution_artifact
            from sequifier.artifacts.model_config import (
                model_execution_from_model_config,
            )
            from sequifier.rl.attachment import bind_policy

            execution, name = model_execution_from_model_config(
                read_execution_artifact(
                    Path(self.project_root) / self.initialize_from
                ).model_config.values,
                interface_name=self.model_interface,
            )
            if self.model_interface is None and len(execution.interface_names) > 1:
                raise ValueError(
                    "Select model_interface when initialize_from has multiple interfaces"
                )
            return bind_policy(execution, name, self.policy_binding)
        observations = list(self.source.columns.observations)
        static = list(self.source.columns.static_inputs)
        if not observations:
            resolved = source or self.resolved_source()
            manifest = resolved.manifest or {}
            observations = list(manifest.get("observation_columns", ()))
            static = static or list(manifest.get("static_columns", ()))
            if not observations and resolved.kind == "trajectory":
                from sequifier.trajectory.store import load_trajectory

                seed = load_trajectory(resolved.paths[0]).seed
                observations = list(seed.observations)
                static = static or list(seed.static_inputs)
        if not observations:
            raise ValueError(
                "sequence import requires source.columns.observations or initialize_from"
            )
        # Import validates and stores domain episodes, without constructing a model.
        return ResolvedPolicyBinding(
            **self.policy_binding.model_dump(),
            context_length=1,
            interface_name="default",
            observation_columns=observations,
            static_columns=static,
            action_order=list(self.policy_binding.actions),
        )


class SequenceImportConfig(EpisodeSourceConfig[CommandRunSettings]):
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    output_path: str = Field(min_length=1)
    operation: Literal["import_sequences"]
    passthrough_columns: list[str] = Field(default_factory=list)
    episodes_per_shard: int = Field(default=64, gt=0)

    @field_validator("passthrough_columns")
    @classmethod
    def validate_passthrough_columns(cls, names):
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("passthrough_columns must contain unique, non-empty names")
        return names


class PreparePolicyConfig(
    EpisodeSourceConfig[SeededPreprocessingRun], SequencePreprocessingSettings
):
    run: SeededPreprocessingRun = Field(default_factory=SeededPreprocessingRun)
    indexed_output_path: str = Field(min_length=1)
    operation: Literal["prepare_policy"]
    cardinality_config: dict[str, CardinalityLimitModel] = Field(default_factory=dict)
    normalize_on_all_data: bool = False

    normalize_real_columns: bool = True
    metadata_config_path: str | None = None
    mask_column: str | None = Field(default=None, min_length=1)
    indexed_passthrough: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def resolve_input_path(cls, values):
        if not isinstance(values, dict) or "preprocessing_data_path" not in values:
            return values
        source = dict(values.get("source", {}))
        if source.get("paths") or source.get("manifest_path") or source.get("split"):
            raise ValueError("Select preprocessing_data_path or a source location")
        values = {**values, "source": source}
        select_policy_input(values, values["preprocessing_data_path"])
        return values

    @model_validator(mode="after")
    def validate_resources(self):
        names = self.indexed_passthrough
        if len(set(names)) != len(names) or set(names) & {
            "",
            "sequenceId",
            "itemPosition",
            self.mask_column,
        }:
            raise ValueError("Indexed passthrough must name distinct auxiliary columns")
        limited = set(self.cardinality_config)
        # Action names can also denote independent observed features. Validate
        # cardinality against resolved observation/static columns during preparation.
        if limited & set(self.policy_binding.feedback):
            raise ValueError("cardinality_config cannot change feedback vocabularies")
        if limited & set(self.real_columns or ()) or (
            self.categorical_columns and limited - set(self.categorical_columns)
        ):
            raise ValueError("cardinality_config must select categorical features")
        self.validate_split_method()
        self.validate_window_placement()
        return self
