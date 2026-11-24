"""Explicit sequence import with the same source and authored task contracts."""

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
        if not observations and self.source.manifest_path:
            manifest = (source or self.resolved_source()).manifest or {}
            observations = list(manifest.get("observation_columns", ()))
            static = static or list(manifest.get("static_columns", ()))
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

    @model_validator(mode="after")
    def validate_resources(self):
        limited = set(self.cardinality_config)
        if limited & (
            set(self.policy_binding.actions) | set(self.policy_binding.feedback)
        ):
            raise ValueError(
                "cardinality_config cannot change action or feedback vocabularies"
            )
        if limited & set(self.real_columns or ()) or (
            self.categorical_columns and limited - set(self.categorical_columns)
        ):
            raise ValueError("cardinality_config must select categorical features")
        self.validate_split_method()
        self.validate_window_placement()
        return self
