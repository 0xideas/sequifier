"""Explicit sequence import with the same source and authored task contracts."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from sequifier.config.command_ownership import CommandRunSettings
from sequifier.config.policy_config import (
    PolicyBinding,
    PolicyConfigBase,
    ResolvedPolicyBinding,
)
from sequifier.config.policy_train_config import SequenceSourceConfig


class SequenceImportConfig(PolicyConfigBase):
    operation: Literal["import_sequences"]
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    source: SequenceSourceConfig
    policy_binding: PolicyBinding
    initialize_from: str | None = None
    output_path: str
    episodes_per_shard: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def validate_resources(self):
        unsupported = self.run.model_fields_set & {"execution", "reproducibility"}
        if unsupported:
            raise ValueError(
                f"sequence import does not support run settings: {sorted(unsupported)}"
            )
        return self

    @property
    def project_root(self):
        return self.run.project_root

    def resolved_source(self) -> SequenceSourceConfig:
        from sequifier.config.policy_train_config import resolve_sequence_source

        return resolve_sequence_source(self.source, Path(self.project_root).resolve())

    def resolved_policy(self) -> ResolvedPolicyBinding:
        if self.initialize_from:
            from sequifier.artifacts.model_artifact import read_execution_artifact
            from sequifier.artifacts.model_config import (
                resolved_config_from_model_config,
            )
            from sequifier.rl.attachment import resolve_binding

            execution, name = resolved_config_from_model_config(
                read_execution_artifact(
                    Path(self.project_root) / self.initialize_from
                ).model_config.values,
                device="cpu",
            )
            if len(execution.interface_names) != 1:
                raise ValueError(
                    "sequence import initialize_from requires one model interface"
                )
            return resolve_binding(execution, name, self.policy_binding)
        observations = list(self.source.columns.observations)
        static = list(self.source.columns.static_inputs)
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
            input_schema={},
        )
