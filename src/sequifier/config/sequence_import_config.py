"""Explicit sequence import with the same source and authored task contracts."""

from pathlib import Path
from typing import Generic, Literal, TypeVar

from pydantic import Field, model_validator

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
from sequifier.config.splits import WindowSplit

CommandRun = TypeVar("CommandRun", bound=CommandRunSettings)


class EpisodeSourceConfig(PolicyConfigBase, Generic[CommandRun]):
    run: CommandRun
    source: SequenceSourceConfig
    policy_binding: PolicyBinding
    initialize_from: str | None = None
    output_path: str
    continue_preprocessing: bool = False

    @property
    def project_root(self):
        return self.run.project_root

    def resolved_source(self):
        from sequifier.trajectory.sources import resolve_source

        return resolve_source(self.source, self.project_root)

    def resolved_policy(self) -> ResolvedPolicyBinding:
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
            )
            if len(execution.interface_names) != 1:
                raise ValueError(
                    "sequence import initialize_from requires one model interface"
                )
            return bind_policy(execution, name, self.policy_binding)
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
        )


class SequenceImportConfig(EpisodeSourceConfig[CommandRunSettings]):
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    operation: Literal["import_sequences"]
    episodes_per_shard: int = Field(default=64, gt=0)


class PreparePolicyConfig(EpisodeSourceConfig[SeededPreprocessingRun]):
    run: SeededPreprocessingRun = Field(default_factory=SeededPreprocessingRun)
    operation: Literal["prepare_policy"]
    splits: dict[Literal["train", "validation"], WindowSplit]

    @property
    def validation_fraction(self):
        return self.splits["validation"].fraction

    @property
    def window_lengths(self):
        return self.splits["train"].window_length, self.splits[
            "validation"
        ].window_length

    window_stride: int = Field(default=128, gt=0)
    max_target_offset: int = Field(default=1, ge=0)
    categorical_columns: list[str] | None = None
    real_columns: list[str] | None = None
    normalize_real_columns: bool = True
    metadata_config_path: str | None = None

    @model_validator(mode="after")
    def validate_resources(self):
        import math

        if (
            set(self.splits) != {"train", "validation"}
            or any(
                s.fraction is None or s.prediction_aligned or s.cutoff is not None
                for s in self.splits.values()
            )
            or not math.isclose(
                sum(s.fraction for s in self.splits.values() if s.fraction is not None),
                1,
            )
        ):
            raise ValueError(
                "Policy preparation requires train and validation fractions summing to one"
            )
        if any(width <= self.max_target_offset for width in self.window_lengths):
            raise ValueError("Each window width must exceed max_target_offset")
        return self
