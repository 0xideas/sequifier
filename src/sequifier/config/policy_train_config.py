"""Authored and resolved trajectory training contracts for the train command."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    NonemptyStr,
    PolicyConfigBase,
    PolicyOptimizationConfig,
)
from sequifier.config.training_plan import (
    PolicyTrainingPlanModel,
    ReturnConditionedPhaseConfig,
)


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


class _PolicyTrainConfigBase(PolicyConfigBase):
    project_root: str = "."
    model_name: str = Field(min_length=1)
    device: str = "cpu"
    initialize_from: str | None = None
    seed: int = Field(default=1010, ge=0)
    trajectory_sources: dict[str, SequenceSourceConfig] = Field(min_length=1)
    policies: dict[str, CategoricalPolicyConfig] = Field(min_length=1)
    training_plan: PolicyTrainingPlanModel
    # Epochs and seed belong to the phase and run respectively.
    optimization: PolicyOptimizationConfig = Field(
        default_factory=PolicyOptimizationConfig
    )

    @property
    def global_training(self) -> PolicyOptimizationConfig:
        """Expose the shared checkpoint-selection contract."""
        return self.optimization

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("model_name must be a single path component")
        return value

    @model_validator(mode="after")
    def validate_references(self):
        if len(self.training_plan.phases) != 1 or not isinstance(
            self.training_plan.phases[0], ReturnConditionedPhaseConfig
        ):
            raise ValueError("this learner supports one return_conditioned phase")
        phase = self.training_plan.phases[0]
        if phase.policy not in self.policies:
            raise ValueError(f"unknown policy {phase.policy!r}")
        for names in (phase.trajectory_sources, phase.validation_sources or []):
            if len(names) != len(set(names)):
                raise ValueError("trajectory source references must be unique")
            unknown = set(names) - self.trajectory_sources.keys()
            if unknown:
                raise ValueError(f"unknown trajectory sources: {sorted(unknown)}")
        if set(phase.trajectory_sources) & set(phase.validation_sources or []):
            raise ValueError("training and validation sources must be disjoint")
        return self


class PolicyTrainConfig(_PolicyTrainConfigBase):
    """Canonical model/dataset authoring, resolved before field validation."""

    model_path: NonemptyStr | None = None
    report_path: NonemptyStr | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_authoring(cls, values):
        from sequifier.config.train_config import ModelSpecModel

        values = dict(values)
        if {
            "policies",
            "trajectory_sources",
            "data_sources",
            "optimization",
            "actor_path",
        } & values.keys():
            raise ValueError(
                "use model, dataset_training, global_training and model_path"
            )
        model = ModelSpecModel.model_validate(values.pop("model", None))
        optimization = dict(values.pop("global_training", {}))
        context = optimization.pop(
            "context_length", model.backbone.architecture.max_context_length
        )
        if optimization.pop("training_objective", "policy") != "policy":
            raise ValueError("policy phases require training_objective: policy")
        values["optimization"] = optimization
        policies = {}
        for name, interface in model.interfaces.items():
            schema = interface.policy
            if schema is None:
                raise ValueError("policy model interfaces require a policy schema")
            if (
                interface.input_columns
                and interface.input_columns
                != schema.observation_columns + schema.static_columns
            ):
                raise ValueError(
                    "policy inputs must list observations and static columns in order"
                )
            if interface.target_columns != [schema.action_column]:
                raise ValueError("policy targets must name its action column")
            if any(
                (
                    interface.categorical_hashing,
                    interface.static_conditioning,
                    interface.preconditioning_image_encoder,
                    interface.feature_layout,
                )
            ):
                raise ValueError("policy learning supports scalar inputs")
            policies[name] = CategoricalPolicyConfig(
                **schema.model_dump(),
                interface_name=name,
                context_length=context,
                backbone=model.backbone.architecture,
                backbone_component=model.backbone,
                ingestion=interface.ingestion,
                decoder=interface.decoder,
            )
        values["policies"] = policies
        phase = PolicyTrainingPlanModel.model_validate(
            values.get("training_plan")
        ).phases[0]
        sources = {}
        for name, dataset in values.pop("dataset_training", {}).items():
            if set(dataset) - {"model_interface", "parts"} or not dataset.get("parts"):
                raise ValueError(
                    "policy datasets require model_interface and nonempty source parts"
                )
            for part, source in dataset["parts"].items():
                reference = f"{name}.{part}"
                if (
                    reference
                    in phase.trajectory_sources + (phase.validation_sources or [])
                    and dataset.get("model_interface") != phase.policy
                ):
                    raise ValueError(
                        "dataset model_interface disagrees with phase policy"
                    )
                sources[reference] = source
        values["trajectory_sources"] = sources
        return values


class ResolvedPolicyTrainConfig(_PolicyTrainConfigBase):
    model_path: NonemptyStr
    report_path: NonemptyStr


def resolve_policy_train_config(config: PolicyTrainConfig) -> ResolvedPolicyTrainConfig:
    root = Path(config.project_root).resolve()

    def absolute(path):
        return str((root / path).resolve())

    values = config.model_dump(mode="json")
    values["project_root"] = str(root)
    values["trajectory_sources"] = {
        name: resolve_sequence_source(source, root).model_dump()
        for name, source in config.trajectory_sources.items()
    }
    if config.initialize_from is not None:
        values["initialize_from"] = absolute(config.initialize_from)
    values["model_path"] = absolute(
        config.model_path or f"models/{config.model_name}/{config.model_name}.pt"
    )
    values["report_path"] = absolute(
        config.report_path or f"outputs/training/{config.model_name}/report.json"
    )
    if values["model_path"] == values["report_path"]:
        raise ValueError("model_path and report_path must differ")
    return ResolvedPolicyTrainConfig.model_validate(values)
