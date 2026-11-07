"""Authored and resolved trajectory training contracts for the train command."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from sequifier.config.policy_config import (
    CategoricalPolicyConfig,
    PolicyConfigBase,
    PolicyOptimizationConfig,
)
from sequifier.config.training_plan import (
    PolicyTrainingPlanModel,
    ReturnConditionedPhaseConfig,
)


class PolicyTableColumns(PolicyConfigBase):
    """Explicit row timing: current observation/action, then resulting reward."""

    sequence_id: str = "sequenceId"
    item_position: str = "itemPosition"
    observations: dict[str, str] = Field(default_factory=dict)
    static_inputs: dict[str, str] = Field(default_factory=dict)
    action: str | None = None
    reward: str | None = "reward"
    terminated: str | None = "terminated"
    truncated: str | None = "truncated"
    previous_action: str | None = None
    successors: dict[str, str] = Field(default_factory=dict)
    bootstrap_allowed: str | None = None
    group: str | None = None
    episode_id: str | None = None

    @model_validator(mode="after")
    def validate_columns(self):
        names = [
            value for value in self.model_dump().values() if isinstance(value, str)
        ]
        names += (
            list(self.observations.values())
            + list(self.static_inputs.values())
            + list(self.successors.values())
        )
        if any(not value for value in names):
            raise ValueError("table column names must be nonempty")
        return self


class SequenceSourceConfig(PolicyConfigBase):
    paths: list[str] = Field(default_factory=list)
    manifest: str | None = None
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
        if any(not path for path in self.paths) or self.manifest == "":
            raise ValueError("trajectory paths must be nonempty")
        if self.split_index is not None and self.metadata_config_path is None:
            raise ValueError("split_index requires metadata_config_path")
        if self.encoding == "model" and self.metadata_config_path is None:
            raise ValueError("model-encoded sources require metadata_config_path")
        if self.format == "prepared" and self.metadata_config_path is None:
            raise ValueError("prepared sources require metadata_config_path")
        return self


# Existing Python callers and authored trajectory_sources remain supported.
TrajectorySourceConfig = SequenceSourceConfig


PolicyTrainingPlanConfig = PolicyTrainingPlanModel


def resolve_sequence_source(
    source: SequenceSourceConfig, root: Path
) -> SequenceSourceConfig:
    values = source.model_dump()
    values["paths"] = [str((root / path).resolve()) for path in source.paths]
    for key in ("manifest", "metadata_config_path", "static_data_path"):
        if values[key] is not None:
            values[key] = str((root / values[key]).resolve())
    if (
        values["metadata_config_path"] is not None
        and values["static_data_path"] is None
    ):
        from sequifier.config.metadata import load_dataset_metadata

        metadata = load_dataset_metadata(values["metadata_config_path"])
        prepared = (
            source.format == "prepared"
            or source.split_index is not None
            or any(Path(path).suffix.lower() == ".pt" for path in source.paths)
        )
        if prepared and metadata.static_features_path:
            values["static_data_path"] = str(
                (root / metadata.static_features_path).resolve()
            )
    if source.split_index is not None:
        from sequifier.config.metadata import load_dataset_metadata

        metadata_path = values["metadata_config_path"]
        if not isinstance(metadata_path, str):
            raise ValueError("prepared split sources require metadata_config_path")
        metadata = load_dataset_metadata(metadata_path)
        if source.split_index >= len(metadata.split_paths):
            raise ValueError(
                "source split_index is outside the prepared metadata splits"
            )
        path = (root / metadata.split_paths[source.split_index]).resolve()
        values["paths"] = (
            sorted(
                str(item)
                for item in path.glob(
                    "*.pt" if path.is_dir() and any(path.glob("*.pt")) else "*.parquet"
                )
            )
            if path.is_dir()
            else [str(path)]
        )
        values["format"] = "prepared"
        values["split_index"] = None
    return SequenceSourceConfig.model_validate(values)


class _PolicyTrainConfigBase(PolicyConfigBase):
    project_root: str = "."
    model_name: str = Field(min_length=1)
    device: str = "cpu"
    initialize_from: str | None = None
    seed: int = Field(default=1010, ge=0)
    trajectory_sources: dict[str, SequenceSourceConfig] = Field(
        min_length=1,
        validation_alias=AliasChoices("data_sources", "trajectory_sources"),
    )
    policies: dict[str, CategoricalPolicyConfig] = Field(min_length=1)
    training_plan: PolicyTrainingPlanConfig
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

    @model_validator(mode="before")
    @classmethod
    def normalize_plan(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        source_interfaces = {}
        if "global_training" in values:
            if "optimization" in values:
                raise ValueError("configure global_training or optimization, not both")
            training = dict(values.pop("global_training"))
            context = training.pop("context_length", None)
            if context is not None:
                values["context_length"] = context
            if training.pop("training_objective", "policy") != "policy":
                raise ValueError("policy phases require training_objective: policy")
            values["optimization"] = training
        if "dataset_training" in values:
            if "data_sources" in values or "trajectory_sources" in values:
                raise ValueError("configure dataset_training or data_sources, not both")
            sources = {}
            for name, dataset in values.pop("dataset_training").items():
                if set(dataset) - {"model_interface", "parts"}:
                    raise ValueError(
                        "policy datasets configure model_interface and source parts; action NLL is supplied by the algorithm"
                    )
                if not dataset.get("parts"):
                    raise ValueError("policy datasets require nonempty source parts")
                for part, source in dataset["parts"].items():
                    sources[f"{name}.{part}"] = source
                    source_interfaces[f"{name}.{part}"] = dataset.get("model_interface")
            values["data_sources"] = sources
        # Canonical model authoring is shared with supervised learning. The
        # compact policies registry remains a compatibility/convenience surface.
        if "model" in values:
            if "policies" in values:
                raise ValueError("configure model or policies, not both")
            from sequifier.config.train_config import ModelSpecModel

            model = ModelSpecModel.model_validate(values.pop("model"))
            policies = {}
            for name, interface in model.interfaces.items():
                if interface.policy is None:
                    raise ValueError(
                        "policy training model interfaces require a policy schema"
                    )
                schema = interface.policy.model_dump()
                expected = schema["observation_columns"] + schema["static_columns"]
                if interface.input_columns and interface.input_columns != expected:
                    raise ValueError(
                        "policy interface inputs must name its observation and static columns in order"
                    )
                target = schema["action_column"]
                if interface.target_columns != [target]:
                    raise ValueError(
                        "policy interface targets must name its action column"
                    )
                if (
                    interface.categorical_hashing
                    or interface.static_conditioning
                    or interface.preconditioning_image_encoder
                    or interface.feature_layout
                ):
                    raise ValueError(
                        "this policy learner currently supports scalar policy inputs"
                    )
                policies[name] = {
                    **schema,
                    "interface_name": name,
                    "backbone": model.backbone.architecture.model_dump(),
                    "backbone_component": model.backbone.model_dump(),
                    "context_length": values.get(
                        "context_length", model.backbone.architecture.max_context_length
                    ),
                    "ingestion": interface.ingestion.model_dump(),
                    "decoder": interface.decoder.model_dump(),
                }
            values.pop("context_length", None)
            values["policies"] = policies
        elif "context_length" in values:
            context = values.pop("context_length")
            values["policies"] = {
                name: {**policy, "context_length": context}
                for name, policy in values.get("policies", {}).items()
            }
        plan = values.get("training_plan")
        if plan is not None and isinstance(plan, BaseModel):
            plan = plan.model_dump()
            values["training_plan"] = plan
        if isinstance(plan, dict) and "phases" not in plan:
            values["training_plan"] = {"phases": [plan]}
        plan = values.get("training_plan", {})
        if not isinstance(plan, dict):
            plan = {}
        normalized_phases = []
        for phase in plan.get("phases", []):
            if isinstance(phase, BaseModel):
                phase = phase.model_dump()
            if not isinstance(phase, dict):
                normalized_phases.append(phase)
                continue
            phase = dict(phase)
            if "policy" not in phase and len(values.get("policies", {})) == 1:
                phase["policy"] = next(iter(values["policies"]))
            for reference in phase.get("sources", phase.get("trajectory_sources", [])):
                if isinstance(reference, BaseModel):
                    reference = reference.model_dump()
                name = (
                    reference
                    if isinstance(reference, str)
                    else reference.get("source")
                    if isinstance(reference, dict)
                    else None
                )
                if source_interfaces.get(name) is not None and source_interfaces[
                    name
                ] != phase.get("policy"):
                    raise ValueError(
                        "policy dataset model_interface disagrees with its phase policy"
                    )
            # Work on copied phases, never mutate the caller's authored mapping.
            normalized_phases.append(phase)
        if plan.get("phases"):
            values["training_plan"] = {**plan, "phases": normalized_phases}
        optimization = values.get("optimization", {})
        if isinstance(optimization, dict) and {"epochs", "seed"} & optimization.keys():
            raise ValueError("set epochs on the phase and seed on the run")
        return values

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

    @field_validator("actor_path", "report_path", check_fields=False)
    @classmethod
    def validate_output_path(cls, value):
        if value == "":
            raise ValueError("output paths must be nonempty")
        return value


class PolicyTrainConfig(_PolicyTrainConfigBase):
    """Authored paths may be omitted to select the standard output locations."""

    actor_path: str | None = Field(
        default=None, validation_alias=AliasChoices("model_path", "actor_path")
    )
    report_path: str | None = None

    @property
    def model_path(self) -> str | None:
        return self.actor_path


class ResolvedPolicyTrainConfig(_PolicyTrainConfigBase):
    """Run paths are absolute and the authored phase has been validated."""

    actor_path: str = Field(validation_alias=AliasChoices("model_path", "actor_path"))
    report_path: str

    @property
    def model_path(self) -> str:
        return self.actor_path


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
    values["actor_path"] = absolute(
        config.actor_path or f"models/{config.model_name}/{config.model_name}.pt"
    )
    values["report_path"] = absolute(
        config.report_path or f"outputs/training/{config.model_name}/report.json"
    )
    if values["actor_path"] == values["report_path"]:
        raise ValueError("actor_path and report_path must differ")
    return ResolvedPolicyTrainConfig.model_validate(values)
