"""Canonical named training configuration and resolution.

All cardinalities use named interfaces, datasets, parts and ordered phases.
Only the current ownership structure is accepted.
"""

from __future__ import annotations

import copy
import inspect
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar, Literal, Optional

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from sequifier.config.components import BackboneComponentConfig
from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.image_config import MediaPath, bind_image_paths
from sequifier.config.learning_config import SupervisedLearningConfig
from sequifier.config.metadata import (
    DatasetMetadata,
    FeatureMetadata,
    load_dataset_metadata,
    validate_authored_metadata,
)
from sequifier.config.model_interface import (
    ModelInterfaceSpecModel,
    ModelSpecModel,
    ResolvedInterfaceStorage,
    ResolvedModelInterface,
)
from sequifier.config.naming import ConfigModel
from sequifier.config.normalization_config import FrozenNormalizationConfig
from sequifier.config.ownership import (
    DataLoadingSettings,
    InputContract,
    ModelExecutionSettings,
    OutputSettings,
    RunSettings,
    SplitSettings,
)
from sequifier.config.parameter_selection import TrainabilityConfig
from sequifier.config.phase_controls import (
    CurriculumConfig,
    EpochBudget,
    EvaluationTriggers,
    PhaseControls,
)
from sequifier.config.policy_config import PolicyBinding, ResolvedPolicyBinding
from sequifier.config.policy_optimization import SupervisedOptimizationConfig
from sequifier.config.policy_plan import (
    GRPOPhaseConfig,
    OnlineEnvironmentConfig,
    PolicyImitationPhaseConfig,
    PolicyPhase,
    PolicyPhaseVariant,
    PPOPhaseConfig,
)
from sequifier.config.policy_train_config import (
    PolicyDataSettings,
    SequenceSourceConfig,
)
from sequifier.config.run_capabilities import RunMode
from sequifier.config.sources import SourceReference, TrainingSourceSpecModel

# Reusable schemas are independent of the canonical root configuration.
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    derive_target_column_types,
    normalize_path,
    resolve_window_view,
)
from sequifier.io.window_placement import PredictionAlignment
from sequifier.model.embedding import validate_embedding_layer_names
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_options import ModelExecutionOptions
from sequifier.objectives import BERTObjective, get_objective_class
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, resolve_categorical_decoder_ids
from sequifier.typechecking import beartype


@beartype
def _validate_constructor_arguments(
    component: str,
    name: str,
    constructor: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Validate configured arguments without constructing the component."""

    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        raise ValueError(f"Invalid {component} {name} arguments: {error}") from error


class SourceMedia(ConfigModel):
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None


class DatasetPartSpecModel(SourceMedia):
    file_order: Literal["shuffled", "name"] = "shuffled"
    type: Literal["prepared"]
    metadata_config_path: str = Field(min_length=1)
    read_format: Literal["csv", "parquet", "pt"] = "parquet"


class IndexedSource(SourceMedia):
    type: Literal["indexed_sequence"]
    path: str = Field(min_length=1)


class SequencePartSpecModel(SequenceSourceConfig):
    type: Literal["sequence"]


class PreferencePartSpecModel(ConfigModel):
    type: Literal["preferences"]
    manifest_path: str = Field(min_length=1)


DatasetPart = Annotated[
    DatasetPartSpecModel
    | IndexedSource
    | SequencePartSpecModel
    | PreferencePartSpecModel,
    Field(discriminator="type"),
]


class EvaluationMonitorSpecModel(SourceReference):
    metric: Literal["loss"] = "loss"
    direction: Literal["min", "max"] = "min"


class EvaluationSpecModel(EvaluationTriggers):
    model_config = ConfigDict(extra="forbid")

    batch_size: int | None = Field(default=None, gt=0)
    sources: list[TrainingSourceSpecModel] = Field(..., min_length=1)
    monitor: Optional[EvaluationMonitorSpecModel] = None

    @model_validator(mode="after")
    @beartype
    def validate_monitor(self):
        if any(
            source.selection_weight is not None
            or source.batches_per_selection is not None
            for source in self.sources
        ):
            raise ValueError(
                "Evaluation sources cannot configure selection_weight or batches_per_selection"
            )
        sources = [source.source for source in self.sources]
        if len(sources) != len(set(sources)):
            raise ValueError("Evaluation sources must be unique")
        if self.monitor is not None and self.monitor.source not in sources:
            raise ValueError("evaluation.monitor.source must be an evaluation source")
        if len(sources) == 1 and self.monitor is None:
            self.monitor = EvaluationMonitorSpecModel(source=sources[0])
        return self


class SupervisedPhaseSettings(PhaseControls):
    algorithm: Literal["supervised"] = "supervised"
    name: SkipJsonSchema[str]
    budget: EpochBudget
    mode: Literal["sequential", "interleaved"] = "sequential"
    evaluation: EvaluationSpecModel | None = None
    learning: SupervisedLearningConfig = Field(default_factory=SupervisedLearningConfig)
    trainability: TrainabilityConfig = Field(default_factory=TrainabilityConfig)
    curriculum: CurriculumConfig | None = None
    optimizer_state: Literal["reset", "continue"] = "reset"
    batch_size: int = Field(default=32, gt=0)
    accumulation_batches: int = Field(default=1, gt=0)


class TrainingPhaseSpecModel(SupervisedPhaseSettings):
    optimization_roles: ClassVar[tuple[str, ...]] = ("model",)
    optimization: dict[Literal["model"], SupervisedOptimizationConfig] = Field(
        default_factory=lambda: {"model": SupervisedOptimizationConfig()}
    )

    def optimization_configs(self):
        return {"model": self.optimization.get("model", SupervisedOptimizationConfig())}

    selection: Optional[Literal["round_robin", "weighted_random"]] = None
    sources: list[TrainingSourceSpecModel] = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    @beartype
    def validate_name(cls, value):
        return _identifier(value, "Training phase name")

    @model_validator(mode="after")
    @beartype
    def validate_mode_fields(self):
        datasets = {ref.dataset for ref in self.sources}
        self.trainability.validate_roles(self.optimization_roles, datasets=datasets)
        learning_datasets = datasets | {
            ref.dataset for ref in (self.evaluation.sources if self.evaluation else [])
        }
        if unknown := self.learning.datasets.keys() - learning_datasets:
            raise ValueError(
                f"learning declarations require training/evaluation datasets: {sorted(unknown)}"
            )
        if self.mode == "sequential":
            if self.selection is not None:
                raise ValueError("Sequential phases cannot configure selection")
            if any(source.batches_per_selection is not None for source in self.sources):
                raise ValueError(
                    "batches_per_selection is invalid in sequential phases"
                )
            if any(source.selection_weight is not None for source in self.sources):
                raise ValueError("selection_weight is invalid in sequential phases")
        else:
            self.selection = self.selection or "round_robin"
            if self.selection == "round_robin" and any(
                source.selection_weight is not None for source in self.sources
            ):
                raise ValueError(
                    "selection_weight is only valid for weighted_random selection"
                )
        return self


TrainingPhase = Annotated[
    TrainingPhaseSpecModel | PolicyPhaseVariant, Field(discriminator="algorithm")
]


def validate_optimizer_transitions(phases):
    from sequifier.runtime.plan import validate_continuation

    for index, phase in enumerate(phases):
        if phase.optimizer_state != "continue":
            continue
        previous = phases[index - 1] if index else None
        validate_continuation(previous, phase)
        assert previous is not None
        if phase.algorithm == "supervised" and (
            {ref.dataset for ref in previous.data_sources.training}
            != {ref.dataset for ref in phase.data_sources.training}
        ):
            raise ValueError(
                "supervised optimizer continuation requires identical dataset ownership"
            )
        prior_roles = previous.optimization_configs()
        for role, current in phase.optimization_configs().items():
            prior = prior_roles[role]
            compare_budget = (
                prior.has_update_budget_override and current.has_update_budget_override
            )
            if prior.compatibility_settings(
                compare_budget=compare_budget
            ) != current.compatibility_settings(compare_budget=compare_budget):
                raise ValueError(
                    "optimizer continuation requires identical role optimization settings"
                )
        if phase.algorithm != "supervised" and any(
            item.scheduler.name == "OneCycleLR"
            for item in phase.optimization_configs().values()
        ):
            raise ValueError(
                "policy OneCycleLR continuation is unsupported; use optimizer_state: reset"
            )


def validate_training_contract(run, model, phases, settings: ModelExecutionOptions):
    """Check model and phase requirements without reading training data."""
    for phase in phases:
        evaluation = phase.evaluation
        if phase.algorithm != "supervised":
            if any(
                ref.selection_weight is not None
                or ref.batches_per_selection is not None
                for ref in phase.data_sources
            ):
                raise ValueError(
                    "policy source references cannot configure supervised interleaving"
                )
            continue
        role = (
            phase.optimization_configs()["model"]
            if isinstance(phase, TrainingPhaseSpecModel)
            else phase.optimization["model"]
        )
        needs_monitor = (
            phase.handoff == "best"
            or role.scheduler.name == "ReduceLROnPlateau"
            or phase.stopping is not None
            or (evaluation is not None and evaluation.on_snapshot)
        )
        if needs_monitor and (evaluation is None or evaluation.monitor is None):
            raise ValueError(
                f"{phase.name}: best handoff, plateau scheduling, validation saving, and early stopping require evaluation.monitor"
            )
        if (
            role.scheduler.name == "ReduceLROnPlateau"
            and phase.evaluation_interval_epochs != 1
        ):
            raise ValueError("ReduceLROnPlateau requires evaluation_interval_epochs: 1")

    supervised = [phase for phase in phases if phase.algorithm == "supervised"]
    if not supervised:
        return
    for interface in model.interfaces.values():
        validate_embedding_layer_names(
            run.export.embedding_layer_names,
            SimpleNamespace(
                backbone=model.backbone,
                decoder=interface.decoder,
            ),
        )

    objective = get_objective_class(settings.training_objective)
    for phase in supervised:
        learning = phase.learning
        stage_objective = get_objective_class(
            learning.objective_name(settings.training_objective)
        )
        if (stage_objective.forward_looking, stage_objective.uses_causal_attention) != (
            objective.forward_looking,
            objective.uses_causal_attention,
        ):
            raise ValueError(
                f"{phase.name}: the objective requires different prediction geometry from the shared model"
            )
        if issubclass(stage_objective, BERTObjective) and learning.bert_spec is None:
            raise ValueError(
                f"{phase.name}: BERT training requires learning.objective masking settings"
            )

    context_length = settings.context_length
    prefix_positions = int(
        any(
            route.static_conditioning is not None
            or route.preconditioning_image_encoder is not None
            for route in model.interfaces.values()
        )
    )
    if (
        context_length + prefix_positions
        > model.backbone.architecture.max_context_length
    ):
        raise ValueError(
            "context_length plus static prefix exceeds backbone max_context_length"
        )
    for name, interface in model.interfaces.items():
        if interface.decoder.support_length > context_length:
            raise ValueError(
                f"Interface {name!r} decoder support exceeds context_length"
            )
        decoded_length = context_length - interface.decoder.support_length + 1
        if interface.decoder.prediction_length > decoded_length:
            raise ValueError(
                f"Interface {name!r} prediction_length exceeds decoded length"
            )
        objective.validate_prediction_length(
            interface.decoder.prediction_length,
            context_length,
            usage="training",
        )


class ResolvedDatasetPart(ConfigModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    metadata_config_path: str
    metadata: DatasetMetadata
    metadata_from_file: bool
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None
    view_path: str
    split_paths: dict[Literal["training", "validation"], str]
    read_format: Literal["csv", "parquet", "pt"] = "parquet"
    storage_form: Literal["file", "folder", "indexed"]
    file_order: Literal["shuffled", "name"] = "shuffled"


class ResolvedDatasetTrainingSpec(ConfigModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    model_interface: str
    storage: ResolvedInterfaceStorage | None = None
    parts: dict[str, ResolvedDatasetPart]
    views: dict[str, tuple[str, ...]]


class ResolvedTrainingSource(SourceReference):
    parts: tuple[str, ...]
    selection_weight: float = 1.0
    batches_per_selection: int = 1


class ResolvedTrainingPhase(SupervisedPhaseSettings):
    name: str
    resolved_seed: int | None = None
    trainability_summary: dict[str, list[str]] = Field(default_factory=dict)

    evaluation_sources: list[ResolvedTrainingSource] = Field(default_factory=list)
    evaluation_monitor: EvaluationMonitorSpecModel | None = None

    optimization: dict[str, SupervisedOptimizationConfig]

    selection: Literal["round_robin", "weighted_random"] = "round_robin"
    sources: list[ResolvedTrainingSource]

    def part_roles(
        self,
    ) -> Iterator[tuple[str, str, Literal["training", "validation"]]]:
        """Yield dataset, part and loader role selected by this phase."""
        for source in self.sources:
            for part in source.parts:
                yield source.dataset, part, "training"
        for source in self.evaluation_sources:
            for part in source.parts:
                yield source.dataset, part, "validation"


def resolve_phases(authored_phases, sources, *, seed: int):
    """Resolve role defaults and epoch budgets without accessing datasets."""
    phases = [
        ResolvedTrainingPhase(
            **{
                name: getattr(phase, name)
                for name in SupervisedPhaseSettings.model_fields
            },
            resolved_seed=seed + phase_index,
            evaluation_sources=[
                sources.supervised_source(ref) for ref in phase.evaluation.sources
            ]
            if phase.evaluation is not None
            else [],
            evaluation_monitor=phase.evaluation.monitor
            if phase.evaluation is not None
            else None,
            optimization={
                role: value.model_copy(deep=True)
                for role, value in phase.optimization_configs().items()
            },
            selection=phase.selection or "round_robin",
            sources=[
                sources.supervised_source(source, phase.accumulation_batches)
                for source in phase.sources
            ],
        )
        if phase.algorithm == "supervised"
        else phase
        for phase_index, phase in enumerate(authored_phases)
    ]
    # A continued epoch scheduler spans the entire contiguous ownership chain.
    from sequifier.runtime.plan import continuation_groups

    for _, group in continuation_groups(phases):
        if not group or group[0].algorithm != "supervised":
            continue
        role = group[0].optimization["model"]
        if role.scheduler.name == "OneCycleLR" and role.scheduler_step_on == "epoch":
            budget = sum(phase.budget.epochs for phase in group)
            for phase in group:
                phase.optimization["model"] = phase.optimization[
                    "model"
                ].resolve_budget(budget)
    return phases


class ResolvedSequifierConfig(ConfigModel):
    """Compiled run, model, source, phase, and publication records."""

    _prepared_inputs: Any = PrivateAttr(default=None)
    _policy_bindings: dict[str, ResolvedPolicyBinding] = PrivateAttr()

    normalization_by_interface: dict[str, FrozenNormalizationConfig] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    execution: ModelExecution
    run: RunSettings
    model_metadata_paths: dict[str, str] = Field(default_factory=dict)
    dataset_training: dict[str, ResolvedDatasetTrainingSpec]
    initialize_from: str | None = None
    actor_initialization: Literal["attach", "history_only"] = "attach"
    policy_data: PolicyDataSettings | None = None
    training_plan: list[ResolvedTrainingPhase | PolicyPhase]
    curriculum_training: bool = False
    curriculum_column: Optional[str] = None
    run_mode: RunMode = "supervised"

    def model_post_init(self, context: Any) -> None:
        from sequifier.rl.attachment import bind_policy

        self._policy_bindings = {
            name: bind_policy(self.execution, name)
            for name, interface in self.interfaces.items()
            if interface.policy is not None
        }

    def policy_binding(self, name: str | None) -> ResolvedPolicyBinding:
        if name is None or name not in self._policy_bindings:
            raise ValueError(f"Interface {name!r} requires a policy binding")
        return self._policy_bindings[name]

    @property
    def outputs(self) -> OutputSettings:
        return self.run.outputs

    @property
    def project_root(self) -> str:
        return self.run.project_root

    @property
    def model_name(self) -> str:
        return self.run.model_name

    @property
    def device(self) -> str:
        return self.run.execution.device

    @property
    def seed(self) -> int:
        seed = self.run.reproducibility.seed
        if not isinstance(seed, int):
            raise ValueError("Resolved training requires an integer seed")
        return seed

    @property
    def model(self) -> ModelSpecModel:
        return self.execution.model

    @property
    def interfaces(self) -> dict[str, ResolvedModelInterface]:
        return self.execution.interfaces

    @property
    def evaluation_sources(self) -> list[ResolvedTrainingSource]:
        """All supervised evaluation dependencies, for loader and metric setup."""
        return list(
            {
                source.source: source
                for phase in self.training_plan
                if phase.algorithm == "supervised"
                for source in phase.evaluation_sources
            }.values()
        )

    @property
    def embedding_layer_names(self) -> list[str]:
        return self.run.export.embedding_layer_names

    def validate_next_occurrence_phases(self):
        for phase in self.training_plan:
            if phase.algorithm != "supervised":
                continue
            next_config = phase.learning.next_occurrence_config
            if next_config is None:
                continue
            column = next_config.column_name
            names = {
                source.dataset for source in phase.sources + phase.evaluation_sources
            }
            for dataset_name in names:
                dataset = self.dataset_training[dataset_name]
                interface = self.interface(dataset.model_interface)
                if column not in interface.target_columns:
                    raise ValueError(
                        f"{phase.name}: learning.objective.column_name must be "
                        f"a target column for dataset {dataset_name!r}, got {column!r}"
                    )
                if interface.target_column_types.get(column) != "categorical":
                    raise ValueError(
                        f"{phase.name}: learning.objective.column_name must refer "
                        "to a categorical target column"
                    )
                if column not in interface.id_maps:
                    raise ValueError(
                        f"{phase.name}: learning.objective.column_name must have "
                        f"a preprocessing id_map, got {column!r}"
                    )
                missing = [
                    value
                    for value in next_config.target_values
                    if value not in interface.id_maps[column]
                ]
                if missing:
                    raise ValueError(
                        f"{phase.name}: learning.objective.target_values must "
                        f"match keys in id_maps[{column!r}], missing {missing!r}"
                    )

    @model_validator(mode="after")
    @beartype
    def validate_model_execution_plans(self):
        """Compile each route while validation errors still retain Pydantic context."""

        for dataset in self.dataset_training.values():
            if dataset.model_interface not in self.interfaces:
                raise ValueError(
                    f"Dataset {dataset.name!r} references unknown interface "
                    f"{dataset.model_interface!r}"
                )
            if self.interface(dataset.model_interface).depth_layouts and any(
                part.read_format != "pt" for part in dataset.parts.values()
            ):
                raise ValueError("Preprocessed depth inputs require read_format: pt")
        for phase in self.training_plan:
            if isinstance(phase, ResolvedTrainingPhase):
                for name, learning in phase.learning.datasets.items():
                    interface = self.interface(
                        self.dataset_training[name].model_interface
                    )
                    learning.validate_interface(interface, f"{phase.name}: {name}")
            else:
                binding = self.policy_binding(phase.model_interface)
                for environment in (
                    phase.training_environment,
                    phase.evaluation_environment,
                ):
                    if environment and environment.units != binding.units:
                        raise ValueError(
                            f"{phase.name}: environment units differ from policy units"
                        )
        self.validate_next_occurrence_phases()
        self.execution.validate()
        return self

    @property
    @beartype
    def dataset_count(self) -> int:
        return len(self.dataset_training)

    @property
    @beartype
    def interface_names(self) -> tuple[str, ...]:
        return self.execution.interface_names

    @beartype
    def dataset(self, name: Optional[str] = None) -> ResolvedDatasetTrainingSpec:
        if name is None:
            if len(self.dataset_training) != 1:
                raise ValueError("A dataset selection is required")
            return next(iter(self.dataset_training.values()))
        return self.dataset_training[name]

    @beartype
    def interface(self, name: Optional[str] = None) -> ResolvedModelInterface:
        return self.execution.interface(name)

    def dataset_interface(self, name: str) -> ResolvedModelInterface:
        dataset = self.dataset_training[name]
        interface = self.interface(dataset.model_interface)
        if not dataset.parts:
            return interface
        metadata = next(iter(dataset.parts.values())).metadata
        return interface.model_copy(
            update={
                "image_sources": {
                    view: metadata.image_sources[view]
                    for view in interface.image_sources
                },
                "preconditioning_image": metadata.preconditioning_image
                if interface.preconditioning_image_encoder is not None
                else None,
            }
        )

    def dataset_storage(self, name: str) -> ResolvedInterfaceStorage:
        dataset = self.dataset_training[name]
        return (
            dataset.storage or self.execution.interface_storage[dataset.model_interface]
        )


@dataclass(frozen=True)
class LoadedTrainConfig:
    config: Any
    resolved: ResolvedSequifierConfig
    metadata: dict[str, DatasetMetadata]


@beartype
def _layout_for_part_path(
    metadata: DatasetMetadata, path: str, project_root: str
) -> StoredWindowLayout:
    split_paths = [
        normalize_path(split_path, project_root) for split_path in metadata.split_paths
    ]
    if path in split_paths:
        return metadata.layout_for_split(split_paths.index(path))
    folder_metadata_path = os.path.join(path, "metadata.json")
    if os.path.isfile(folder_metadata_path):
        return load_dataset_metadata(folder_metadata_path).storage_layout
    if metadata.split_index is not None:
        return metadata.layout_for_split(metadata.split_index)
    return metadata.storage_layout


@beartype
def _part_features(
    metadata: DatasetMetadata,
    interface: ModelInterfaceSpecModel,
    *,
    require_static_table: bool = True,
) -> FeatureMetadata:
    selection = interface.feature_selection
    features = selection.resolve(metadata)
    if (
        selection.static
        and require_static_table
        and metadata.static_features_path is None
    ):
        raise ValueError(f"Metadata lacks a static table for {selection.static}")

    def validate_frozen_size(encoder, source, usage):
        if encoder is None or encoder.type != "frozen":
            return
        expected_resize = (
            232
            if encoder.model == "resnet50" and encoder.weights == "IMAGENET1K_V2"
            else 256
        )
        if (source.resize_size, source.crop_size) != (expected_resize, 224):
            raise ValueError(
                f"{usage} requires resize_size={expected_resize}, crop_size=224 "
                f"for {encoder.model} {encoder.weights}"
            )

    def validate_image_ingestion(ingestion):
        if ingestion.type == "images":
            for view in ingestion.views:
                validate_frozen_size(
                    ingestion.encoder, metadata.image_sources[view], view
                )
        elif ingestion.type == "structured":
            for view in ingestion.image_cells.values():
                validate_frozen_size(
                    ingestion.image_encoder, metadata.image_sources[view], view
                )
        elif ingestion.type == "composite":
            for child in ingestion.branches.values():
                validate_image_ingestion(child)

    validate_image_ingestion(interface.ingestion)
    if interface.preconditioning_image_encoder is not None:
        validate_frozen_size(
            interface.preconditioning_image_encoder,
            metadata.preconditioning_image,
            "preconditioning image",
        )
    return features


@beartype
def _assert_compatible(
    expected: dict[str, Any], actual: dict[str, Any], usage: str
) -> None:
    mismatches = [key for key in expected if expected[key] != actual.get(key)]
    if mismatches:
        raise ValueError(f"{usage} has incompatible schema fields {mismatches}")


@beartype
def _resolve_interface(
    name: str,
    spec: ModelInterfaceSpecModel,
    metadata: DatasetMetadata,
    window_view: ModelWindowView,
) -> ResolvedModelInterface:
    features = _part_features(metadata, spec, require_static_table=False)
    target_types = derive_target_column_types(
        spec.target_columns, features.column_data_types
    )
    categorical_columns, real_columns = features.split_columns(spec.input_columns)
    static_categorical_columns, static_real_columns = features.split_columns(
        spec.static_conditioning.columns if spec.static_conditioning else []
    )
    categorical_targets = {
        column
        for column, type_name in target_types.items()
        if type_name == "categorical"
    }
    invalid_token_columns = (
        set(spec.categorical_decoder_special_tokens) - categorical_targets
    )
    if invalid_token_columns:
        raise ValueError(
            "categorical_decoder_special_tokens may only reference categorical "
            f"targets, found {sorted(invalid_token_columns)}"
        )
    categorical = (
        set(categorical_columns) | set(static_categorical_columns) | categorical_targets
    )
    if missing_classes := categorical - metadata.n_classes.keys():
        raise ValueError(
            f"Metadata lacks categorical vocabulary sizes: {sorted(missing_classes)}"
        )
    n_classes = features.n_classes
    categorical_decoder_special_tokens = {
        column: list(spec.categorical_decoder_special_tokens.get(column, ["other"]))
        for column in categorical_targets
    }
    target_decoder_ids = resolve_categorical_decoder_ids(
        spec.target_columns,
        target_types,
        n_classes,
        categorical_decoder_special_tokens,
    )
    hashing = dict(spec.categorical_hashing)
    allowed_hash_columns = set(categorical_columns) | categorical_targets
    if invalid := set(hashing) - allowed_hash_columns:
        raise ValueError(
            f"categorical_hashing requires categorical input or target columns: {sorted(invalid)}"
        )
    from sequifier.model.hash_codec import CategoricalHashCodec, HashCodeCollision

    hash_contracts = {}
    for column, hash_config in hashing.items():
        try:
            codec = CategoricalHashCodec(
                hash_config, n_classes[column], target_decoder_ids.get(column)
            )
        except ValueError as error:
            collision_ids = error.ids if isinstance(error, HashCodeCollision) else ()
            labels = {
                global_id: label
                for label, global_id in metadata.id_maps.get(column, {}).items()
                if global_id in collision_ids
            }
            raise ValueError(
                f"categorical_hashing[{column!r}]: {error} (category labels: {labels})"
            ) from error
        hash_contracts[column] = codec.contract()
    generated_metric_names = {
        f"@hash/{column}/{index}"
        for column in categorical_targets & set(hashing)
        for index in range(hashing[column].num_hashes)
    }
    if collision := generated_metric_names & set(spec.target_columns):
        raise ValueError(
            f"Target names conflict with hash component metrics: {sorted(collision)}"
        )
    return ResolvedModelInterface(
        name=name,
        policy=None,
        **{
            key: getattr(spec, key)
            for key in ModelInterfaceSpecModel.model_fields
            if key != "categorical_decoder_special_tokens"
        },
        **features.feature_values(),
        static_categorical_columns=static_categorical_columns,
        static_real_columns=static_real_columns,
        categorical_hash_contracts=hash_contracts,
        target_column_types=target_types,
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        categorical_decoder_special_tokens=categorical_decoder_special_tokens,
        target_decoder_ids=target_decoder_ids,
        window_view=window_view,
    )


@beartype
def resolve_supervised_datasets(
    config,
    sources,
    phases,
    dataset_names,
    view: ModelWindowView,
    declarations,
    *,
    metadata_from_file: bool,
):
    """Resolve prepared dataset parts and their model compatibility contracts."""
    from sequifier.io.indexed_sequences import prepared_split_form

    roles_by_part = {}
    for phase in phases:
        if isinstance(phase, ResolvedTrainingPhase):
            for name, part, role in phase.part_roles():
                roles_by_part.setdefault((name, part), set()).add(role)
    resolved_datasets: dict[str, ResolvedDatasetTrainingSpec] = {}
    inputs_by_interface = {}

    for dataset_name, dataset_spec in config.datasets.items():
        if dataset_name not in dataset_names:
            continue
        if dataset_spec.model_interface is None:
            raise ValueError(f"Dataset {dataset_name!r} requires a model interface")
        interface_name = dataset_spec.model_interface
        interface_spec = config.model.interfaces[interface_name]
        resolved_parts = {}
        declared = declarations.get(interface_name)
        expected_signature = declared.window_contract() if declared else None
        expected_form = None
        first_metadata = None
        first_layout: StoredWindowLayout | None = None

        # Select the representative storage layout from training before ancillary views.
        def role_order(item):
            roles = roles_by_part.get((dataset_name, item[0]), ())
            return 0 if "training" in roles else 1 if roles else 2

        for part_name, part_spec in sorted(
            sources.view_parts[dataset_name].items(), key=role_order
        ):
            ref = f"{dataset_name}.{part_name}"
            if not isinstance(part_spec, (DatasetPartSpecModel, IndexedSource)):
                if (dataset_name, part_name) not in roles_by_part:
                    continue
                raise ValueError(f"{ref}: supervised sources require prepared parts")
            record = sources.parts[ref]
            part_metadata = record.metadata
            if part_metadata is None:
                raise ValueError(f"Missing metadata for part: {ref}")
            roles = roles_by_part.get((dataset_name, part_name), ())
            path = sources.split_path(ref, required=bool(roles))
            if path is None:
                if roles:
                    raise ValueError(f"{ref}: missing selected split")
                continue
            if first_metadata is None:
                first_metadata = part_metadata
            if record.indexed is None:
                signature = part_metadata.window_contract()
                if expected_signature is None:
                    expected_signature = signature
                else:
                    _assert_compatible(expected_signature, signature, ref)

            paths = dict.fromkeys(sorted(roles), path)
            if path is None or record.metadata_path is None:
                raise ValueError(f"{ref}: expected metadata and a split path")
            form = (
                "indexed"
                if record.indexed is not None
                else prepared_split_form(Path(path))
            )
            if expected_form is None:
                expected_form = form
            elif expected_form != form:
                raise ValueError(
                    f"Dataset {dataset_name!r} parts must share file/folder storage form"
                )
            layout = (
                sources.indexed_view(Path(path).resolve()).layout(view)
                if form == "indexed"
                else _layout_for_part_path(part_metadata, path, config.run.project_root)
            )
            resolve_window_view(layout, view)
            if first_layout is None:
                first_layout = layout
            resolved_parts[part_name] = ResolvedDatasetPart(
                name=part_name,
                read_format=part_spec.read_format
                if isinstance(part_spec, DatasetPartSpecModel) and form != "indexed"
                else "pt",
                metadata_config_path=str(record.metadata_path),
                metadata=part_metadata,
                metadata_from_file=metadata_from_file,
                image_paths=part_spec.image_paths,
                preconditioning_image_paths=part_spec.preconditioning_image_paths,
                view_path=path,
                split_paths=paths,
                storage_form=form,
                file_order=part_spec.file_order
                if isinstance(part_spec, DatasetPartSpecModel)
                else "shuffled",
            )

        if not resolved_parts:
            continue
        assert first_metadata is not None
        assert first_layout is not None
        storage = ResolvedInterfaceStorage.from_metadata(first_metadata, first_layout)
        for part_name, resolved_part in resolved_parts.items():
            ref = f"{dataset_name}.{part_name}"
            aligned_metadata = resolved_part.metadata
            alignment = PredictionAlignment.from_settings(aligned_metadata)
            if alignment is None:
                continue
            for index in alignment.splits:
                if resolved_part.storage_form != "indexed":
                    alignment.validate_width(
                        aligned_metadata.width_for_split(index),
                        view,
                        f"{ref} split {index}",
                    )
                path = normalize_path(
                    aligned_metadata.split_paths[index], config.run.project_root
                )
                if path in resolved_part.split_paths.values():
                    alignment.validate_decoder(
                        view, interface_spec.decoder.prediction_length, ref
                    )
        prior_metadata, prior_storage = inputs_by_interface.setdefault(
            interface_name, (first_metadata, storage)
        )
        if prior_storage.storage_layout != first_layout:
            raise ValueError(
                f"Datasets sharing interface {interface_name!r} require compatible storage layouts"
            )
        if (
            not first_metadata.normalize_real_columns
            and prior_metadata is not first_metadata
        ):
            # Supervised datasets sharing a decoder also share its stored output statistics.
            _assert_compatible(
                _part_features(prior_metadata, interface_spec).signature(),
                _part_features(first_metadata, interface_spec).signature(),
                interface_name,
            )

        resolved_datasets[dataset_name] = ResolvedDatasetTrainingSpec(
            name=dataset_name,
            model_interface=interface_name,
            storage=storage,
            parts=resolved_parts,
            views={
                name: sources.members[f"{dataset_name}.{name}"]
                for name in dataset_spec.views
                if set(sources.members[f"{dataset_name}.{name}"])
                <= resolved_parts.keys()
            },
        )

    for phase in phases:
        if isinstance(phase, ResolvedTrainingPhase):
            training_paths = set()
            evaluation_paths = set()
            for dataset, part, role in phase.part_roles():
                path = Path(resolved_datasets[dataset].parts[part].view_path).resolve()
                (training_paths if role == "training" else evaluation_paths).add(path)
            if training_paths & evaluation_paths:
                raise ValueError(
                    f"{phase.name}: training and evaluation select the same physical data"
                )
    return resolved_datasets, inputs_by_interface


def resolve_sequifier_config(
    config: SequifierConfig,
    metadata: DatasetMetadata | dict[str, DatasetMetadata],
) -> ResolvedSequifierConfig:
    """Compile a training plan with caller-provided dataset metadata."""
    from sequifier.config.training_resolution import resolve_training_run

    if isinstance(metadata, DatasetMetadata):
        parts = [
            f"{name}.{part}"
            for name, dataset in config.datasets.items()
            for part in dataset.sources
        ]
        if len(parts) != 1:
            raise ValueError(
                "Metadata must be keyed by dataset.part for multiple parts"
            )
        metadata = {parts[0]: metadata}
    return resolve_training_run(config, metadata_by_part=metadata)


_INLINE_METADATA_KEYS = {
    "depth_layouts",
    "image_sources",
    "preconditioning_image",
    "tensor_payload_kind",
    "metadata_by_part",
    "column_data_types",
    "static_columns",
    "static_features_path",
    "n_classes",
    "id_maps",
    "special_token_ids",
    "selected_columns_statistics",
    "normalize_real_columns",
    "window_length",
    "window_lengths",
    "max_target_offset",
    "storage_layout",
    "split_paths",
    "split_names",
    "prediction_aligned_splits",
    "prediction_alignment_steps",
    "prediction_length",
    "target_offset",
}


@beartype
def _inline_metadata(
    values: dict[str, Any], global_spec: ModelExecutionSettings
) -> DatasetMetadata:
    validate_authored_metadata(values)
    layout = values.get("storage_layout")
    if isinstance(layout, StoredWindowLayout):
        window_length = layout.window_length
        max_target_offset = layout.max_target_offset
    elif isinstance(layout, dict):
        window_length = layout.get("window_length")
        max_target_offset = layout.get("max_target_offset", 1)
    else:
        window_length = values.get("window_length")
        max_target_offset = values.get("max_target_offset", 1)
    if window_length is None:
        if global_spec.context_length is None:
            raise ValueError("inline metadata requires context_length or window_length")
        window_length = global_spec.context_length + max(1, global_spec.target_offset)
    return DatasetMetadata.model_validate(
        {
            "depth_layouts": values.get("depth_layouts", {}),
            **(
                {"tensor_payload_kind": values["tensor_payload_kind"]}
                if "tensor_payload_kind" in values
                else {}
            ),
            "split_paths": values.get("split_paths", []),
            "split_names": values.get("split_names", []),
            "column_data_types": values.get("column_data_types", {}),
            "static_columns": values.get("static_columns", []),
            "static_features_path": values.get("static_features_path"),
            "image_sources": values.get("image_sources", {}),
            "preconditioning_image": values.get("preconditioning_image"),
            "n_classes": values.get("n_classes", {}),
            "id_maps": values.get("id_maps", {}),
            "special_token_ids": values.get(
                "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
            ),
            "selected_columns_statistics": values.get(
                "selected_columns_statistics", {}
            ),
            "normalize_real_columns": values.get("normalize_real_columns", True),
            "prediction_aligned_splits": values.get("prediction_aligned_splits", []),
            "prediction_alignment_steps": values.get("prediction_alignment_steps", 1),
            "window_lengths": values.get("window_lengths"),
            "prediction_length": values.get("prediction_length"),
            "target_offset": values.get("target_offset"),
            "window_length": window_length,
            "max_target_offset": max_target_offset,
        }
    )


@beartype
def load_train_config_with_source(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> LoadedTrainConfig:
    from sequifier.config.composition import load_composed_yaml_config
    from sequifier.helpers import try_catch_excess_keys

    raw = load_composed_yaml_config(config_path)
    args = {
        key: value
        for key, value in args_config.items()
        if key != "skip_metadata" and value is not None
    }
    metadata_inline_values = {
        key: args.pop(key) for key in list(args) if key in _INLINE_METADATA_KEYS
    }
    from sequifier.config.command_ownership import apply_run_overrides

    raw, args = apply_run_overrides(raw, args)
    from sequifier.config.composition import merge_config_fragments

    overrides = {
        key: args.pop(key) for key in ("run", "model", "plan", "phases") if key in args
    }
    raw = merge_config_fragments((raw, overrides))
    if args:
        raise ValueError(f"Unsupported training CLI overrides: {sorted(args)}")

    authored = try_catch_excess_keys(config_path, SequifierConfig, raw)
    from sequifier.config.training_resolution import resolve_training_run

    metadata_by_part: dict[str, DatasetMetadata] = {}
    if skip_metadata:
        execution = (
            authored.model.execution if authored.model else ModelExecutionSettings()
        )
        parts = [
            f"{name}.{part}"
            for name, dataset in authored.datasets.items()
            for part in dataset.sources
        ]
        if "metadata_by_part" in metadata_inline_values:
            metadata_by_part = {
                ref: _inline_metadata(values, execution)
                for ref, values in metadata_inline_values["metadata_by_part"].items()
            }
        elif len(parts) == 1:
            metadata_by_part[parts[0]] = _inline_metadata(
                metadata_inline_values, execution
            )
        else:
            raise ValueError(
                "skip_metadata with multiple parts requires metadata_by_part"
            )

    resolved = resolve_training_run(
        authored,
        metadata_by_part=metadata_by_part if skip_metadata else None,
        loaded_metadata=metadata_by_part,
    )
    return LoadedTrainConfig(
        config=authored, resolved=resolved, metadata=metadata_by_part
    )


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> ResolvedSequifierConfig:
    return load_train_config_with_source(
        config_path, args_config, skip_metadata
    ).resolved


@dataclass(frozen=True)
class SelectedInterfaceConfig:
    """Project model-owned fields into objective and loader views."""

    interface: ResolvedModelInterface

    @property
    def training_objective(self):
        return self.interface.window_view.objective

    @property
    def static_columns(self):
        static = self.interface.static_conditioning
        return static.columns if static else []

    def __getattr__(self, name):
        if name in ResolvedModelInterface.model_fields:
            return getattr(object.__getattribute__(self, "interface"), name)
        raise AttributeError(name)


@dataclass(frozen=True)
class SelectedObjectiveConfig(SelectedInterfaceConfig):
    learning: SupervisedLearningConfig

    @property
    def training_objective(self):
        return self.learning.objective_name(self.interface.window_view.objective)


@dataclass(frozen=True)
class SelectedDatasetPartConfig(SelectedInterfaceConfig):
    """Explicit dataset-part configuration used by data loaders."""

    project_root: str
    device: str
    seed: int
    data_loading: DataLoadingSettings
    static_features_path: str | None
    tensor_payload_version: int
    storage_layout: StoredWindowLayout
    batch_size: int
    dataset_training: ResolvedDatasetTrainingSpec
    part: ResolvedDatasetPart
    curriculum_training: bool
    curriculum_column: Optional[str]
    epochs: int
    read_format: str
    window_stride: Optional[int]
    metadata_config_path: str


@beartype
def dataset_part_view(
    config: ResolvedSequifierConfig,
    dataset_name: str,
    part_name: str,
    *,
    batch_size: int,
    window_stride: int | None,
    curriculum_column: str | None = None,
) -> SelectedDatasetPartConfig:
    """Select one dataset part for data loading."""

    dataset = config.dataset_training[dataset_name]
    part = dataset.parts[part_name]
    interface = config.interface(dataset.model_interface)
    selected_sources = {
        name: part.metadata.image_sources[name] for name in interface.image_sources
    }
    selected_prefix = (
        part.metadata.preconditioning_image
        if interface.preconditioning_image_encoder is not None
        else None
    )
    if (
        part.storage_form != "indexed"
        or part.image_paths
        or part.preconditioning_image_paths
    ):
        selected_sources, selected_prefix = bind_image_paths(
            selected_sources,
            selected_prefix,
            part.image_paths,
            part.preconditioning_image_paths,
        )
    interface = interface.model_copy(
        update={
            "image_sources": selected_sources,
            "preconditioning_image": selected_prefix,
        }
    )

    return SelectedDatasetPartConfig(
        data_loading=config.run.execution.supervised_settings.data_loading,
        static_features_path=part.metadata.static_features_path,
        tensor_payload_version=part.metadata.tensor_payload_version,
        storage_layout=config.dataset_storage(dataset_name).storage_layout,
        batch_size=batch_size,
        interface=interface,
        project_root=config.project_root,
        device=config.device,
        seed=config.seed,
        dataset_training=dataset,
        part=part,
        curriculum_training=curriculum_column is not None,
        curriculum_column=curriculum_column,
        epochs=sum(
            phase.budget.epochs
            for phase in config.training_plan
            if isinstance(phase, (ResolvedTrainingPhase, PolicyImitationPhaseConfig))
        ),
        read_format=part.read_format,
        window_stride=window_stride,
        metadata_config_path=part.metadata_config_path,
    )


TrainModel = ResolvedSequifierConfig


class OwnedModelInterfaceSpec(ModelInterfaceSpecModel):
    input_contract: InputContract = Field(default_factory=InputContract)


class OwnedModelSpec(ConfigModel):
    backbone: BackboneComponentConfig | None = None
    execution: ModelExecutionSettings = Field(default_factory=ModelExecutionSettings)
    interfaces: dict[str, OwnedModelInterfaceSpec] = Field(default_factory=dict)

    @classmethod
    def from_execution(cls, execution, declared=None, *, settings=None):
        """Project model facts without discarding authored input assertions."""
        values = execution.model.model_dump()
        values["execution"] = settings or (
            declared.execution.model_dump(exclude_unset=True) if declared else {}
        )
        if declared:
            for name, interface in declared.interfaces.items():
                if name in values["interfaces"]:
                    values["interfaces"][name]["input_contract"] = (
                        interface.input_contract
                    )
        return cls.model_validate(values)

    def architecture_model(self):
        if self.backbone is None:
            raise ValueError("model backbone has not been resolved")
        return ModelSpecModel.model_validate(
            self.model_dump(
                exclude={
                    "execution": True,
                    "interfaces": {"__all__": {"input_contract"}},
                }
            )
        )


class DatasetView(ConfigModel):
    parts: list[str] | None = Field(default=None, min_length=1)
    split: str | None = None


class DatasetTrainingSpecModel(ConfigModel):
    model_interface: str | None = None
    sources: dict[str, DatasetPart] = Field(min_length=1)
    partition: SplitSettings | None = None
    views: dict[str, DatasetView] = Field(default_factory=dict)

    @property
    def uses_metadata_views(self):
        return not self.views and all(
            source.type == "prepared" for source in self.sources.values()
        )

    @model_validator(mode="after")
    def validate_views(self):
        for name in self.sources:
            _identifier(name, "Dataset source")
        if self.partition:
            kinds = {source.type for source in self.sources.values()}
            if kinds not in ({"sequence"}, {"preferences"}):
                raise ValueError(
                    "A generated partition requires raw sequences or preference manifests"
                )
        if not self.views:
            if self.partition:
                self.views = {
                    name: DatasetView(split=name) for name in ("train", "validation")
                }
            elif not self.uses_metadata_views:
                self.views = {name: DatasetView(parts=[name]) for name in self.sources}
        for name, view in self.views.items():
            _identifier(name, "Dataset view")
            parts = view.parts or list(self.sources)
            if set(parts) - self.sources.keys() or len(set(parts)) != len(parts):
                raise ValueError(f"{name}: view requires unique known source parts")
            for part in parts:
                source = self.sources[part]
                if self.partition and view.split not in {"train", "validation"}:
                    raise ValueError(
                        f"{name}: select train or validation from the partition"
                    )
                if not self.partition and (source.type == "prepared") != (
                    view.split is not None
                ):
                    raise ValueError(
                        f"{name}: a prepared source requires a metadata split; a whole source takes no split"
                    )
        return self


class SequifierConfig(ConfigModel):
    """Reusable phase and data definitions with an explicit execution order."""

    run: RunSettings
    model: OwnedModelSpec | None = None
    initialize_from: str | None = None
    actor_initialization: Literal["attach", "history_only"] = "attach"
    datasets: dict[str, DatasetTrainingSpecModel] = Field(default_factory=dict)
    environments: dict[str, OnlineEnvironmentConfig] = Field(default_factory=dict)
    policy_bindings: dict[str, PolicyBinding] = Field(default_factory=dict)
    phases: dict[str, TrainingPhase] = Field(min_length=1)
    plan: list[str] = Field(min_length=1)

    @field_validator("phases", mode="before")
    @classmethod
    def name_phases(cls, phases):
        if not isinstance(phases, dict):
            return phases
        named = {}
        for name, phase in phases.items():
            if isinstance(phase, PhaseControls):
                phase = phase.model_dump(exclude={"name"})
            if not isinstance(phase, dict) or "name" in phase:
                raise ValueError(f"phases.{name}: the mapping key names the phase")
            named[name] = {**phase, "name": name}
        return named

    @model_serializer(mode="wrap")
    def serialize_definitions(self, handler):
        result = handler(self)
        for phase in result.get("phases", {}).values():
            phase.pop("name", None)
        return result

    @property
    def ordered_phases(self):
        return [self.phases[name] for name in self.plan]

    @model_validator(mode="before")
    @classmethod
    def default_backbone_capacity(cls, values):
        if not isinstance(values, dict) or not isinstance(values.get("model"), dict):
            return values
        values = copy.deepcopy(values)
        model = values["model"]
        context = model.get("execution", {}).get("context_length")
        architecture = (model.get("backbone") or {}).get("architecture")
        if isinstance(architecture, dict) and context is not None:
            prefix = any(
                interface.get("static_conditioning")
                or interface.get("preconditioning_image_encoder")
                for interface in model.get("interfaces", {}).values()
            )
            architecture.setdefault("max_context_length", context + int(prefix))
        return values

    @model_validator(mode="after")
    def validate_contract(self):
        if len(set(self.plan)) != len(self.plan) or set(self.plan) - self.phases.keys():
            raise ValueError(
                "plan must select distinct, defined phase names in execution order"
            )
        from sequifier.config.run_capabilities import validate_runner_settings

        active = self.ordered_phases
        validate_optimizer_transitions(active)
        validate_runner_settings(
            self.run,
            supervised=any(p.algorithm == "supervised" for p in active),
            policy=any(p.algorithm != "supervised" for p in active),
        )
        if self.run.execution.offline is not None and not any(
            p.algorithm != "supervised" and p.source_kind != "none" for p in active
        ):
            raise ValueError("run.execution.offline requires an offline policy phase")
        for name in self.environments:
            _identifier(name, "Environment")
        for phase in self.phases.values():
            if isinstance(phase, (PPOPhaseConfig, GRPOPhaseConfig)):
                evaluation = phase.evaluation
                if evaluation is None:
                    raise ValueError(f"phases.{phase.name}: evaluation is required")
                for environment in (
                    phase.collection.environment,
                    evaluation.environment,
                ):
                    if (
                        not isinstance(environment, str)
                        or environment not in self.environments
                    ):
                        raise ValueError(
                            f"phases.{phase.name}: select a named environment"
                        )
            elif (
                phase.algorithm in {"policy_imitation", "iql", "dpo"}
                and not phase.data_sources.evaluation
            ):
                raise ValueError(
                    f"phases.{phase.name}: select explicit evaluation source views"
                )
        if self.actor_initialization == "history_only" and (
            not self.initialize_from
            or self.model is None
            or self.model.backbone is None
            or not self.model.interfaces
            or not self.policy_bindings
            or self.ordered_phases[0].algorithm == "supervised"
        ):
            raise ValueError(
                "actor_initialization: history_only requires initialize_from, "
                "a complete downstream model, policy_bindings and a policy first phase"
            )
        if (
            self.model is None or self.model.backbone is None
        ) and self.initialize_from is None:
            raise ValueError("a run requires a complete model or initialize_from")
        for name in self.datasets:
            _identifier(name, "Dataset name")
        for phase in self.phases.values():
            refs = phase.data_sources
            for ref in refs:
                ref.validate_location(self.datasets, f"phase {phase.name!r}")
            if phase.algorithm == "supervised":
                for name in {ref.dataset for ref in refs}:
                    if name not in phase.learning.datasets:
                        raise ValueError(
                            f"{phase.name}: declare learning.datasets.{name}"
                        )
                    interface_name = self.datasets[name].model_interface
                    if self.model and interface_name in self.model.interfaces:
                        phase.learning.datasets[name].validate_targets(
                            self.model.interfaces[interface_name].target_columns,
                            f"{phase.name}: learning.datasets.{name}",
                        )
        return self

    def trial_config_values(self):
        """Keep inherited execution fields unset when materializing a search trial."""
        return self.model_dump(mode="python", exclude_unset=True)


ResolvedSequifierConfig.model_rebuild()
