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

from sequifier.config.components import (
    BackboneComponentConfig,
    DecoderComponentConfig,
    IngestionComponentConfig,
    ResumeConfig,
)
from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.image_config import MediaPath, bind_image_paths, image_views
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
from sequifier.config.ownership import ModelExecutionSettings
from sequifier.config.parameter_selection import TrainabilityConfig
from sequifier.config.phase_controls import (
    CurriculumConfig,
    EvaluationTriggers,
    PhaseControls,
)
from sequifier.config.policy_config import PolicyBinding
from sequifier.config.policy_optimization import SupervisedOptimizationConfig
from sequifier.config.policy_plan import (
    POLICY_PHASE_SCHEMAS,
    PolicyImitationPhaseConfig,
    PolicyPhase,
    PolicyPhaseVariant,
)
from sequifier.config.policy_train_config import (
    PolicyDataSettings,
    SequenceSourceConfig,
)
from sequifier.config.run_capabilities import PublicationPlan
from sequifier.config.sources import SourceReference, TrainingSourceSpecModel

# Reusable schemas are independent of the canonical root configuration.
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    derive_target_column_types,
    normalize_path,
    resolve_window_view,
)
from sequifier.model.embedding import validate_embedding_layer_names
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_options import ExecutionOptions
from sequifier.objectives import (
    BERTObjective,
    NextOccurrenceObjective,
    get_objective_class,
    target_offset_for_objective,
)
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


class TrainingRuntimeSettings(ExecutionOptions):
    """Internal native execution projection; never an authored ownership surface."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    episode_cache_size: int = Field(default=8, gt=0)
    inference_batch_size: int = Field(default=1, gt=0)

    log_interval_batches: int = Field(default=10, gt=0)
    log_interval_iterations: int = Field(default=10, gt=0)
    latest_checkpoint_interval_minutes: Optional[float] = None
    resume: Optional[ResumeConfig] = None
    enforce_determinism: bool = False
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: float = Field(default=16, gt=0, allow_inf_nan=False)
    world_size: int = Field(default=1, gt=0)
    num_workers: int = Field(default=0, ge=0)
    backend: str = "nccl"
    layer_autocast: bool = False
    data_parallelism: Optional[Literal["ddp", "fsdp"]] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: Literal["outer", "inner", "none"] = "outer"
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"

    @model_validator(mode="after")
    @beartype
    def validate_distribution(self):
        if self.distributed and self.data_parallelism is None:
            raise ValueError("distributed=true requires data_parallelism")
        if self.data_parallelism != "fsdp" and self.fsdp_cpu_offload is not None:
            raise ValueError("fsdp_cpu_offload is only valid with FSDP")
        if self.data_parallelism == "fsdp":
            if self.fsdp_cpu_offload is None:
                raise ValueError("FSDP requires fsdp_cpu_offload")
            if self.layer_type_dtypes is not None:
                raise ValueError("FSDP does not support manual layer pre-casting")
            if self.torch_compile == "outer":
                raise ValueError("FSDP requires torch_compile 'none' or 'inner'")
        if self.data_parallelism == "ddp" and self.torch_compile == "inner":
            raise ValueError("DDP requires torch_compile 'none' or 'outer'")
        interval = self.latest_checkpoint_interval_minutes
        if (
            interval is not None
            and interval <= 0
            and os.getenv("SEQUIFIER_TESTING") != "1"
        ):
            raise ValueError(
                "latest_checkpoint_interval_minutes must be larger than zero"
            )
        return self


class DatasetPartSpecModel(ConfigModel):
    type: Literal["prepared", "indexed_sequence"] = "prepared"
    path: str | None = None
    read_format: Literal["csv", "parquet", "pt"] = "parquet"

    metadata_config_path: str | None = None
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None
    file_order: Literal["shuffled", "name"] = "shuffled"

    @model_validator(mode="after")
    def validate_location(self):
        if self.type == "indexed_sequence":
            if not self.path:
                raise ValueError("An indexed_sequence part requires path")
            if self.metadata_config_path is not None:
                raise ValueError(
                    "Indexed sequence parts resolve metadata from their manifest"
                )
        elif not self.metadata_config_path or self.path is not None:
            raise ValueError("A prepared part requires metadata_config_path")
        return self


class SequencePartSpecModel(SequenceSourceConfig):
    type: Literal["sequence"]


class PreferencePartSpecModel(ConfigModel):
    type: Literal["preferences"]
    manifest_path: str = Field(min_length=1)


DatasetPart = Annotated[
    DatasetPartSpecModel | SequencePartSpecModel | PreferencePartSpecModel,
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
    name: str
    epochs: int = Field(gt=0)
    mode: Literal["sequential", "interleaved"]
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

_SEQUENCE_ALGORITHMS = {"supervised"} | {
    name
    for name, schema in POLICY_PHASE_SCHEMAS.items()
    if schema.source_kind == "sequence"
}


class TrainingPlanModel(ConfigModel):
    phases: list[TrainingPhase] = Field(..., min_length=1)

    @model_validator(mode="after")
    @beartype
    def validate_unique_names(self):
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Training phase names must be unique")
        from sequifier.runtime.plan import validate_continuation

        for index, phase in enumerate(self.phases):
            if phase.optimizer_state != "continue":
                continue
            previous = self.phases[index - 1] if index else None
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
                    prior.has_update_budget_override
                    and current.has_update_budget_override
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
        return self


def _validate_next_occurrence_config(phase, objective):
    if (phase.learning.next_occurrence_config is not None) != issubclass(
        objective, NextOccurrenceObjective
    ):
        raise ValueError(
            f"{phase.name}: learning.next_occurrence_config must be set "
            "exactly for next_occurrence training"
        )


def validate_training_contract(run, model, phases, settings):
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
    if settings.training_objective is None or settings.context_length is None:
        raise ValueError(
            "supervised training requires training_objective and context_length"
        )
    for interface in model.interfaces.values():
        validate_embedding_layer_names(
            run.export.embedding_layer_names,
            SimpleNamespace(
                backbone=model.backbone,
                decoder=interface.decoder,
            ),
        )

    objective = get_objective_class(settings.training_objective)
    is_bert = issubclass(objective, BERTObjective)
    for phase in supervised:
        _validate_next_occurrence_config(phase, objective)
        if (phase.learning.bert_spec is not None) != is_bert:
            raise ValueError("bert_spec must be configured exactly for BERT training")

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


class ResolvedTrainingSource(SourceReference):
    parts: tuple[str, ...]
    selection_weight: float = 1.0
    batches_per_selection: int = 1


class ResolvedTrainingPhase(SupervisedPhaseSettings):
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
            budget = sum(phase.epochs for phase in group)
            for phase in group:
                phase.optimization["model"] = phase.optimization[
                    "model"
                ].resolve_budget(budget)
    return phases


class ResolvedSequifierConfig(ConfigModel):
    """Compiled run, model, source, phase, and publication records."""

    _prepared_inputs: Any = PrivateAttr(default=None)

    ownership: dict[str, Any] = Field(default_factory=dict)
    normalization_by_interface: dict[str, FrozenNormalizationConfig] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    execution: ModelExecution
    runtime_settings: TrainingRuntimeSettings
    model_metadata_paths: dict[str, str] = Field(default_factory=dict)
    dataset_training: dict[str, ResolvedDatasetTrainingSpec]
    authored: SequifierConfig | None = Field(default=None, exclude=True)
    initialize_from: str | None = None
    actor_initialization: Literal["attach", "history_only"] = "attach"
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""
    policy_data: PolicyDataSettings | None = None
    training_plan: list[ResolvedTrainingPhase | PolicyPhase]
    curriculum_training: bool = False
    curriculum_column: Optional[str] = None
    publication: PublicationPlan

    @property
    def project_root(self) -> str:
        return self.execution.project_root

    @property
    def model_name(self) -> str:
        return self.execution.model_name

    @property
    def device(self) -> str:
        return self.execution.device

    @property
    def seed(self) -> int:
        return self.execution.seed

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
        return self.publication.exports.embedding_layer_names

    def validate_next_occurrence_phases(self):
        objective = get_objective_class(self.runtime_settings.training_objective)
        for phase in self.training_plan:
            if phase.algorithm != "supervised":
                continue
            next_config = phase.learning.next_occurrence_config
            _validate_next_occurrence_config(phase, objective)
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
                        f"{phase.name}: next_occurrence_config.column_name must be "
                        f"a target column for dataset {dataset_name!r}, got {column!r}"
                    )
                if interface.target_column_types.get(column) != "categorical":
                    raise ValueError(
                        f"{phase.name}: next_occurrence_config.column_name must refer "
                        "to a categorical target column"
                    )
                if column not in interface.id_maps:
                    raise ValueError(
                        f"{phase.name}: next_occurrence_config.column_name must have "
                        f"a preprocessing id_map, got {column!r}"
                    )
                missing = [
                    value
                    for value in next_config.target_values
                    if value not in interface.id_maps[column]
                ]
                if missing:
                    raise ValueError(
                        f"{phase.name}: next_occurrence_config.target_values must "
                        f"match keys in id_maps[{column!r}], missing {missing!r}"
                    )

    @model_validator(mode="after")
    @beartype
    def validate_model_execution_plans(self):
        """Compile each route while validation errors still retain Pydantic context."""

        if not self.interfaces:
            raise ValueError("A resolved model requires at least one interface")
        for name, interface in self.interfaces.items():
            if name not in self.model.interfaces or name != interface.name:
                raise ValueError(f"Invalid resolved model interface {name!r}")
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
def _storage_form(path: str) -> Literal["file", "folder", "indexed"]:
    value = Path(path)
    if not value.exists():
        raise ValueError(f"Training data path {path!r} does not exist")
    from sequifier.io.indexed_sequences import is_indexed_view

    if is_indexed_view(value):
        return "indexed"
    return "folder" if value.is_dir() else "file"


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
    static_columns = (
        interface.static_conditioning.columns
        if interface.static_conditioning is not None
        else []
    )
    if missing_static := set(static_columns) - set(metadata.static_columns):
        raise ValueError(f"Metadata lacks static features: {sorted(missing_static)}")
    if (
        static_columns
        and require_static_table
        and metadata.static_features_path is None
    ):
        raise ValueError(f"Metadata lacks a static table for {static_columns}")
    if (
        interface.preconditioning_image_encoder is not None
        and metadata.preconditioning_image is None
    ):
        raise ValueError("Metadata lacks the configured preconditioning image source")

    selected_views = image_views(interface.ingestion)
    missing_views = selected_views - set(metadata.image_sources)
    if missing_views:
        raise ValueError(f"Metadata lacks image views: {sorted(missing_views)}")

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
    relevant = list(
        dict.fromkeys(
            interface.input_columns + interface.target_columns + static_columns
        )
    )
    if any(metadata.depth_layouts.is_deep_column(c) for c in interface.target_columns):
        raise ValueError("Depth targets are not supported")
    missing = set(relevant) - set(metadata.column_data_types)
    if missing:
        raise ValueError(f"Metadata is missing interface columns: {sorted(missing)}")
    features = metadata.select_features(
        relevant, selected_views, interface.preconditioning_image_encoder is not None
    )
    if unknown_types := {
        column
        for column, dtype in features.column_data_types.items()
        if not any(kind in dtype.lower() for kind in ("int", "float"))
    }:
        raise ValueError(
            f"Unsupported metadata dtypes for columns: {sorted(unknown_types)}"
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
    global_spec: TrainingRuntimeSettings,
) -> ResolvedModelInterface:
    if global_spec.training_objective is None or global_spec.context_length is None:
        raise ValueError(
            "resolved training requires training_objective and context_length"
        )
    features = _part_features(metadata, spec, require_static_table=False)
    target_types = derive_target_column_types(
        spec.target_columns, metadata.column_data_types
    )
    categorical_columns = [
        column
        for column in spec.input_columns
        if "int" in metadata.column_data_types[column].lower()
    ]
    real_columns = [
        column
        for column in spec.input_columns
        if "float" in metadata.column_data_types[column].lower()
    ]
    static_columns = (
        spec.static_conditioning.columns if spec.static_conditioning else []
    )
    static_categorical_columns = [
        column
        for column in static_columns
        if "int" in metadata.column_data_types[column].lower()
    ]
    static_real_columns = [
        column
        for column in static_columns
        if "float" in metadata.column_data_types[column].lower()
    ]
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
    features.n_classes = {column: features.n_classes[column] for column in categorical}
    features.id_maps = {
        column: mapping
        for column, mapping in features.id_maps.items()
        if column in categorical
    }
    features.selected_columns_statistics = {
        column: stats
        for column, stats in features.selected_columns_statistics.items()
        if column not in categorical
    }
    features.depth_layouts = features.depth_layouts.relevant_layouts(spec.input_columns)
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
    target_n_classes = {column: len(ids) for column, ids in target_decoder_ids.items()}
    target_global_to_decoder = {}
    for column, ids in target_decoder_ids.items():
        inverse = {global_id: decoder_id for decoder_id, global_id in enumerate(ids)}
        target_global_to_decoder[column] = [
            inverse.get(global_id, -1) for global_id in range(n_classes[column])
        ]
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
    target_offset = target_offset_for_objective(
        global_spec.training_objective, global_spec.target_offset
    )
    window_view = ModelWindowView(
        context_length=global_spec.context_length,
        objective=global_spec.training_objective,
        target_offset=target_offset,
    )
    return ResolvedModelInterface(
        name=name,
        policy=None,
        input_columns=spec.input_columns,
        prediction_distributions=spec.prediction_distributions,
        static_conditioning=spec.static_conditioning,
        **features.feature_values(),
        preconditioning_image_encoder=spec.preconditioning_image_encoder,
        static_categorical_columns=static_categorical_columns,
        static_real_columns=static_real_columns,
        target_columns=spec.target_columns,
        categorical_hashing=hashing,
        categorical_hash_contracts=hash_contracts,
        target_column_types=target_types,
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        categorical_decoder_special_tokens={
            column: list(tokens)
            for column, tokens in categorical_decoder_special_tokens.items()
        },
        feature_layout=spec.feature_layout,
        ingestion=spec.ingestion,
        decoder=spec.decoder,
        target_decoder_ids=target_decoder_ids,
        target_n_classes=target_n_classes,
        target_global_to_decoder=target_global_to_decoder,
        window_view=window_view,
    )


@beartype
def _interface_semantics(interface: ResolvedModelInterface) -> dict[str, Any]:
    architecture_fields = {"name", "ingestion", "decoder", "feature_layout"}
    return {
        **interface.model_dump(
            exclude=architecture_fields | set(FeatureMetadata.model_fields)
        ),
        **interface.signature(),
    }


@beartype
def resolve_supervised_datasets(
    config,
    sources,
    phases,
    dataset_names,
    settings,
    declarations,
    *,
    metadata_from_file: bool,
):
    """Resolve prepared dataset parts and their model compatibility contracts."""

    roles_by_part = {}
    for phase in phases:
        if isinstance(phase, ResolvedTrainingPhase):
            for name, part, role in phase.part_roles():
                roles_by_part.setdefault((name, part), set()).add(role)
    resolved_datasets: dict[str, ResolvedDatasetTrainingSpec] = {}
    interfaces = {}
    interface_storage = {}
    dataset_semantics: dict[str, dict[str, Any]] = {}

    for dataset_name, dataset_spec in config.dataset_training.items():
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
        view = ModelWindowView(
            context_length=settings.context_length,
            objective=settings.training_objective,
            target_offset=target_offset_for_objective(
                settings.training_objective,
                settings.target_offset,
            ),
        )
        first_layout: StoredWindowLayout | None = None
        for part_name, part_spec in dataset_spec.parts.items():
            ref = f"{dataset_name}.{part_name}"
            if not isinstance(part_spec, DatasetPartSpecModel):
                raise ValueError(f"{ref}: supervised sources require prepared parts")
            record = sources.parts[ref]
            part_metadata = record.metadata
            if part_metadata is None:
                raise ValueError(f"Missing metadata for part: {ref}")
            if first_metadata is None:
                first_metadata = part_metadata
            if record.indexed is None:
                signature = part_metadata.window_contract()
                if expected_signature is None:
                    expected_signature = signature
                else:
                    _assert_compatible(expected_signature, signature, ref)

            paths = {}
            roles = roles_by_part.get((dataset_name, part_name), ())
            for role in sorted(roles):
                path = sources.split_path(ref, role)
                if path is None:
                    raise ValueError(f"{ref}: missing {role} split path")
                paths[role] = path

            form = None
            layout_paths = (
                dict.fromkeys(paths.values())
                if paths
                else [sources.split_path(ref, "training")]
            )
            for path in layout_paths:
                if path is None or record.metadata_path is None:
                    raise ValueError(f"{ref}: expected metadata and a split path")
                form = "indexed" if record.indexed is not None else _storage_form(path)
                if expected_form is None:
                    expected_form = form
                elif expected_form != form:
                    raise ValueError(
                        f"Dataset {dataset_name!r} parts must share file/folder storage form"
                    )
                layout = (
                    sources.indexed_view(Path(path).resolve()).layout(view)
                    if form == "indexed"
                    else _layout_for_part_path(
                        part_metadata, path, config.run.project_root
                    )
                )
                resolve_window_view(layout, view)
                if first_layout is None:
                    first_layout = layout
            assert form is not None
            resolved_parts[part_name] = ResolvedDatasetPart(
                name=part_name,
                read_format="pt" if form == "indexed" else part_spec.read_format,
                metadata_config_path=str(record.metadata_path),
                metadata=part_metadata,
                metadata_from_file=metadata_from_file,
                image_paths=part_spec.image_paths,
                preconditioning_image_paths=part_spec.preconditioning_image_paths,
                split_paths=paths,
                storage_form=form,
                file_order=part_spec.file_order,
            )

        assert first_metadata is not None
        assert first_layout is not None
        interface = _resolve_interface(
            interface_name,
            interface_spec,
            first_metadata,
            settings,
        )
        storage = ResolvedInterfaceStorage.from_metadata(first_metadata, first_layout)
        for part_name, resolved_part in resolved_parts.items():
            ref = f"{dataset_name}.{part_name}"
            aligned_metadata = resolved_part.metadata
            if not aligned_metadata.prediction_aligned_splits:
                continue
            assert aligned_metadata.prediction_length is not None
            required_aligned_width = (
                interface.window_view.context_length
                + interface.window_view.target_offset
                + (aligned_metadata.prediction_alignment_steps - 1)
                * aligned_metadata.prediction_length
            )
            for index in aligned_metadata.prediction_aligned_splits:
                if (
                    resolved_part.storage_form != "indexed"
                    and aligned_metadata.width_for_split(index) < required_aligned_width
                ):
                    raise ValueError(
                        f"Prediction aligned split {index} for {ref!r} requires "
                        f"window width at least {required_aligned_width}."
                    )
            normalized_splits = [
                normalize_path(path, config.run.project_root)
                for path in aligned_metadata.split_paths
            ]
            aligned_paths = {
                normalized_splits[index]
                for index in aligned_metadata.prediction_aligned_splits
            }
            if aligned_paths.isdisjoint(resolved_part.split_paths.values()):
                continue
            if aligned_metadata.target_offset != interface.window_view.target_offset:
                raise ValueError(
                    f"Prediction alignment target_offset for {ref!r} is "
                    f"{aligned_metadata.target_offset}, but the selected interface "
                    f"uses {interface.window_view.target_offset}."
                )
            if (
                aligned_metadata.prediction_length
                != interface.decoder.prediction_length
            ):
                raise ValueError(
                    f"Prediction alignment prediction_length for {ref!r} is "
                    f"{aligned_metadata.prediction_length}, but the selected interface "
                    f"uses {interface.decoder.prediction_length}."
                )
        semantic_contract = {
            **_interface_semantics(interface),
            "storage_layout": first_layout,
        }
        prior_contract = dataset_semantics.get(interface_name)
        if prior_contract is not None:
            _assert_compatible(
                prior_contract,
                semantic_contract,
                f"Datasets sharing interface {interface_name!r}",
            )
        else:
            dataset_semantics[interface_name] = semantic_contract
        interfaces.setdefault(interface_name, interface)
        interface_storage.setdefault(interface_name, storage)

        resolved_datasets[dataset_name] = ResolvedDatasetTrainingSpec(
            name=dataset_name,
            model_interface=interface_name,
            storage=storage,
            parts=resolved_parts,
        )

    return resolved_datasets, interfaces, interface_storage


def resolve_sequifier_config(
    config: SequifierConfig,
    metadata: DatasetMetadata | dict[str, DatasetMetadata],
) -> ResolvedSequifierConfig:
    """Compile a training plan with caller-provided dataset metadata."""
    from sequifier.config.training_resolution import resolve_training_run

    if isinstance(metadata, DatasetMetadata):
        parts = [
            f"{name}.{part}"
            for name, dataset in config.dataset_training.items()
            for part in dataset.parts
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
        key: args.pop(key) for key in ("run", "model", "training_plan") if key in args
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
            for name, dataset in authored.dataset_training.items()
            for part in dataset.parts
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
class SelectedModelComponents:
    backbone: BackboneComponentConfig
    ingestion: IngestionComponentConfig
    decoder: DecoderComponentConfig


@dataclass(frozen=True)
class SelectedInterfaceConfig:
    """Explicit run and interface configuration for model construction."""

    project_root: str
    model_name: str
    device: str
    seed: int
    model: SelectedModelComponents
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


@dataclass(frozen=True)
class SelectedDatasetPartConfig(SelectedInterfaceConfig):
    """Explicit dataset-part configuration used by data loaders."""

    runtime_settings: TrainingRuntimeSettings
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
def interface_build_view(
    config: Any,
    interface: ResolvedModelInterface,
) -> SelectedInterfaceConfig:
    """Select one interface for model and objective construction."""

    return SelectedInterfaceConfig(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        interface=interface,
        model=SelectedModelComponents(
            backbone=config.model.backbone,
            ingestion=interface.ingestion,
            decoder=interface.decoder,
        ),
    )


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
        runtime_settings=config.runtime_settings,
        static_features_path=part.metadata.static_features_path,
        tensor_payload_version=part.metadata.tensor_payload_version,
        storage_layout=config.dataset_storage(dataset_name).storage_layout,
        batch_size=batch_size,
        **vars(interface_build_view(config, interface)),
        dataset_training=dataset,
        part=part,
        curriculum_training=curriculum_column is not None,
        curriculum_column=curriculum_column,
        epochs=sum(
            phase.epochs
            for phase in config.training_plan
            if isinstance(phase, (ResolvedTrainingPhase, PolicyImitationPhaseConfig))
        ),
        read_format=part.read_format,
        window_stride=window_stride,
        metadata_config_path=part.metadata_config_path,
    )


TrainModel = ResolvedSequifierConfig


# Public authored API; native schemas above are resolved execution projections.
from sequifier.config.ownership import (  # noqa: E402
    InputContract,
    RunSettings,
    SplitSettings,
)


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


class DatasetTrainingSpecModel(ConfigModel):
    model_interface: str | None = None
    parts: dict[str, DatasetPart] = Field(min_length=1)
    split: SplitSettings = Field(default_factory=SplitSettings)


class SequifierConfig(ConfigModel):
    """The authored training configuration, with one owner for each setting."""

    run: RunSettings
    model: OwnedModelSpec | None = None
    initialize_from: str | None = None
    actor_initialization: Literal["attach", "history_only"] = "attach"
    dataset_training: dict[str, DatasetTrainingSpecModel] = Field(default_factory=dict)
    policy_bindings: dict[str, PolicyBinding] = Field(default_factory=dict)
    training_plan: TrainingPlanModel

    @model_serializer(mode="wrap")
    def serialize_actor_initialization(self, handler):
        values = handler(self)
        if self.actor_initialization == "attach":
            values.pop("actor_initialization", None)
        return values

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, values):
        if not isinstance(values, dict):
            return values
        values = copy.deepcopy(values)
        model = values.get("model")
        if isinstance(model, ConfigModel):
            model = values["model"] = model.model_dump(exclude_unset=True)
        model = model or {}
        if not isinstance(model, dict):
            return values
        interfaces = model.get("interfaces", {})
        backbone = model.get("backbone")
        context = model.get("execution", {}).get("context_length")
        if isinstance(backbone, dict) and context is not None:
            architecture = backbone.get("architecture")
            if isinstance(architecture, dict):
                architecture.setdefault("max_context_length", context)
        datasets = values.get("dataset_training", {})
        if not isinstance(datasets, dict):
            return values
        for name, dataset in datasets.items():
            if isinstance(dataset, ConfigModel):
                dataset = datasets[name] = dataset.model_dump(exclude_unset=True)
            if not isinstance(dataset, dict):
                continue
            if isinstance(interfaces, dict) and len(interfaces) == 1:
                dataset.setdefault("model_interface", next(iter(interfaces)))
            for part in dataset.get("parts", {}).values():
                if isinstance(part, dict):
                    part.setdefault("type", "prepared")
        plan = values.get("training_plan", {})
        if isinstance(plan, ConfigModel):
            plan = values["training_plan"] = plan.model_dump(exclude_unset=True)
        if not isinstance(plan, dict):
            return values
        implicit = None
        if len(datasets) == 1:
            name, dataset = next(iter(datasets.items()))
            parts = dataset.get("parts", {})
            implicit = f"{name}.{next(iter(parts))}" if len(parts) == 1 else name
        for phase in plan.get("phases", []):
            phase.setdefault("algorithm", "supervised")
            if implicit and phase["algorithm"] in _SEQUENCE_ALGORITHMS:
                phase.setdefault("sources", [{"source": implicit}])
        return values

    @model_validator(mode="after")
    def validate_contract(self):
        if self.actor_initialization == "history_only" and (
            not self.initialize_from
            or self.model is None
            or self.model.backbone is None
            or not self.model.interfaces
            or not self.policy_bindings
            or self.training_plan.phases[0].algorithm == "supervised"
        ):
            raise ValueError(
                "actor_initialization: history_only requires initialize_from, "
                "a complete downstream model, policy_bindings and a policy first phase"
            )
        if (
            self.model is None or self.model.backbone is None
        ) and self.initialize_from is None:
            raise ValueError("a run requires a complete model or initialize_from")
        for name, dataset in self.dataset_training.items():
            _identifier(name, "Dataset name")
            for part in dataset.parts:
                _identifier(part, "Part name")
        for phase in self.training_plan.phases:
            refs = phase.data_sources
            for ref in refs:
                ref.validate_location(self.dataset_training, f"phase {phase.name!r}")
            if phase.algorithm == "supervised":
                for name in {ref.dataset for ref in refs}:
                    if name not in phase.learning.datasets:
                        raise ValueError(
                            f"{phase.name}: declare learning.datasets.{name}"
                        )
                    interface_name = self.dataset_training[name].model_interface
                    if self.model and interface_name in self.model.interfaces:
                        phase.learning.datasets[name].validate_targets(
                            self.model.interfaces[interface_name].target_columns,
                            f"{phase.name}: learning.datasets.{name}",
                        )
        return self

    def trial_config_values(self):
        """Keep inherited execution fields unset when materializing a search trial."""
        values = self.model_dump(mode="python")
        values["run"] = self.run.model_dump(exclude_unset=True)
        if self.model is not None:
            values["model"]["execution"] = self.model.execution.model_dump(
                exclude_unset=True
            )
        return values

    def build_runtime_settings(self) -> TrainingRuntimeSettings:
        """Resolve execution settings from their current owners."""
        run = self.run
        execution = run.execution
        contract = (
            self.model.execution.model_dump(exclude_unset=True) if self.model else {}
        )
        training = {
            **contract,
            **execution.precision.model_dump(),
            **execution.compilation.model_dump(),
            **execution.data_loading.model_dump(),
            **run.logging.model_dump(),
            "max_concat_tensors": execution.max_concat_tensors,
            "distributed": execution.distribution.strategy != "local",
            "data_parallelism": None
            if execution.distribution.strategy == "local"
            else execution.distribution.strategy,
            "world_size": execution.distribution.world_size,
            "backend": execution.distribution.backend,
            "fsdp_cpu_offload": execution.distribution.cpu_offload,
            "resume": run.recovery.resume.model_dump(),
            "latest_checkpoint_interval_minutes": run.recovery.latest_checkpoint_interval_minutes,
            "enforce_determinism": run.reproducibility.enforce_determinism,
            "inference_batch_size": run.export.batch_size,
        }
        return TrainingRuntimeSettings.model_validate(training)


ResolvedSequifierConfig.model_rebuild()
