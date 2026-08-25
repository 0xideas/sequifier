"""Canonical composable-dataset training configuration and resolution.

This module intentionally contains no migration adapter for the historical
single-dataset YAML schema.  A few properties on the resolved models provide
flat runtime views for existing low-level builders; those properties are not
authored configuration fields.
"""

from __future__ import annotations

import keyword
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional

import torch
import torch_optimizer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

import sequifier
import sequifier.optimizers
from sequifier.config.freezing_config import LayerFreezingConfigFields
from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata

# These canonical component definitions are imported from the public config
# facade to keep a single component schema.
from sequifier.config.train_config import (  # noqa: E402  (late circular import)
    BackboneComponentConfig,
    BERTSpecModel,
    DecoderComponentConfig,
    DotDict,
    FeatureLayoutRegistryModel,
    IngestionComponentConfig,
    NextOccurrenceConfigModel,
    ResumeConfig,
)
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    derive_target_column_types,
    metadata_config_path_from_preprocessing_data_path,
    normalize_path,
    resolve_window_view,
)
from sequifier.model.embedding import validate_embedding_layer_names
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    BERTObjective,
    NextOccurrenceObjective,
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.special_tokens import (
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_NAMES,
    resolve_categorical_decoder_ids,
)
from sequifier.typechecking import beartype


@beartype
def _identifier(value: str, usage: str) -> str:
    if "." in value or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"{usage} {value!r} must be a valid identifier and cannot contain '.'."
        )
    return value


@beartype
def _unique_columns(value: list[str], usage: str) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError(f"{usage} cannot contain duplicate columns.")
    return value


class GlobalTrainingSpecModel(BaseModel):
    """Run-wide data, optimization, precision, and distribution settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    read_format: Literal["csv", "parquet", "pt"] = "parquet"
    training_objective: str
    context_length: int = Field(gt=0)
    target_offset: int = Field(default=1, ge=0)
    model_window_stride: Optional[int] = Field(default=None, gt=0)
    inference_batch_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    accumulation_steps: Optional[int] = Field(default=None, gt=0)
    learning_rate: float = Field(gt=0)
    optimizer: DotDict = Field(default_factory=lambda: DotDict({"name": "Adam"}))
    scheduler: DotDict = Field(
        default_factory=lambda: DotDict(
            {"name": "StepLR", "step_size": 1, "gamma": 0.99}
        )
    )
    scheduler_step_on: Literal["epoch", "batch"] = "epoch"
    gradient_clip: Optional[float] = Field(default=None, gt=0)
    bert_spec: Optional[BERTSpecModel] = None
    next_occurrence_config: Optional[NextOccurrenceConfigModel] = None

    device_max_concat_length: int = Field(default=12, gt=0)
    log_interval: int = Field(default=10, gt=0)
    early_stopping_epochs: Optional[int] = Field(default=None, gt=0)
    save_interval_epochs: int = Field(default=1, gt=0)
    save_latest_interval_minutes: Optional[float] = None
    save_interval_minutes: Optional[float] = None
    save_interval_batches: Optional[int] = None
    save_interval_val_loss: bool = True
    calculate_validation_loss_on_initialization: bool = True
    resume: Optional[ResumeConfig] = None
    enforce_determinism: bool = False
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: int | float = 16
    world_size: int = Field(default=1, gt=0)
    num_workers: int = Field(default=0, ge=0)
    backend: str = "nccl"
    layer_type_dtypes: Optional[dict[str, str]] = None
    layer_autocast: bool = False
    data_parallelism: Optional[Literal["DDP", "FSDP"]] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: Literal["outer", "inner", "none"] = "outer"
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"

    @field_validator("training_objective")
    @classmethod
    @beartype
    def validate_objective(cls, value: str) -> str:
        if value not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {value}"
            )
        return value

    @field_validator("optimizer", mode="before")
    @classmethod
    @beartype
    def validate_optimizer(cls, value: Any) -> DotDict:
        value = dict(value)
        name = value.get("name")
        if not name:
            raise ValueError("optimizer must specify 'name'")
        if not (
            hasattr(torch.optim, name)
            or hasattr(torch_optimizer, name)
            or hasattr(sequifier.optimizers, name)
        ):
            raise ValueError(f"{name} not in the configured optimizer registries")
        return DotDict(value)

    @field_serializer("optimizer", "scheduler")
    @beartype
    def serialize_dot_dict(self, value: DotDict) -> dict[str, Any]:
        return dict(value)

    @field_validator("scheduler", mode="before")
    @classmethod
    @beartype
    def validate_scheduler(cls, value: Any) -> DotDict:
        value = dict(value)
        name = value.get("name")
        if not name:
            raise ValueError("scheduler must specify 'name'")
        if not hasattr(torch.optim.lr_scheduler, name):
            raise ValueError(f"{name} not in torch.optim.lr_scheduler")
        return DotDict(value)

    @field_validator("layer_type_dtypes")
    @classmethod
    @beartype
    def validate_layer_dtypes(cls, value: Optional[dict[str, str]]):
        if value is None:
            return value
        allowed_keys = {"embedding", "linear", "conv", "norm", "decoder"}
        allowed_types = {
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "float8_e4m3fn",
            "float8_e5m2",
        }
        if invalid := set(value) - allowed_keys:
            raise ValueError(f"Invalid layer_type_dtypes keys: {sorted(invalid)}")
        if invalid := set(value.values()) - allowed_types:
            raise ValueError(f"Invalid layer_type_dtypes values: {sorted(invalid)}")
        return value

    @model_validator(mode="after")
    @beartype
    def validate_distribution(self):
        if self.distributed and self.data_parallelism is None:
            raise ValueError("distributed=true requires data_parallelism")
        if self.data_parallelism != "FSDP" and self.fsdp_cpu_offload is not None:
            raise ValueError("fsdp_cpu_offload is only valid with FSDP")
        if self.data_parallelism == "FSDP":
            if self.fsdp_cpu_offload is None:
                raise ValueError("FSDP requires fsdp_cpu_offload")
            if self.layer_type_dtypes is not None:
                raise ValueError("FSDP does not support manual layer pre-casting")
            if self.torch_compile == "outer":
                raise ValueError("FSDP requires torch_compile 'none' or 'inner'")
        if self.data_parallelism == "DDP" and self.torch_compile == "inner":
            raise ValueError("DDP requires torch_compile 'none' or 'outer'")
        for name in (
            "save_latest_interval_minutes",
            "save_interval_minutes",
            "save_interval_batches",
        ):
            value = getattr(self, name)
            if (
                value is not None
                and value <= 0
                and os.getenv("SEQUIFIER_TESTING") != "1"
            ):
                raise ValueError(f"{name} must be larger than zero")
        return self


class ModelInterfaceSpecModel(BaseModel):
    """Architecture and selected-column contract for one named model route."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_columns: list[str] = Field(..., min_length=1)
    target_columns: list[str] = Field(..., min_length=1)
    categorical_decoder_special_tokens: dict[
        str, list[Literal["unknown", "other", "mask"]]
    ] = Field(default_factory=dict)
    feature_layout: Optional[FeatureLayoutRegistryModel] = None
    ingestion: IngestionComponentConfig
    decoder: DecoderComponentConfig

    @field_validator("input_columns", "target_columns")
    @classmethod
    @beartype
    def validate_columns(cls, value: list[str], info):
        return _unique_columns(value, info.field_name)

    @field_validator("categorical_decoder_special_tokens")
    @classmethod
    @beartype
    def validate_decoder_tokens(cls, value):
        if any(len(tokens) != len(set(tokens)) for tokens in value.values()):
            raise ValueError(
                "categorical_decoder_special_tokens cannot contain duplicate tokens"
            )
        return {
            column: [name for name in SPECIAL_TOKEN_NAMES if name in tokens]
            for column, tokens in value.items()
        }

    @model_validator(mode="after")
    @beartype
    def validate_interface_contract(self):
        input_columns = set(self.input_columns)
        auxiliary_columns = set(self.ingestion.auxiliary_input_columns)
        if missing := auxiliary_columns - input_columns:
            raise ValueError(
                "ingestion.auxiliary_input_columns references unknown input "
                f"columns: {sorted(missing)}"
            )
        if self.feature_layout is not None:
            for layout_name, layout in self.feature_layout.items():
                if missing := set(layout.columns) - input_columns:
                    raise ValueError(
                        f"feature_layout {layout_name!r} references unknown "
                        f"columns outside input_columns: {sorted(missing)}"
                    )
        return self


class ModelSpecModel(BaseModel):
    """One shared backbone and one or more named interfaces."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    backbone: BackboneComponentConfig
    interfaces: dict[str, ModelInterfaceSpecModel] = Field(..., min_length=1)

    @field_validator("interfaces")
    @classmethod
    @beartype
    def validate_interface_names(cls, value):
        for name in value:
            _identifier(name, "Model interface name")
        return value

    @beartype
    def _single_interface(self) -> ModelInterfaceSpecModel:
        if len(self.interfaces) != 1:
            raise AttributeError(
                "A model interface selection is required when multiple interfaces "
                "are configured"
            )
        return next(iter(self.interfaces.values()))

    @property
    @beartype
    def ingestion(self) -> IngestionComponentConfig:
        """Single-interface compatibility view for low-level builders."""

        return self._single_interface().ingestion

    @property
    @beartype
    def decoder(self) -> DecoderComponentConfig:
        """Single-interface compatibility view for low-level builders."""

        return self._single_interface().decoder


class DatasetPartSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_config_path: str


class DatasetFreezingSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: LayerFreezingConfigFields = Field(
        default_factory=LayerFreezingConfigFields
    )
    ingestion: LayerFreezingConfigFields = Field(
        default_factory=LayerFreezingConfigFields
    )
    ingestion_adapter: bool = False
    decoder: LayerFreezingConfigFields = Field(
        default_factory=LayerFreezingConfigFields
    )

    @property
    @beartype
    def active(self) -> bool:
        return self.ingestion_adapter or any(
            value.has_freezing_policy
            for value in (self.backbone, self.ingestion, self.decoder)
        )


class DatasetTrainingSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_interface: str
    parts: dict[str, DatasetPartSpecModel] = Field(..., min_length=1)
    criterion: dict[str, str] = Field(..., min_length=1)
    class_weights: Optional[dict[str, list[float]]] = None
    loss_weights: Optional[dict[str, float]] = None
    class_share_log_columns: list[str] = Field(default_factory=list)
    freezing: DatasetFreezingSpecModel = Field(default_factory=DatasetFreezingSpecModel)

    @field_validator("model_interface")
    @classmethod
    @beartype
    def validate_interface_name(cls, value):
        return _identifier(value, "Model interface reference")

    @field_validator("parts")
    @classmethod
    @beartype
    def validate_part_names(cls, value):
        for name in value:
            _identifier(name, "Dataset part name")
        return value

    @field_validator("criterion")
    @classmethod
    @beartype
    def validate_criteria(cls, value):
        for name in value.values():
            if not hasattr(torch.nn, name):
                raise ValueError(f"Criterion {name!r} not found in torch.nn")
        return value


class TrainingSourceSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    weight: Optional[float] = Field(default=None, gt=0)
    batches_per_selection: Optional[int] = Field(default=None, gt=0)

    @field_validator("ref")
    @classmethod
    @beartype
    def validate_ref(cls, value):
        parts = value.split(".")
        if len(parts) > 2:
            raise ValueError("Source refs use only dataset or dataset.part")
        for part in parts:
            _identifier(part, "Source reference component")
        return value


class TrainingPhaseSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    epochs: int = Field(gt=0)
    mode: Literal["sequential", "interleaved"]
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
        if self.mode == "sequential":
            if self.selection is not None:
                raise ValueError("Sequential phases cannot configure selection")
            if any(source.batches_per_selection is not None for source in self.sources):
                raise ValueError(
                    "batches_per_selection is invalid in sequential phases"
                )
            if any(source.weight is not None for source in self.sources):
                raise ValueError("weight is invalid in sequential phases")
        else:
            self.selection = self.selection or "round_robin"
            if self.selection == "round_robin" and any(
                source.weight is not None for source in self.sources
            ):
                raise ValueError("weight is only valid for weighted_random selection")
        return self


class TrainingPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phases: list[TrainingPhaseSpecModel] = Field(..., min_length=1)

    @model_validator(mode="after")
    @beartype
    def validate_unique_names(self):
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Training phase names must be unique")
        return self


class EvaluationMonitorSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    metric: Literal["loss"] = "loss"
    mode: Literal["min", "max"] = "min"


class EvaluationSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[TrainingSourceSpecModel] = Field(..., min_length=1)
    monitor: Optional[EvaluationMonitorSpecModel] = None

    @model_validator(mode="after")
    @beartype
    def validate_monitor(self):
        if any(
            source.weight is not None or source.batches_per_selection is not None
            for source in self.sources
        ):
            raise ValueError(
                "Evaluation sources cannot configure weight or batches_per_selection"
            )
        refs = [source.ref for source in self.sources]
        if len(refs) != len(set(refs)):
            raise ValueError("Evaluation sources must be unique")
        if self.monitor is not None and self.monitor.source not in refs:
            raise ValueError("evaluation.monitor.source must be an evaluation source")
        if len(refs) == 1 and self.monitor is None:
            self.monitor = EvaluationMonitorSpecModel(source=refs[0])
        return self


class SequifierConfig(BaseModel):
    """Canonical user-authored training configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    model_name: str
    device: str
    seed: int = 1010
    global_training_spec: GlobalTrainingSpecModel
    model_spec: ModelSpecModel
    dataset_training_spec: dict[str, DatasetTrainingSpecModel] = Field(
        ..., min_length=1
    )
    training_plan: TrainingPlanModel
    evaluation: Optional[EvaluationSpecModel] = None

    export_generative_model: bool = True
    export_embedding_model: bool = False
    embedding_layer_names: list[StrictStr] = Field(
        default_factory=lambda: ["backbone.final_norm"], min_length=1
    )
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    @field_validator("dataset_training_spec")
    @classmethod
    @beartype
    def validate_dataset_names(cls, value):
        for name in value:
            _identifier(name, "Dataset name")
        return value

    @field_validator("model_name")
    @classmethod
    @beartype
    def validate_model_name(cls, value):
        if "embedding" in value:
            raise ValueError("model_name cannot contain 'embedding'")
        return value

    @model_validator(mode="after")
    @beartype
    def validate_relationships(self):
        if self.global_training_spec.early_stopping_epochs is not None and (
            self.evaluation is None or self.evaluation.monitor is None
        ):
            raise ValueError("early stopping requires evaluation.monitor")

        for interface in self.model_spec.interfaces.values():
            validate_embedding_layer_names(
                self.embedding_layer_names,
                SimpleNamespace(
                    backbone=self.model_spec.backbone,
                    decoder=interface.decoder,
                ),
            )

        scheduler_total_steps = self.global_training_spec.scheduler.get("total_steps")
        if scheduler_total_steps is not None:
            total_epochs = sum(phase.epochs for phase in self.training_plan.phases)
            if self.global_training_spec.scheduler_step_on == "epoch":
                if scheduler_total_steps != total_epochs:
                    raise ValueError(
                        "scheduler total steps: "
                        f"{scheduler_total_steps} != {total_epochs}: total epochs"
                    )
            else:
                warnings.warn(
                    f"{scheduler_total_steps} scheduler steps at {total_epochs} "
                    "epochs implies "
                    f"{scheduler_total_steps / total_epochs:.2f} batches. "
                    "Does this seem correct?",
                    stacklevel=2,
                )

        referenced_interfaces = set()
        for dataset_name, dataset in self.dataset_training_spec.items():
            if dataset.model_interface not in self.model_spec.interfaces:
                raise ValueError(
                    f"Dataset {dataset_name!r} references unknown model interface "
                    f"{dataset.model_interface!r}."
                )
            referenced_interfaces.add(dataset.model_interface)
            interface = self.model_spec.interfaces[dataset.model_interface]
            if set(dataset.criterion) != set(interface.target_columns):
                raise ValueError(
                    f"Dataset {dataset_name!r} criterion keys must equal interface "
                    f"target_columns."
                )
            if dataset.loss_weights is not None and set(dataset.loss_weights) - set(
                interface.target_columns
            ):
                raise ValueError(
                    f"Dataset {dataset_name!r} loss_weights references unknown targets."
                )
            if dataset.class_weights is not None and set(dataset.class_weights) - set(
                interface.target_columns
            ):
                raise ValueError(
                    f"Dataset {dataset_name!r} class_weights references unknown targets."
                )
        unreferenced = set(self.model_spec.interfaces) - referenced_interfaces
        if unreferenced:
            warnings.warn(
                f"Unreferenced model interfaces: {sorted(unreferenced)}",
                stacklevel=2,
            )

        for phase in self.training_plan.phases:
            for source in phase.sources:
                self._validate_source(source.ref, f"training phase {phase.name!r}")
        if self.evaluation is not None:
            for source in self.evaluation.sources:
                self._validate_source(source.ref, "evaluation")
            needs_monitor = len(self.evaluation.sources) > 1 and (
                self.global_training_spec.save_interval_val_loss
                or self.global_training_spec.early_stopping_epochs is not None
            )
            if needs_monitor and self.evaluation.monitor is None:
                raise ValueError(
                    "Multiple evaluation sources require evaluation.monitor when "
                    "validation-based saving or early stopping is enabled."
                )

        objective = get_objective_class(self.global_training_spec.training_objective)
        is_bert = issubclass(objective, BERTObjective)
        is_next = issubclass(objective, NextOccurrenceObjective)
        if (self.global_training_spec.bert_spec is not None) != is_bert:
            raise ValueError("bert_spec must be configured exactly for BERT training")
        if (self.global_training_spec.next_occurrence_config is not None) != is_next:
            raise ValueError(
                "next_occurrence_config must be configured exactly for "
                "next_occurrence training"
            )
        if (
            len(self.dataset_training_spec) > 1
            and self.global_training_spec.data_parallelism == "FSDP"
        ):
            raise ValueError("Multi-dataset training does not support FSDP")

        context_length = self.global_training_spec.context_length
        if context_length > self.model_spec.backbone.architecture.max_context_length:
            raise ValueError(
                "global_training_spec.context_length exceeds backbone "
                "max_context_length"
            )
        for name, interface in self.model_spec.interfaces.items():
            if interface.decoder.support > context_length:
                raise ValueError(
                    f"Interface {name!r} decoder support exceeds context_length"
                )
            decoded_length = context_length - interface.decoder.support + 1
            if interface.decoder.prediction_length > decoded_length:
                raise ValueError(
                    f"Interface {name!r} prediction_length exceeds decoded length"
                )
            objective.validate_prediction_length(
                interface.decoder.prediction_length,
                context_length,
                usage="training",
            )
        if (
            not self.export_generative_model
            and not self.export_embedding_model
            and os.getenv("SEQUIFIER_PREVENT_EXPORT") is None
        ):
            raise ValueError("At least one model export must be enabled")
        return self

    @beartype
    def _validate_source(self, ref: str, usage: str) -> None:
        dataset_name, _, part_name = ref.partition(".")
        dataset = self.dataset_training_spec.get(dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown {usage} dataset source {dataset_name!r}")
        if part_name and part_name not in dataset.parts:
            raise ValueError(f"Unknown {usage} part source {ref!r}")


class ResolvedDatasetPart(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    metadata_config_path: str
    metadata: DatasetMetadata
    training_data_path: str
    validation_data_path: Optional[str] = None
    storage_form: Literal["file", "folder"]


class ResolvedModelInterface(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    input_columns: list[str]
    target_columns: list[str]
    target_column_types: dict[str, str]
    column_data_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    categorical_decoder_special_tokens: dict[str, list[str]]
    feature_layout: Optional[FeatureLayoutRegistryModel] = None
    ingestion: IngestionComponentConfig
    decoder: DecoderComponentConfig
    n_classes: dict[str, int]
    id_maps: dict[str, dict[str | int, int]]
    special_token_ids: dict[str, int]
    selected_columns_statistics: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    normalize_real_columns: bool = True
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView


class ResolvedDatasetTrainingSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    model_interface: str
    interface: ResolvedModelInterface
    parts: dict[str, ResolvedDatasetPart]
    criterion: dict[str, str]
    class_weights: Optional[dict[str, list[float]]] = None
    loss_weights: Optional[dict[str, float]] = None
    class_share_log_columns: list[str]
    freezing: DatasetFreezingSpecModel


class ResolvedTrainingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    dataset: str
    part: Optional[str] = None
    weight: float = 1.0
    batches_per_selection: int = 1


class ResolvedTrainingPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    epochs: int
    mode: Literal["sequential", "interleaved"]
    selection: Literal["round_robin", "weighted_random"] = "round_robin"
    sources: list[ResolvedTrainingSource]


class _TrainingSpecRuntimeView:
    """Attribute view joining run-wide and active-dataset training policy."""

    @beartype
    def __init__(
        self,
        global_spec: GlobalTrainingSpecModel,
        dataset: ResolvedDatasetTrainingSpec,
        epochs: int,
    ):
        object.__setattr__(self, "_global", global_spec)
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "epochs", epochs)

    @beartype
    def __getattr__(self, name: str) -> Any:
        dataset = object.__getattribute__(self, "_dataset")
        if hasattr(dataset, name):
            return getattr(dataset, name)
        return getattr(object.__getattribute__(self, "_global"), name)

    @beartype
    def __setattr__(self, name: str, value: Any) -> None:
        global_spec = object.__getattribute__(self, "_global")
        if hasattr(global_spec, name):
            setattr(global_spec, name, value)
            return
        raise AttributeError(name)


class ResolvedSequifierConfig(BaseModel):
    """Runtime configuration after all dataset parts and interfaces resolve."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    model_name: str
    device: str
    seed: int
    global_training_spec: GlobalTrainingSpecModel
    model_spec: ModelSpecModel
    dataset_training_spec: dict[str, ResolvedDatasetTrainingSpec]
    training_plan: list[ResolvedTrainingPhase]
    evaluation_sources: list[ResolvedTrainingSource]
    evaluation_monitor: Optional[EvaluationMonitorSpecModel]
    export_generative_model: bool
    export_embedding_model: bool
    embedding_layer_names: list[str]
    export_onnx: bool
    export_pt: bool
    export_with_dropout: bool = False

    @model_validator(mode="after")
    @beartype
    def validate_next_occurrence_metadata(self):
        objective = get_objective_class(self.global_training_spec.training_objective)
        if not issubclass(objective, NextOccurrenceObjective):
            return self

        next_config = self.global_training_spec.next_occurrence_config
        if next_config is None:
            raise ValueError(
                "next_occurrence_config must be set for next_occurrence training"
            )
        column = next_config.column_name
        for dataset_name, dataset in self.dataset_training_spec.items():
            interface = dataset.interface
            if column not in interface.target_columns:
                raise ValueError(
                    "next_occurrence_config.column_name must be one of "
                    f"target_columns for dataset {dataset_name!r}, got {column!r}"
                )
            if interface.target_column_types.get(column) != "categorical":
                raise ValueError(
                    "next_occurrence_config.column_name must refer to a "
                    "categorical target column"
                )
            if column not in interface.id_maps:
                raise ValueError(
                    "next_occurrence_config.column_name must have a "
                    f"preprocessing id_map, got {column!r}"
                )
            missing = [
                value
                for value in next_config.target_values
                if value not in interface.id_maps[column]
            ]
            if missing:
                raise ValueError(
                    "next_occurrence_config.target_values must match keys in "
                    f"id_maps[{column!r}] exactly, missing {missing!r}"
                )
        return self

    @model_validator(mode="after")
    @beartype
    def validate_model_execution_plans(self):
        """Compile each route while validation errors still retain Pydantic context."""

        from sequifier.model.decoders import resolve_decoding_plan
        from sequifier.model.ingestion_compiler import resolve_ingestion_plan

        validated_interfaces: set[str] = set()
        for dataset in self.dataset_training_spec.values():
            if dataset.model_interface in validated_interfaces:
                continue
            view = interface_build_view(self, dataset.interface)
            resolve_ingestion_plan(view)
            resolve_decoding_plan(view)
            validated_interfaces.add(dataset.model_interface)
        return self

    @property
    @beartype
    def dataset_count(self) -> int:
        return len(self.dataset_training_spec)

    @property
    @beartype
    def interface_names(self) -> tuple[str, ...]:
        return tuple(self.model_spec.interfaces)

    @beartype
    def dataset(self, name: Optional[str] = None) -> ResolvedDatasetTrainingSpec:
        if name is None:
            if len(self.dataset_training_spec) != 1:
                raise ValueError("A dataset selection is required")
            return next(iter(self.dataset_training_spec.values()))
        return self.dataset_training_spec[name]

    @beartype
    def interface(self, name: Optional[str] = None) -> ResolvedModelInterface:
        if name is None:
            names = {
                dataset.model_interface
                for dataset in self.dataset_training_spec.values()
            }
            if len(names) != 1:
                raise ValueError("A model interface selection is required")
            name = next(iter(names))
        for dataset in self.dataset_training_spec.values():
            if dataset.model_interface == name:
                return dataset.interface
        raise KeyError(name)

    # Internal, single-route compatibility properties used by existing IO and
    # objective implementations.  They cannot be authored in YAML.
    @property
    @beartype
    def training_spec(self):
        dataset = next(iter(self.dataset_training_spec.values()))
        return _TrainingSpecRuntimeView(
            self.global_training_spec,
            dataset,
            sum(phase.epochs for phase in self.training_plan),
        )

    @property
    @beartype
    def training_objective(self):
        return self.global_training_spec.training_objective

    @property
    @beartype
    def read_format(self):
        return self.global_training_spec.read_format

    @property
    @beartype
    def context_length(self):
        return self.global_training_spec.context_length

    @property
    @beartype
    def target_offset(self):
        return self.global_training_spec.target_offset

    @property
    @beartype
    def model_window_stride(self):
        return self.global_training_spec.model_window_stride

    @property
    @beartype
    def inference_batch_size(self):
        return self.global_training_spec.inference_batch_size

    @beartype
    def __getattr__(self, name: str):
        if name in {
            "input_columns",
            "target_columns",
            "target_column_types",
            "column_data_types",
            "categorical_columns",
            "real_columns",
            "categorical_decoder_special_tokens",
            "feature_layout",
            "n_classes",
            "id_maps",
            "special_token_ids",
            "selected_columns_statistics",
            "normalize_real_columns",
            "storage_layout",
            "window_view",
        }:
            dataset = next(iter(self.dataset_training_spec.values()))
            return getattr(dataset.interface, name)
        if name in {"data_path", "validation_data_path", "metadata_config_path"}:
            dataset = next(iter(self.dataset_training_spec.values()))
            part = next(iter(dataset.parts.values()))
            return {
                "data_path": part.training_data_path,
                "validation_data_path": part.validation_data_path,
                "metadata_config_path": part.metadata_config_path,
            }[name]
        raise AttributeError(name)


@dataclass(frozen=True)
class LoadedTrainConfig:
    config: SequifierConfig
    resolved: ResolvedSequifierConfig
    metadata: dict[str, DatasetMetadata]


@beartype
def _source(
    ref: str, accumulation_steps: Optional[int], **values: Any
) -> ResolvedTrainingSource:
    dataset, _, part = ref.partition(".")
    return ResolvedTrainingSource(
        ref=ref,
        dataset=dataset,
        part=part or None,
        weight=values.get("weight") or 1.0,
        batches_per_selection=(
            values.get("batches_per_selection") or accumulation_steps or 1
        ),
    )


@beartype
def _storage_form(path: str) -> Literal["file", "folder"]:
    value = Path(path)
    if value.exists():
        return "folder" if value.is_dir() else "file"
    return "file" if value.suffix else "folder"


@beartype
def _evaluated_parts(config: SequifierConfig) -> set[str]:
    selected: set[str] = set()
    if config.evaluation is None:
        return selected
    for source in config.evaluation.sources:
        dataset_name, _, part_name = source.ref.partition(".")
        if part_name:
            selected.add(source.ref)
        else:
            selected.update(
                f"{dataset_name}.{name}"
                for name in config.dataset_training_spec[dataset_name].parts
            )
    return selected


@beartype
def _part_signature(
    metadata: DatasetMetadata, interface: ModelInterfaceSpecModel
) -> dict[str, Any]:
    relevant = list(dict.fromkeys(interface.input_columns + interface.target_columns))
    missing = set(relevant) - set(metadata.column_data_types)
    if missing:
        raise ValueError(f"Metadata is missing interface columns: {sorted(missing)}")
    categorical = [
        column
        for column in relevant
        if "int" in metadata.column_data_types[column].lower()
    ]
    real = [
        column
        for column in relevant
        if "float" in metadata.column_data_types[column].lower()
    ]
    unknown_types = set(relevant) - set(categorical) - set(real)
    if unknown_types:
        raise ValueError(
            f"Unsupported metadata dtypes for columns: {sorted(unknown_types)}"
        )
    return {
        "column_data_types": {
            column: metadata.column_data_types[column] for column in relevant
        },
        "storage_layout": metadata.storage_layout,
        "n_classes": {
            column: metadata.n_classes[column]
            for column in categorical
            if column in metadata.n_classes
        },
        "id_maps": {column: metadata.id_maps.get(column, {}) for column in categorical},
        "special_token_ids": metadata.special_token_ids,
        "normalize_real_columns": metadata.normalize_real_columns,
        "normalization_statistics": (
            {
                column: metadata.selected_columns_statistics.get(column, {})
                for column in real
            }
            if metadata.normalize_real_columns
            else {}
        ),
    }


@beartype
def _assert_compatible(
    expected: dict[str, Any], actual: dict[str, Any], usage: str
) -> None:
    mismatches = [key for key in expected if expected[key] != actual.get(key)]
    if mismatches:
        raise ValueError(
            f"{usage} has incompatible schema fields {mismatches}. The increment "
            "must be preprocessed against the dataset's established schema."
        )


@beartype
def _resolve_interface(
    name: str,
    spec: ModelInterfaceSpecModel,
    metadata: DatasetMetadata,
    global_spec: GlobalTrainingSpecModel,
) -> ResolvedModelInterface:
    signature = _part_signature(metadata, spec)
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
    n_classes = {
        column: metadata.n_classes[column]
        for column in set(categorical_columns) | categorical_targets
    }
    target_decoder_ids = resolve_categorical_decoder_ids(
        spec.target_columns,
        target_types,
        n_classes,
        spec.categorical_decoder_special_tokens,
    )
    target_n_classes = {column: len(ids) for column, ids in target_decoder_ids.items()}
    target_global_to_decoder = {}
    for column, ids in target_decoder_ids.items():
        inverse = {global_id: decoder_id for decoder_id, global_id in enumerate(ids)}
        target_global_to_decoder[column] = [
            inverse.get(global_id, -1) for global_id in range(n_classes[column])
        ]
    target_offset = target_offset_for_objective(
        global_spec.training_objective, global_spec.target_offset
    )
    window_view = ModelWindowView(
        context_length=global_spec.context_length,
        objective=global_spec.training_objective,
        target_offset=target_offset,
    )
    resolve_window_view(metadata.storage_layout, window_view)
    return ResolvedModelInterface(
        name=name,
        input_columns=spec.input_columns,
        target_columns=spec.target_columns,
        target_column_types=target_types,
        column_data_types=signature["column_data_types"],
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        categorical_decoder_special_tokens={
            column: list(tokens)
            for column, tokens in spec.categorical_decoder_special_tokens.items()
        },
        feature_layout=spec.feature_layout,
        ingestion=spec.ingestion,
        decoder=spec.decoder,
        n_classes=n_classes,
        id_maps={
            column: metadata.id_maps[column]
            for column in n_classes
            if column in metadata.id_maps
        },
        special_token_ids=metadata.special_token_ids,
        selected_columns_statistics={
            column: metadata.selected_columns_statistics.get(column, {})
            for column in real_columns
        },
        normalize_real_columns=metadata.normalize_real_columns,
        target_decoder_ids=target_decoder_ids,
        target_n_classes=target_n_classes,
        target_global_to_decoder=target_global_to_decoder,
        storage_layout=metadata.storage_layout,
        window_view=window_view,
    )


@beartype
def _interface_semantics(interface: ResolvedModelInterface) -> dict[str, Any]:
    return interface.model_dump(
        mode="python",
        exclude={"name", "ingestion", "decoder", "feature_layout"},
    )


@beartype
def resolve_sequifier_config(
    config: SequifierConfig,
    metadata: DatasetMetadata | dict[str, DatasetMetadata],
    *,
    part_overrides: Optional[dict[str, dict[str, str]]] = None,
) -> ResolvedSequifierConfig:
    """Resolve every dataset part, compatibility contract, and source ref."""

    part_refs = [
        f"{dataset_name}.{part_name}"
        for dataset_name, dataset in config.dataset_training_spec.items()
        for part_name in dataset.parts
    ]
    if isinstance(metadata, DatasetMetadata):
        if len(part_refs) != 1:
            raise ValueError(
                "Metadata must be keyed by dataset.part for multiple parts"
            )
        metadata_by_part = {part_refs[0]: metadata}
    else:
        metadata_by_part = metadata
    missing_metadata = set(part_refs) - set(metadata_by_part)
    if missing_metadata:
        raise ValueError(f"Missing metadata for parts: {sorted(missing_metadata)}")

    evaluated = _evaluated_parts(config)
    overrides = part_overrides or {}
    resolved_datasets: dict[str, ResolvedDatasetTrainingSpec] = {}
    interface_semantics: dict[str, dict[str, Any]] = {}

    for dataset_name, dataset_spec in config.dataset_training_spec.items():
        interface_spec = config.model_spec.interfaces[dataset_spec.model_interface]
        resolved_parts = {}
        expected_signature = None
        expected_form = None
        first_metadata = None
        for part_name, part_spec in dataset_spec.parts.items():
            ref = f"{dataset_name}.{part_name}"
            part_metadata = metadata_by_part[ref]
            signature = _part_signature(part_metadata, interface_spec)
            if expected_signature is None:
                expected_signature = signature
                first_metadata = part_metadata
            else:
                _assert_compatible(expected_signature, signature, ref)

            split_paths = list(part_metadata.split_paths)
            override = overrides.get(ref, {})
            training_path = override.get("data_path") or (
                split_paths[0] if split_paths else None
            )
            if training_path is None:
                raise ValueError(f"Training part {ref!r} needs at least one split path")
            validation_path = override.get("validation_data_path") or (
                split_paths[1] if len(split_paths) > 1 else None
            )
            if ref in evaluated and validation_path is None:
                raise ValueError(
                    f"Evaluation part {ref!r} requires a second split path"
                )
            training_path = normalize_path(training_path, config.project_root)
            validation_path = (
                normalize_path(validation_path, config.project_root)
                if validation_path is not None
                else None
            )
            form = _storage_form(training_path)
            if expected_form is None:
                expected_form = form
            elif expected_form != form:
                raise ValueError(
                    f"Dataset {dataset_name!r} parts must share file/folder storage form"
                )
            resolved_parts[part_name] = ResolvedDatasetPart(
                name=part_name,
                metadata_config_path=(
                    override.get("metadata_config_path")
                    or part_spec.metadata_config_path
                ),
                metadata=part_metadata,
                training_data_path=training_path,
                validation_data_path=validation_path,
                storage_form=form,
            )

        assert first_metadata is not None
        target_types = derive_target_column_types(
            interface_spec.target_columns,
            first_metadata.column_data_types,
        )
        for column in dataset_spec.class_share_log_columns:
            if column not in interface_spec.target_columns:
                raise ValueError(f"Class-share column {column!r} must be a target")
            if target_types[column] != "categorical":
                raise ValueError(f"Class-share column {column!r} must be categorical")
            if column not in first_metadata.n_classes:
                raise ValueError(f"Class-share column {column!r} needs n_classes")
            if column not in first_metadata.id_maps:
                raise ValueError(f"Class-share column {column!r} needs an id_map")
        interface = _resolve_interface(
            dataset_spec.model_interface,
            interface_spec,
            first_metadata,
            config.global_training_spec,
        )
        semantic_contract = _interface_semantics(interface)
        prior_contract = interface_semantics.get(dataset_spec.model_interface)
        if prior_contract is not None:
            _assert_compatible(
                prior_contract,
                semantic_contract,
                f"Datasets sharing interface {dataset_spec.model_interface!r}",
            )
        else:
            interface_semantics[dataset_spec.model_interface] = semantic_contract

        if dataset_spec.class_weights is not None:
            for column, weights in dataset_spec.class_weights.items():
                if interface.target_column_types[column] != "categorical":
                    raise ValueError(
                        f"class_weights[{column!r}] requires a categorical target"
                    )
                valid_lengths = {
                    interface.n_classes[column],
                    interface.target_n_classes[column],
                }
                if len(weights) not in valid_lengths:
                    raise ValueError(
                        f"class_weights[{column!r}] has length {len(weights)}; "
                        f"expected one of {sorted(valid_lengths)}"
                    )

        resolved_datasets[dataset_name] = ResolvedDatasetTrainingSpec(
            name=dataset_name,
            model_interface=dataset_spec.model_interface,
            interface=interface,
            parts=resolved_parts,
            criterion=dataset_spec.criterion,
            class_weights=dataset_spec.class_weights,
            loss_weights=dataset_spec.loss_weights,
            class_share_log_columns=dataset_spec.class_share_log_columns,
            freezing=dataset_spec.freezing,
        )

    accumulation = config.global_training_spec.accumulation_steps
    phases = [
        ResolvedTrainingPhase(
            name=phase.name,
            epochs=phase.epochs,
            mode=phase.mode,
            selection=phase.selection or "round_robin",
            sources=[
                _source(
                    source.ref,
                    accumulation,
                    weight=source.weight,
                    batches_per_selection=(
                        None
                        if phase.mode == "sequential"
                        else source.batches_per_selection
                    ),
                )
                for source in phase.sources
            ],
        )
        for phase in config.training_plan.phases
    ]
    evaluation_sources = (
        [_source(source.ref, accumulation) for source in config.evaluation.sources]
        if config.evaluation is not None
        else []
    )
    resolved = ResolvedSequifierConfig(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        global_training_spec=config.global_training_spec,
        model_spec=config.model_spec,
        dataset_training_spec=resolved_datasets,
        training_plan=phases,
        evaluation_sources=evaluation_sources,
        evaluation_monitor=(
            config.evaluation.monitor if config.evaluation is not None else None
        ),
        export_generative_model=config.export_generative_model,
        export_embedding_model=config.export_embedding_model,
        embedding_layer_names=config.embedding_layer_names,
        export_onnx=config.export_onnx,
        export_pt=config.export_pt,
        export_with_dropout=config.export_with_dropout,
    )
    return resolved


_SENSITIVE_OVERRIDES = {
    "data_path",
    "validation_data_path",
    "metadata_config_path",
    "preprocessing_data_path",
    "input_columns",
}
_INLINE_METADATA_KEYS = {
    "metadata_by_part",
    "column_data_types",
    "column_types",
    "n_classes",
    "id_maps",
    "special_token_ids",
    "selected_columns_statistics",
    "normalize_real_columns",
    "stored_context_width",
    "max_target_offset",
    "stored_window_layout_version",
    "storage_layout",
    "split_paths",
}


@beartype
def _override_part(config: SequifierConfig) -> str:
    if len(config.dataset_training_spec) != 1:
        raise ValueError(
            "Dataset-sensitive legacy CLI overrides require exactly one dataset"
        )
    dataset_name, dataset = next(iter(config.dataset_training_spec.items()))
    if len(dataset.parts) == 1:
        return f"{dataset_name}.{next(iter(dataset.parts))}"
    refs = {
        source.ref
        for phase in config.training_plan.phases
        for source in phase.sources
        if source.ref.startswith(f"{dataset_name}.")
    }
    if config.evaluation is not None:
        refs.update(
            source.ref
            for source in config.evaluation.sources
            if source.ref.startswith(f"{dataset_name}.")
        )
    if len(refs) == 1:
        return next(iter(refs))
    raise ValueError(
        "Legacy metadata/data-path CLI override is ambiguous because multiple "
        "dataset parts remain possible"
    )


@beartype
def _inline_metadata(
    values: dict[str, Any], global_spec: GlobalTrainingSpecModel
) -> DatasetMetadata:
    layout = values.get("storage_layout")
    if isinstance(layout, StoredWindowLayout):
        stored_context_width = layout.stored_context_width
        max_target_offset = layout.max_target_offset
        layout_version = layout.version
    elif isinstance(layout, dict):
        stored_context_width = layout.get("stored_context_width")
        max_target_offset = layout.get("max_target_offset", 1)
        layout_version = layout.get("version", 2)
    else:
        stored_context_width = values.get("stored_context_width")
        max_target_offset = values.get("max_target_offset", 1)
        layout_version = values.get("stored_window_layout_version", 2)
    if stored_context_width is None:
        stored_context_width = global_spec.context_length + max(
            1, global_spec.target_offset
        )
    return DatasetMetadata.model_validate(
        {
            "split_paths": values.get("split_paths", []),
            "column_data_types": values.get(
                "column_data_types", values.get("column_types", {})
            ),
            "n_classes": values.get("n_classes", {}),
            "id_maps": values.get("id_maps", {}),
            "special_token_ids": values.get(
                "special_token_ids", SPECIAL_TOKEN_IDS.ids_by_label
            ),
            "selected_columns_statistics": values.get(
                "selected_columns_statistics", {}
            ),
            "normalize_real_columns": values.get("normalize_real_columns", True),
            "stored_context_width": stored_context_width,
            "max_target_offset": max_target_offset,
            "stored_window_layout_version": layout_version,
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
    sensitive = {
        key: args.pop(key) for key in list(args) if key in _SENSITIVE_OVERRIDES
    }
    for key in ("model_name", "seed", "device"):
        if key in args:
            raw[key] = args.pop(key)
    if args:
        raise ValueError(f"Unsupported training CLI overrides: {sorted(args)}")

    config = try_catch_excess_keys(config_path, SequifierConfig, raw)
    part_overrides: dict[str, dict[str, str]] = {}
    if sensitive:
        selected_part = _override_part(config)
        dataset_name = selected_part.partition(".")[0]
        if "input_columns" in sensitive:
            interface_name = config.dataset_training_spec[dataset_name].model_interface
            config.model_spec.interfaces[interface_name].input_columns = sensitive.pop(
                "input_columns"
            )
        override = {}
        if "preprocessing_data_path" in sensitive:
            override["metadata_config_path"] = (
                metadata_config_path_from_preprocessing_data_path(
                    sensitive.pop("preprocessing_data_path")
                )
            )
        override.update(sensitive)
        part_overrides[selected_part] = override

    metadata_by_part: dict[str, DatasetMetadata] = {}
    all_part_refs = [
        (dataset_name, part_name, part)
        for dataset_name, dataset in config.dataset_training_spec.items()
        for part_name, part in dataset.parts.items()
    ]
    if skip_metadata:
        if len(all_part_refs) != 1 and "metadata_by_part" not in metadata_inline_values:
            raise ValueError(
                "skip_metadata with multiple parts requires metadata_by_part"
            )
        if "metadata_by_part" in metadata_inline_values:
            for ref, values in metadata_inline_values["metadata_by_part"].items():
                metadata_by_part[ref] = _inline_metadata(
                    values, config.global_training_spec
                )
        else:
            dataset_name, part_name, _ = all_part_refs[0]
            metadata_by_part[f"{dataset_name}.{part_name}"] = _inline_metadata(
                metadata_inline_values, config.global_training_spec
            )
    else:
        for dataset_name, part_name, part in all_part_refs:
            ref = f"{dataset_name}.{part_name}"
            configured_path = part_overrides.get(ref, {}).get(
                "metadata_config_path", part.metadata_config_path
            )
            if configured_path is None:
                raise ValueError(f"Part {ref!r} has no metadata_config_path")
            metadata_by_part[ref] = load_dataset_metadata(
                normalize_path(configured_path, config.project_root)
            )

    resolved = resolve_sequifier_config(
        config, metadata_by_part, part_overrides=part_overrides
    )
    return LoadedTrainConfig(
        config=config, resolved=resolved, metadata=metadata_by_part
    )


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> ResolvedSequifierConfig:
    return load_train_config_with_source(
        config_path, args_config, skip_metadata
    ).resolved


@beartype
def interface_build_view(
    config: ResolvedSequifierConfig, interface: ResolvedModelInterface
) -> SimpleNamespace:
    """Return the flat data expected by the existing ingestion/decoder compilers."""

    return SimpleNamespace(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        training_objective=config.global_training_spec.training_objective,
        training_spec=config.global_training_spec,
        input_columns=interface.input_columns,
        target_columns=interface.target_columns,
        target_column_types=interface.target_column_types,
        column_data_types=interface.column_data_types,
        categorical_columns=interface.categorical_columns,
        real_columns=interface.real_columns,
        categorical_decoder_special_tokens=interface.categorical_decoder_special_tokens,
        feature_layout=interface.feature_layout,
        n_classes=interface.n_classes,
        id_maps=interface.id_maps,
        special_token_ids=interface.special_token_ids,
        storage_layout=interface.storage_layout,
        window_view=interface.window_view,
        model_spec=SimpleNamespace(
            backbone=config.model_spec.backbone,
            ingestion=interface.ingestion,
            decoder=interface.decoder,
        ),
    )


@beartype
def dataset_part_view(
    config: ResolvedSequifierConfig,
    dataset_name: str,
    part_name: str,
) -> SimpleNamespace:
    """Return the flat, picklable config used by existing dataset loaders."""

    dataset = config.dataset_training_spec[dataset_name]
    part = dataset.parts[part_name]
    view = interface_build_view(config, dataset.interface)
    view.training_spec = _TrainingSpecRuntimeView(
        config.global_training_spec,
        dataset,
        sum(phase.epochs for phase in config.training_plan),
    )
    view.read_format = config.global_training_spec.read_format
    view.model_window_stride = config.global_training_spec.model_window_stride
    view.data_path = part.training_data_path
    view.validation_data_path = part.validation_data_path
    view.metadata_config_path = part.metadata_config_path
    return view
