"""Canonical Sequifier training configuration and resolution.

The concise singleton authoring surface is normalized to the canonical named
schema before validation. This module intentionally contains no migration
adapter for the historical flat single-dataset YAML schema. Runtime consumers
select a dataset, part, or model interface explicitly.
"""

from __future__ import annotations

import copy
import inspect
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar, Literal, Optional

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictStr,
    field_validator,
    model_validator,
)

# Reusable schemas are independent of the canonical root configuration.
from sequifier.config.components import (
    BackboneComponentConfig,
    BERTSpecModel,
    CategoricalHashingConfig,
    DecoderComponentConfig,
    FeatureLayoutRegistryModel,
    ImageEncoderConfig,
    IngestionComponentConfig,
    NextOccurrenceConfigModel,
    ResumeConfig,
)
from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.config.freezing_config import (
    IngestionFreezingConfig,
    LayerFreezingConfigFields,
)
from sequifier.config.identifiers import identifier as _identifier
from sequifier.config.image_config import (
    ImageSourceConfig,
    MediaPath,
    PreconditioningImageConfig,
    bind_image_paths,
    image_views,
)
from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.config.policy_config import PolicyBinding, ResolvedPolicyBinding
from sequifier.config.policy_optimization import SupervisedOptimizationConfig
from sequifier.config.policy_plan import (
    DPOPhaseConfig,
    FrozenNormalizationConfig,
    GRPOPhaseConfig,
    IQLPhaseConfig,
    PolicyImitationPhaseConfig,
    PolicyPhase,
    PPOPhaseConfig,
)
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.prediction_config import PredictionDistributionConfig
from sequifier.config.sources import TrainingSourceSpecModel
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


@beartype
def _unique_columns(value: list[str], usage: str) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError(f"{usage} cannot contain duplicate columns.")
    return value


_SINGLETON_CONFIG_NAME = "default"
_SINGLE_PHASE_NAME = "train"
_PHASE_FIELD_NAMES = frozenset(
    {
        "algorithm",
        "name",
        "epochs",
        "mode",
        "selection",
        "sources",
        "updates",
        "total_env_steps",
        "total_groups",
        "evaluation",
        "evaluation_interval",
        "checkpoint_interval",
        "batch_size",
        "accumulation_steps",
        "optimization",
        "handoff",
        "optimizer_state",
    }
)
_TRAINING_PLAN_POLICY_FIELDS = frozenset({"curriculum_training", "curriculum_column"})


@beartype
def _normalize_dataset_part_surface(datasets: Any) -> None:
    if not isinstance(datasets, dict):
        return
    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            continue
        if "part" in dataset and "parts" in dataset:
            raise ValueError(
                f"Dataset {dataset_name!r} cannot define both 'part' and 'parts'."
            )
        if "part" in dataset:
            dataset["parts"] = {_SINGLETON_CONFIG_NAME: dataset.pop("part")}


@beartype
def _implicit_source(values: dict[str, Any]) -> str | None:
    datasets = values.get("dataset_training")
    if not isinstance(datasets, dict) or len(datasets) != 1:
        return None
    dataset_name, dataset = next(iter(datasets.items()))
    if not isinstance(dataset, dict):
        return None
    parts = dataset.get("parts")
    if not isinstance(parts, dict) or not parts:
        return None
    if len(parts) == 1:
        return f"{dataset_name}.{next(iter(parts))}"
    return dataset_name


@beartype
def normalize_train_config_surface(values: Any) -> Any:
    """Expand the concise singleton authoring surface to canonical mappings."""

    if not isinstance(values, dict):
        return values
    normalized = copy.deepcopy(values)

    model = normalized.get("model")
    if isinstance(model, dict):
        backbone = model.get("backbone")
        global_training = normalized.get("global_training")
        if isinstance(backbone, dict) and isinstance(global_training, dict):
            architecture = backbone.get("architecture")
            if isinstance(architecture, dict) and "context_length" in global_training:
                architecture.setdefault(
                    "max_context_length", global_training["context_length"]
                )
        if "interface" in model and "interfaces" in model:
            raise ValueError("model cannot define both 'interface' and 'interfaces'.")
        if "interface" in model:
            model["interfaces"] = {_SINGLETON_CONFIG_NAME: model.pop("interface")}

    if "dataset" in normalized and "dataset_training" in normalized:
        raise ValueError(
            "Training config cannot define both 'dataset' and 'dataset_training'."
        )
    if "dataset" in normalized:
        normalized["dataset_training"] = {
            _SINGLETON_CONFIG_NAME: normalized.pop("dataset")
        }

    datasets = normalized.get("dataset_training")
    _normalize_dataset_part_surface(datasets)
    if isinstance(datasets, dict):
        for dataset in datasets.values():
            if isinstance(dataset, dict):
                for part in dataset.get("parts", {}).values():
                    if isinstance(part, dict):
                        part.setdefault("type", "prepared")

    interfaces = model.get("interfaces") if isinstance(model, dict) else None
    if (
        isinstance(interfaces, dict)
        and len(interfaces) == 1
        and isinstance(datasets, dict)
    ):
        interface_name = next(iter(interfaces))
        for dataset in datasets.values():
            if isinstance(dataset, dict):
                dataset.setdefault("model_interface", interface_name)

    if "policy_binding" in normalized and "policy_bindings" in normalized:
        raise ValueError("Use policy_binding or policy_bindings, not both")
    if (
        "policy_binding" in normalized
        and isinstance(interfaces, dict)
        and len(interfaces) == 1
    ):
        normalized["policy_bindings"] = {
            next(iter(interfaces)): normalized.pop("policy_binding")
        }

    implicit_source = _implicit_source(normalized)
    training_plan = normalized.get("training_plan")
    if isinstance(training_plan, dict):
        direct_fields = _PHASE_FIELD_NAMES & set(training_plan)
        if "phases" in training_plan and direct_fields:
            raise ValueError(
                "training_plan cannot combine 'phases' with direct phase fields."
            )
        if "phases" not in training_plan and direct_fields:
            policy = {
                key: training_plan[key]
                for key in _TRAINING_PLAN_POLICY_FIELDS & set(training_plan)
            }
            phase = {
                key: value
                for key, value in training_plan.items()
                if key not in _TRAINING_PLAN_POLICY_FIELDS
            }
            phase.setdefault("name", _SINGLE_PHASE_NAME)
            if phase.get("algorithm", "supervised") == "supervised":
                phase.setdefault("mode", "sequential")
            normalized["training_plan"] = {**policy, "phases": [phase]}
            training_plan = normalized["training_plan"]

        phases = training_plan.get("phases")
        if isinstance(phases, list):
            for phase in phases:
                if isinstance(phase, dict):
                    phase.setdefault("algorithm", "supervised")
                    if implicit_source is not None and phase["algorithm"] in {
                        "supervised",
                        "policy_imitation",
                        "iql",
                    }:
                        phase.setdefault("sources", [{"source": implicit_source}])

    if isinstance(training_plan, dict):
        for phase in training_plan.get("phases", []):
            if not isinstance(phase, dict) or phase.get("algorithm") != "supervised":
                continue
            evaluation = phase.get("evaluation")
            if isinstance(evaluation, bool):
                if not evaluation:
                    phase["evaluation"] = None
                else:
                    sources = phase.get("sources", [])
                    if len(sources) != 1:
                        raise ValueError(
                            "evaluation=true requires exactly one phase source"
                        )
                    phase["evaluation"] = {
                        "sources": [{"source": sources[0]["source"]}]
                    }

    return normalized


@beartype
def _only_base_name(values: Any, usage: str) -> str:
    if not isinstance(values, dict) or len(values) != 1:
        raise ValueError(
            f"The singleton {usage} override requires exactly one {usage} "
            "in the base training config."
        )
    return next(iter(values))


@beartype
def normalize_train_config_parameter_surface(
    parameters: Any,
    base_values: dict[str, Any],
) -> Any:
    """Translate singleton search parameters to canonical base paths."""

    if not isinstance(parameters, dict):
        return parameters
    normalized = copy.deepcopy(parameters)

    model_override = normalized.get("model")
    if isinstance(model_override, dict) and "interface" in model_override:
        if "interfaces" in model_override:
            raise ValueError(
                "model parameters cannot define both 'interface' and 'interfaces'."
            )
        interface_name = _only_base_name(
            base_values.get("model", {}).get("interfaces"),
            "model interface",
        )
        model_override["interfaces"] = {interface_name: model_override.pop("interface")}

    if "dataset" in normalized:
        if "dataset_training" in normalized:
            raise ValueError(
                "Training parameters cannot define both 'dataset' and "
                "'dataset_training'."
            )
        dataset_name = _only_base_name(
            base_values.get("dataset_training"),
            "dataset",
        )
        normalized["dataset_training"] = {dataset_name: normalized.pop("dataset")}

    dataset_overrides = normalized.get("dataset_training")
    base_datasets = base_values.get("dataset_training")
    if isinstance(dataset_overrides, dict):
        for dataset_name, dataset_override in dataset_overrides.items():
            if not isinstance(dataset_override, dict):
                continue
            if "part" in dataset_override and "parts" in dataset_override:
                raise ValueError(
                    f"Dataset override {dataset_name!r} cannot define both "
                    "'part' and 'parts'."
                )
            if "part" not in dataset_override:
                continue
            base_dataset = (
                base_datasets.get(dataset_name)
                if isinstance(base_datasets, dict)
                else None
            )
            part_name = _only_base_name(
                base_dataset.get("parts") if isinstance(base_dataset, dict) else None,
                f"part for dataset {dataset_name!r}",
            )
            dataset_override["parts"] = {part_name: dataset_override.pop("part")}

    plan_override = normalized.get("training_plan")
    if isinstance(plan_override, dict):
        direct_fields = _PHASE_FIELD_NAMES & set(plan_override)
        if "phases" in plan_override and direct_fields:
            raise ValueError(
                "training_plan parameters cannot combine 'phases' with direct "
                "phase fields."
            )
        if "phases" not in plan_override and direct_fields:
            base_phases = base_values.get("training_plan", {}).get("phases")
            if not isinstance(base_phases, list) or len(base_phases) != 1:
                raise ValueError(
                    "Direct training_plan parameters require exactly one phase "
                    "in the base training config."
                )
            policy = {
                key: plan_override[key]
                for key in _TRAINING_PLAN_POLICY_FIELDS & set(plan_override)
            }
            phase_override = {
                key: value
                for key, value in plan_override.items()
                if key not in _TRAINING_PLAN_POLICY_FIELDS
            }
            normalized["training_plan"] = {
                **policy,
                "phases": {0: phase_override},
            }

    return normalized


class GlobalTrainingSpecModel(BaseModel):
    """Run-wide data, precision, execution and distribution settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    episode_cache_size: int = Field(default=8, gt=0)
    metadata_config_path: str | dict[str, str] | None = None
    normalization: FrozenNormalizationConfig | None = None
    read_format: Literal["csv", "parquet", "pt"] = "parquet"
    training_objective: str | None = None
    forward_attention: Literal["causal", "bidirectional"] | None = None
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int = Field(default=1, ge=0)
    window_stride: Optional[int] = Field(default=None, gt=0)
    inference_batch_size: int = Field(default=1, gt=0)
    bert_spec: Optional[BERTSpecModel] = None
    next_occurrence_config: Optional[NextOccurrenceConfigModel] = None

    device_max_concat_length: int = Field(default=12, gt=0)
    log_interval: int = Field(default=10, gt=0)
    early_stopping_epochs: Optional[int] = Field(default=None, gt=0)
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
    data_parallelism: Optional[Literal["ddp", "fsdp"]] = None
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


class StaticConditioningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str] = Field(min_length=1)
    embedding_size: int | None = Field(default=None, gt=0)
    feature_embedding_dims: dict[str, int] | None = None

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        return _unique_columns(columns, "static_conditioning.columns")

    @model_validator(mode="after")
    def validate_embedding_dims(self):
        dims = self.feature_embedding_dims
        if dims is None:
            return self
        if set(dims) != set(self.columns):
            raise ValueError(
                "static_conditioning.feature_embedding_dims must name "
                "exactly the configured static columns"
            )
        if any(dim <= 0 for dim in dims.values()):
            raise ValueError("Static feature embedding dimensions must be positive")
        if (
            self.embedding_size is not None
            and sum(dims.values()) != self.embedding_size
        ):
            raise ValueError(
                "Static feature embedding dimensions must sum to embedding_size"
            )
        return self


class ModelInterfaceSpecModel(BaseModel):
    """Architecture and selected-column contract for one named model route."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_columns: list[str] = Field(default_factory=list)
    prediction_distributions: dict[str, PredictionDistributionConfig] = Field(
        default_factory=dict
    )
    static_conditioning: StaticConditioningConfig | None = None
    preconditioning_image_encoder: ImageEncoderConfig | None = None
    target_columns: list[str] = Field(..., min_length=1)
    categorical_hashing: dict[str, CategoricalHashingConfig] = Field(
        default_factory=dict
    )
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
        if any(
            column.startswith("__image:")
            or column.startswith("__image_mask:")
            or column == "__preconditioning_image"
            for column in value
        ):
            raise ValueError("Image tensor key prefixes are reserved")
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
        if self.prediction_distributions.keys() - set(self.target_columns):
            raise ValueError("prediction_distributions must name model targets")
        input_columns = set(self.input_columns)
        if image_targets := set(self.target_columns) & image_views(self.ingestion):
            raise ValueError(
                f"Image views cannot be target columns: {sorted(image_targets)}"
            )
        if self.static_conditioning is not None:
            static_columns = set(self.static_conditioning.columns)
            if any(
                column.startswith("__image:")
                or column.startswith("__image_mask:")
                or column == "__preconditioning_image"
                for column in static_columns
            ):
                raise ValueError("Image tensor key prefixes are reserved")
            if static_columns & (input_columns | set(self.target_columns)):
                raise ValueError(
                    "Static conditioning columns must be separate from temporal "
                    "inputs and targets"
                )
        auxiliary_columns = set(self.ingestion.auxiliary_input_columns)
        if missing := auxiliary_columns - input_columns:
            raise ValueError(
                "ingestion.auxiliary_input_columns references unknown input "
                f"columns: {sorted(missing)}"
            )
        if self.feature_layout is not None:

            def image_cell_names(ingestion):
                if ingestion.type == "structured":
                    return set(ingestion.image_cells)
                if ingestion.type == "composite":
                    return set().union(
                        *(
                            image_cell_names(child)
                            for child in ingestion.branches.values()
                        )
                    )
                return set()

            image_cells = image_cell_names(self.ingestion)
            if image_cells & (input_columns | set(self.target_columns)):
                raise ValueError(
                    "Structured image cell names must differ from tabular columns"
                )
            for layout_name, layout in self.feature_layout.items():
                if missing := set(layout.columns) - input_columns - image_cells:
                    raise ValueError(
                        f"feature_layout {layout_name!r} references unknown "
                        f"columns outside input_columns: {sorted(missing)}"
                    )
        branches = (
            self.decoder.branches.items()
            if self.decoder.type == "composite"
            else (("default", self.decoder),)
        )
        for branch_name, branch in branches:
            if branch.type != "autoregressive_transformer":
                continue
            nonfinal_hashed_targets = [
                target
                for target in branch.target_columns[:-1]
                if target in self.categorical_hashing
            ]
            if nonfinal_hashed_targets:
                raise ValueError(
                    "Hashed targets must be last in their autoregressive_transformer "
                    f"decoder branch; branch {branch_name!r} has non-final hashed "
                    f"targets {nonfinal_hashed_targets!r}."
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


class DatasetPartSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["prepared"] = "prepared"

    metadata_config_path: str
    image_path: MediaPath = None
    preconditioning_image_path: MediaPath = None
    file_order: Literal["shuffled", "name"] = "shuffled"


class SequencePartSpecModel(SequenceSourceConfig):
    type: Literal["sequence"]


class PreferencePartSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["preferences"]
    manifest: str = Field(min_length=1)


DatasetPart = Annotated[
    DatasetPartSpecModel | SequencePartSpecModel | PreferencePartSpecModel,
    Field(discriminator="type"),
]


class DatasetFreezingSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: LayerFreezingConfigFields = Field(
        default_factory=LayerFreezingConfigFields
    )
    ingestion: IngestionFreezingConfig = Field(default_factory=IngestionFreezingConfig)
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

    model_interface: str | None = None
    parts: dict[str, DatasetPart] = Field(..., min_length=1)
    criterion: dict[str, str] = Field(default_factory=dict)
    class_weights: Optional[dict[str, list[float]]] = None
    loss_weights: Optional[dict[str, float]] = None
    class_share_log_columns: list[str] = Field(default_factory=list)
    freeze: DatasetFreezingSpecModel = Field(default_factory=DatasetFreezingSpecModel)

    @field_validator("model_interface")
    @classmethod
    @beartype
    def validate_interface_name(cls, value):
        return (
            _identifier(value, "Model interface reference")
            if value is not None
            else None
        )

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
            if name != "AccurateCrossEntropyLoss" and not hasattr(torch.nn, name):
                raise ValueError(f"Criterion {name!r} not found in torch.nn")
        return value

    @field_validator("loss_weights")
    @classmethod
    @beartype
    def validate_loss_weights(cls, value):
        if value is None:
            return value
        invalid = {
            target: weight
            for target, weight in value.items()
            if not math.isfinite(weight) or weight < 0
        }
        if invalid:
            raise ValueError(
                "loss_weights must contain finite, non-negative values; "
                f"found {invalid}."
            )
        return value


class EvaluationMonitorSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    metric: str = "loss"
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
        sources = [source.source for source in self.sources]
        if len(sources) != len(set(sources)):
            raise ValueError("Evaluation sources must be unique")
        if self.monitor is not None and self.monitor.source not in sources:
            raise ValueError("evaluation.monitor.source must be an evaluation source")
        if len(sources) == 1 and self.monitor is None:
            self.monitor = EvaluationMonitorSpecModel(source=sources[0])
        return self


class TrainingPhaseSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationSpecModel | None = None
    evaluation_interval: int = Field(default=1, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)

    algorithm: Literal["supervised"] = "supervised"
    handoff: Literal["last", "best"] = "last"

    optimization_roles: ClassVar[tuple[str, ...]] = ("model",)
    optimizer_state: Literal["reset", "continue"] = "reset"
    batch_size: int = Field(default=32, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    optimization: dict[Literal["model"], SupervisedOptimizationConfig] = Field(
        default_factory=lambda: {"model": SupervisedOptimizationConfig()}
    )

    def optimization_configs(self):
        return {"model": self.optimization.get("model", SupervisedOptimizationConfig())}

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


TrainingPhase = Annotated[
    TrainingPhaseSpecModel
    | PolicyImitationPhaseConfig
    | IQLPhaseConfig
    | PPOPhaseConfig
    | DPOPhaseConfig
    | GRPOPhaseConfig,
    Field(discriminator="algorithm"),
]


class TrainingPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    curriculum_training: bool = False
    curriculum_column: Optional[str] = Field(default=None, min_length=1)
    phases: list[TrainingPhase] = Field(..., min_length=1)

    @model_validator(mode="after")
    @beartype
    def validate_unique_names(self):
        if self.curriculum_column is not None and not self.curriculum_training:
            raise ValueError(
                "curriculum_column may only be set when curriculum_training is true"
            )
        if self.curriculum_training and self.curriculum_column is None:
            raise ValueError(
                "curriculum_column must be set when curriculum_training is true"
            )
        names = [phase.name for phase in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("Training phase names must be unique")
        mixed = len({phase.algorithm == "supervised" for phase in self.phases}) > 1
        for index, phase in enumerate(self.phases):
            if phase.optimizer_state != "continue":
                continue
            if index == 0:
                raise ValueError("the first phase cannot continue optimizer state")
            previous = self.phases[index - 1]
            if mixed:
                raise ValueError(
                    "mixed-plan coordination cannot preserve optimizer state"
                )
            if previous.algorithm != phase.algorithm or previous.handoff != "last":
                raise ValueError(
                    "optimizer continuation requires the same algorithm and a last handoff"
                )
            if phase.algorithm == "supervised" and (
                {ref.source.partition(".")[0] for ref in previous.sources}
                != {ref.source.partition(".")[0] for ref in phase.sources}
            ):
                raise ValueError(
                    "supervised optimizer continuation requires identical dataset ownership"
                )
            if getattr(previous, "freeze", None) != getattr(phase, "freeze", None):
                raise ValueError(
                    "optimizer continuation requires identical parameter selection"
                )
            if {
                role: value.continuation_settings()
                for role, value in previous.optimization_configs().items()
            } != {
                role: value.continuation_settings()
                for role, value in phase.optimization_configs().items()
            }:
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


class SequifierConfig(BaseModel):
    """Training configuration with singleton authoring normalization."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    model_name: str
    device: str
    seed: int = 1010
    global_training: GlobalTrainingSpecModel = Field(
        default_factory=GlobalTrainingSpecModel
    )
    model: ModelSpecModel | None = None
    initialize_from: str | None = None
    policy_bindings: dict[str, PolicyBinding] = Field(default_factory=dict)
    policy_binding: PolicyBinding | None = Field(default=None, exclude=True)
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""
    dataset_training: dict[str, DatasetTrainingSpecModel] = Field(default_factory=dict)
    training_plan: TrainingPlanModel

    export_generative_model: bool = True
    export_embedding_model: bool = False
    embedding_layer_names: list[StrictStr] = Field(
        default_factory=lambda: ["backbone.final_norm"], min_length=1
    )
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    @model_validator(mode="before")
    @classmethod
    @beartype
    def normalize_singleton_surface(cls, values):
        return normalize_train_config_surface(values)

    @field_validator("dataset_training")
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
        if self.model is None and self.initialize_from is None:
            raise ValueError("a run requires model or initialize_from")
        for phase in self.training_plan.phases:
            evaluation = phase.evaluation
            refs = list(getattr(phase, "sources", [])) + list(
                getattr(evaluation, "sources", None) or []
            )
            preference = getattr(phase, "preferences", None)
            if preference is not None:
                refs.append(preference)
            for ref in refs:
                self._validate_source(ref.source, f"phase {phase.name!r}")
                if phase.algorithm != "supervised" and (
                    ref.weight is not None or ref.batches_per_selection is not None
                ):
                    raise ValueError(
                        "policy source references cannot configure supervised interleaving"
                    )
            if phase.algorithm != "supervised":
                continue
            role = phase.optimization_configs()["model"]
            needs_monitor = (
                phase.handoff == "best"
                or role.scheduler.name == "ReduceLROnPlateau"
                or self.global_training.early_stopping_epochs is not None
                or (
                    evaluation is not None
                    and self.global_training.save_interval_val_loss
                )
            )
            if needs_monitor and (evaluation is None or evaluation.monitor is None):
                raise ValueError(
                    f"{phase.name}: best handoff, plateau scheduling, validation saving, and early stopping require evaluation.monitor"
                )
            if (
                role.scheduler.name == "ReduceLROnPlateau"
                and phase.evaluation_interval != 1
            ):
                raise ValueError("ReduceLROnPlateau requires evaluation_interval: 1")
            if (
                self.global_training.early_stopping_epochs is not None
                and phase.evaluation_interval != 1
            ):
                raise ValueError(
                    "early_stopping_epochs requires evaluation_interval: 1"
                )
        if any(phase.algorithm != "supervised" for phase in self.training_plan.phases):
            if self.global_training.distributed:
                raise ValueError("policy phases currently require local execution")
            return self
        if self.initialize_from is not None:
            return (
                self  # Model-contract assertions are checked after artifact resolution.
            )
        model = self.model
        assert model is not None
        if (
            self.global_training.training_objective is None
            or self.global_training.context_length is None
        ):
            raise ValueError(
                "supervised training requires training_objective and context_length"
            )
        for interface in model.interfaces.values():
            validate_embedding_layer_names(
                self.embedding_layer_names,
                SimpleNamespace(
                    backbone=model.backbone,
                    decoder=interface.decoder,
                ),
            )

        referenced_interfaces = set()
        for dataset_name, dataset in self.dataset_training.items():
            if dataset.model_interface not in model.interfaces:
                raise ValueError(
                    f"Dataset {dataset_name!r} references unknown model interface "
                    f"{dataset.model_interface!r}."
                )
            referenced_interfaces.add(dataset.model_interface)
            interface = model.interfaces[dataset.model_interface]
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
            if dataset.loss_weights is not None and all(
                dataset.loss_weights.get(target, 1.0) == 0.0
                for target in interface.target_columns
            ):
                raise ValueError(
                    f"Dataset {dataset_name!r} must have at least one target with "
                    "a positive loss weight."
                )
            if dataset.class_weights is not None and set(dataset.class_weights) - set(
                interface.target_columns
            ):
                raise ValueError(
                    f"Dataset {dataset_name!r} class_weights references unknown targets."
                )
        unreferenced = set(model.interfaces) - referenced_interfaces
        if unreferenced:
            warnings.warn(
                f"Unreferenced model interfaces: {sorted(unreferenced)}",
                stacklevel=2,
            )

        for phase in self.training_plan.phases:
            for source in phase.sources:
                self._validate_source(source.source, f"training phase {phase.name!r}")
        objective = get_objective_class(self.global_training.training_objective)
        is_bert = issubclass(objective, BERTObjective)
        is_next = issubclass(objective, NextOccurrenceObjective)
        if (self.global_training.bert_spec is not None) != is_bert:
            raise ValueError("bert_spec must be configured exactly for BERT training")
        if (self.global_training.next_occurrence_config is not None) != is_next:
            raise ValueError(
                "next_occurrence_config must be configured exactly for "
                "next_occurrence training"
            )
        if (
            len(self.dataset_training) > 1
            and self.global_training.data_parallelism == "fsdp"
        ):
            raise ValueError("Multi-dataset training does not support FSDP")

        context_length = self.global_training.context_length
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
    def _validate_source(self, source: str, usage: str) -> None:
        dataset_name, _, part_name = source.partition(".")
        dataset = self.dataset_training.get(dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown {usage} dataset source {dataset_name!r}")
        if part_name and part_name not in dataset.parts:
            raise ValueError(f"Unknown {usage} part source {source!r}")


class ResolvedDatasetPart(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    metadata_config_path: str
    metadata: DatasetMetadata
    image_path: MediaPath = None
    preconditioning_image_path: MediaPath = None
    training_data_path: str
    validation_data_path: Optional[str] = None
    storage_form: Literal["file", "folder"]
    file_order: Literal["shuffled", "name"] = "shuffled"


class ResolvedModelInterface(BaseModel):
    policy: ResolvedPolicyBinding | None = None
    prediction_distributions: dict[str, PredictionDistributionConfig] = Field(
        default_factory=dict
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    depth_layouts: DepthLayoutRegistryModel = Field(
        default_factory=DepthLayoutRegistryModel
    )
    tensor_payload_version: int = 1
    split_paths: list[str] = Field(default_factory=list)
    prediction_aligned_splits: list[int] = Field(default_factory=list)
    prediction_alignment_steps: int = 1
    window_lengths: Optional[list[int]] = None
    prediction_length: Optional[int] = None
    target_offset: Optional[int] = None
    name: str
    input_columns: list[str]
    static_conditioning: StaticConditioningConfig | None = None
    static_features_path: str | None = None
    image_sources: dict[str, ImageSourceConfig] = Field(default_factory=dict)
    preconditioning_image: PreconditioningImageConfig | None = None
    preconditioning_image_encoder: ImageEncoderConfig | None = None
    static_categorical_columns: list[str] = Field(default_factory=list)
    static_real_columns: list[str] = Field(default_factory=list)
    target_columns: list[str]
    categorical_hashing: dict[str, CategoricalHashingConfig] = Field(
        default_factory=dict
    )
    categorical_hash_contracts: dict[str, dict[str, Any]] = Field(default_factory=dict)
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

    @model_validator(mode="after")
    def validate_prediction_targets(self):
        for name in self.prediction_distributions:
            if (
                name not in self.target_columns
                or self.target_column_types.get(name) != "real"
            ):
                raise ValueError(
                    f"{name}: prediction_distributions requires a real model target"
                )
        return self


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
    freeze: DatasetFreezingSpecModel


class ResolvedTrainingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    dataset: str
    part: Optional[str] = None
    weight: float = 1.0
    batches_per_selection: int = 1


class ResolvedTrainingPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_sources: list[ResolvedTrainingSource] = Field(default_factory=list)
    evaluation_monitor: EvaluationMonitorSpecModel | None = None
    evaluation_interval: int = Field(default=1, gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)

    algorithm: Literal["supervised"] = "supervised"

    handoff: Literal["last", "best"] = "last"
    optimizer_state: Literal["reset", "continue"] = "reset"
    batch_size: int = Field(default=32, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    optimization: dict[str, SupervisedOptimizationConfig]

    name: str
    epochs: int
    mode: Literal["sequential", "interleaved"]
    selection: Literal["round_robin", "weighted_random"] = "round_robin"
    sources: list[ResolvedTrainingSource]


def resolve_supervised_phases(authored_phases):
    """Resolve role defaults and epoch budgets without accessing datasets."""
    phases = [
        ResolvedTrainingPhase(
            name=phase.name,
            epochs=phase.epochs,
            evaluation_sources=[
                _source(ref.source, 1) for ref in phase.evaluation.sources
            ]
            if phase.evaluation is not None
            else [],
            evaluation_monitor=phase.evaluation.monitor
            if phase.evaluation is not None
            else None,
            evaluation_interval=phase.evaluation_interval,
            checkpoint_interval=phase.checkpoint_interval,
            handoff=phase.handoff,
            optimizer_state=phase.optimizer_state,
            batch_size=phase.batch_size,
            accumulation_steps=phase.accumulation_steps,
            optimization={
                role: value.model_copy(deep=True)
                for role, value in phase.optimization_configs().items()
            },
            mode=phase.mode,
            selection=phase.selection or "round_robin",
            sources=[
                _source(
                    source.source,
                    phase.accumulation_steps,
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
        for phase in authored_phases
    ]
    # A continued epoch scheduler spans the entire contiguous ownership chain.
    for index, phase in enumerate(phases):
        role = phase.optimization["model"]
        if role.scheduler.name != "OneCycleLR" or role.scheduler_step_on != "epoch":
            continue
        start = index
        while start > 0 and phases[start].optimizer_state == "continue":
            start -= 1
        end = index + 1
        while end < len(phases) and phases[end].optimizer_state == "continue":
            end += 1
        budget = sum(item.epochs for item in phases[start:end])
        explicit = role.scheduler.arguments.get("total_steps")
        if explicit is not None and explicit != budget:
            raise ValueError(
                f"{phase.name}: OneCycleLR.total_steps must equal epoch budget {budget}"
            )
        arguments = {**role.scheduler.arguments, "total_steps": budget}
        role.scheduler = role.scheduler.model_copy(update={"arguments": arguments})
    return phases


class ResolvedSequifierConfig(BaseModel):
    """Runtime configuration after all dataset parts and interfaces resolve."""

    _prepared_inputs: Any = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    model_name: str
    device: str
    seed: int
    global_training: GlobalTrainingSpecModel
    model: ModelSpecModel
    dataset_training: dict[str, ResolvedDatasetTrainingSpec]
    authored: SequifierConfig | None = Field(default=None, exclude=True)
    initialize_from: str | None = None
    model_path: str = ""
    report_path: str = ""
    phase_directory: str = ""
    sequence_sources: dict[str, SequenceSourceConfig] = Field(default_factory=dict)
    preference_sources: dict[str, str] = Field(default_factory=dict)
    policy_bindings: dict[str, ResolvedPolicyBinding] = Field(default_factory=dict)
    training_plan: list[ResolvedTrainingPhase | PolicyPhase]
    curriculum_training: bool = False
    curriculum_column: Optional[str] = None

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

    export_generative_model: bool
    export_embedding_model: bool
    embedding_layer_names: list[str]
    export_onnx: bool
    export_pt: bool
    export_with_dropout: bool = False

    @model_validator(mode="after")
    @beartype
    def validate_next_occurrence_metadata(self):
        objective = get_objective_class(self.global_training.training_objective)
        if not issubclass(objective, NextOccurrenceObjective):
            return self

        next_config = self.global_training.next_occurrence_config
        if next_config is None:
            raise ValueError(
                "next_occurrence_config must be set for next_occurrence training"
            )
        column = next_config.column_name
        for dataset_name, dataset in self.dataset_training.items():
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
        for dataset in self.dataset_training.values():
            if dataset.model_interface in validated_interfaces:
                continue
            if dataset.interface.depth_layouts:
                if self.global_training.read_format != "pt":
                    raise ValueError(
                        "Preprocessed depth inputs require read_format: pt"
                    )
            view = interface_build_view(self, dataset.interface)
            resolve_ingestion_plan(view)
            resolve_decoding_plan(view)
            validated_interfaces.add(dataset.model_interface)
        return self

    @property
    @beartype
    def dataset_count(self) -> int:
        return len(self.dataset_training)

    @property
    @beartype
    def interface_names(self) -> tuple[str, ...]:
        return tuple(self.model.interfaces)

    @beartype
    def dataset(self, name: Optional[str] = None) -> ResolvedDatasetTrainingSpec:
        if name is None:
            if len(self.dataset_training) != 1:
                raise ValueError("A dataset selection is required")
            return next(iter(self.dataset_training.values()))
        return self.dataset_training[name]

    @beartype
    def interface(self, name: Optional[str] = None) -> ResolvedModelInterface:
        if name is None:
            names = {
                dataset.model_interface for dataset in self.dataset_training.values()
            }
            if len(names) != 1:
                raise ValueError("A model interface selection is required")
            name = next(iter(names))
        for dataset in self.dataset_training.values():
            if dataset.model_interface == name:
                return dataset.interface
        raise KeyError(name)


@dataclass(frozen=True)
class LoadedTrainConfig:
    config: SequifierConfig
    resolved: ResolvedSequifierConfig
    metadata: dict[str, DatasetMetadata]


@beartype
def _source(
    source: str, accumulation_steps: Optional[int], **values: Any
) -> ResolvedTrainingSource:
    dataset, _, part = source.partition(".")
    return ResolvedTrainingSource(
        source=source,
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
    if not value.exists():
        raise ValueError(f"Training data path {path!r} does not exist")
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
def _evaluated_parts(config: SequifierConfig) -> set[str]:
    selected: set[str] = set()
    sources = [
        ref
        for phase in config.training_plan.phases
        if phase.algorithm == "supervised" and phase.evaluation is not None
        for ref in phase.evaluation.sources
    ]
    for source in sources:
        dataset_name, _, part_name = source.source.partition(".")
        if part_name:
            selected.add(source.source)
        else:
            selected.update(
                f"{dataset_name}.{name}"
                for name in config.dataset_training[dataset_name].parts
            )
    return selected


@beartype
def _part_signature(
    metadata: DatasetMetadata, interface: ModelInterfaceSpecModel
) -> dict[str, Any]:
    static_columns = (
        interface.static_conditioning.columns
        if interface.static_conditioning is not None
        else []
    )
    if static_columns and (
        set(static_columns) - set(metadata.static_columns)
        or metadata.static_features_path is None
    ):
        raise ValueError("Metadata lacks the configured static conditioning data")
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
        "depth_layouts": metadata.depth_layouts.compatibility_signature(relevant),
        "image_sources": {
            name: metadata.image_sources[name].model_dump(
                exclude={
                    "path",
                    "signature",
                    "time_index_path",
                    "time_index_signature",
                    "video_index_paths",
                }
            )
            for name in sorted(selected_views)
        },
        "preconditioning_image": (
            metadata.preconditioning_image.compatibility_signature()
            if interface.preconditioning_image_encoder is not None
            and metadata.preconditioning_image is not None
            else None
        ),
        "storage_layout": metadata.storage_layout,
        "prediction_aligned_splits": metadata.prediction_aligned_splits,
        "prediction_alignment_steps": metadata.prediction_alignment_steps,
        "window_lengths": metadata.window_lengths,
        "prediction_length": metadata.prediction_length,
        "target_offset": metadata.target_offset,
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
    storage_layout: StoredWindowLayout,
) -> ResolvedModelInterface:
    if global_spec.training_objective is None or global_spec.context_length is None:
        raise ValueError(
            "resolved training requires training_objective and context_length"
        )
    if any(metadata.depth_layouts.is_deep_column(c) for c in spec.target_columns):
        raise ValueError("Depth targets are not supported")
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
    n_classes = {
        column: metadata.n_classes[column]
        for column in set(categorical_columns)
        | set(static_categorical_columns)
        | categorical_targets
    }
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
    resolve_window_view(storage_layout, window_view)
    return ResolvedModelInterface(
        name=name,
        policy=None,
        input_columns=spec.input_columns,
        prediction_distributions=spec.prediction_distributions,
        static_conditioning=spec.static_conditioning,
        static_features_path=metadata.static_features_path,
        image_sources={
            name: metadata.image_sources[name]
            for name in sorted(image_views(spec.ingestion))
        },
        preconditioning_image=(
            metadata.preconditioning_image
            if spec.preconditioning_image_encoder is not None
            else None
        ),
        preconditioning_image_encoder=spec.preconditioning_image_encoder,
        static_categorical_columns=static_categorical_columns,
        static_real_columns=static_real_columns,
        target_columns=spec.target_columns,
        categorical_hashing=hashing,
        categorical_hash_contracts=hash_contracts,
        target_column_types=target_types,
        column_data_types=signature["column_data_types"],
        categorical_columns=categorical_columns,
        real_columns=real_columns,
        categorical_decoder_special_tokens={
            column: list(tokens)
            for column, tokens in categorical_decoder_special_tokens.items()
        },
        depth_layouts=metadata.depth_layouts.relevant_layouts(spec.input_columns),
        tensor_payload_version=metadata.tensor_payload_version,
        split_paths=metadata.split_paths,
        prediction_aligned_splits=metadata.prediction_aligned_splits,
        prediction_alignment_steps=metadata.prediction_alignment_steps,
        window_lengths=metadata.window_lengths,
        prediction_length=metadata.prediction_length,
        target_offset=metadata.target_offset,
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
            for column in dict.fromkeys(
                real_columns
                + static_real_columns
                + [
                    target
                    for target in spec.target_columns
                    if target_types[target] == "real"
                ]
            )
            if column in metadata.selected_columns_statistics
        },
        normalize_real_columns=metadata.normalize_real_columns,
        target_decoder_ids=target_decoder_ids,
        target_n_classes=target_n_classes,
        target_global_to_decoder=target_global_to_decoder,
        storage_layout=storage_layout,
        window_view=window_view,
    )


@beartype
def _interface_semantics(interface: ResolvedModelInterface) -> dict[str, Any]:
    values = interface.model_dump(
        mode="python",
        exclude={
            "name",
            "ingestion",
            "decoder",
            "feature_layout",
            "tensor_payload_version",
            "split_paths",
            "prediction_aligned_splits",
            "prediction_alignment_steps",
            "window_lengths",
            "prediction_length",
            "target_offset",
            "static_features_path",
            "image_sources",
            "preconditioning_image",
        },
    )
    values["depth_layouts"] = interface.depth_layouts.compatibility_signature(
        interface.input_columns
    )
    values["image_sources"] = {
        name: source.model_dump(
            exclude={
                "path",
                "signature",
                "time_index_path",
                "time_index_signature",
                "video_index_paths",
            }
        )
        for name, source in interface.image_sources.items()
    }
    values["preconditioning_image"] = (
        interface.preconditioning_image.compatibility_signature()
        if interface.preconditioning_image is not None
        else None
    )
    return values


@beartype
def resolve_sequifier_config(
    config: SequifierConfig,
    metadata: DatasetMetadata | dict[str, DatasetMetadata],
    *,
    part_overrides: Optional[dict[str, dict[str, str]]] = None,
) -> ResolvedSequifierConfig:
    """Resolve every dataset part, compatibility contract, and source."""

    if config.model is None:
        raise ValueError("resolved training requires a model")
    if (
        config.global_training.context_length is None
        or config.global_training.training_objective is None
    ):
        raise ValueError(
            "resolved training requires training_objective and context_length"
        )

    part_refs = [
        f"{dataset_name}.{part_name}"
        for dataset_name, dataset in config.dataset_training.items()
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

    for dataset_name, dataset_spec in config.dataset_training.items():
        if dataset_spec.model_interface is None:
            raise ValueError(f"Dataset {dataset_name!r} requires a model interface")
        interface_spec = config.model.interfaces[dataset_spec.model_interface]
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
            if validation_path is not None and not Path(validation_path).exists():
                raise ValueError(
                    f"Validation data path {validation_path!r} does not exist"
                )
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
                image_path=part_spec.image_path,
                preconditioning_image_path=part_spec.preconditioning_image_path,
                training_data_path=training_path,
                validation_data_path=validation_path,
                storage_form=form,
                file_order=part_spec.file_order,
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
        view = ModelWindowView(
            context_length=config.global_training.context_length,
            objective=config.global_training.training_objective,
            target_offset=target_offset_for_objective(
                config.global_training.training_objective,
                config.global_training.target_offset,
            ),
        )
        first_training_layout: StoredWindowLayout | None = None
        for part_name, resolved_part in resolved_parts.items():
            ref = f"{dataset_name}.{part_name}"
            selected_paths = [resolved_part.training_data_path]
            if ref in evaluated and resolved_part.validation_data_path is not None:
                selected_paths.append(resolved_part.validation_data_path)
            for path in selected_paths:
                layout = _layout_for_part_path(
                    resolved_part.metadata, path, config.project_root
                )
                resolve_window_view(layout, view)
                if first_training_layout is None:
                    first_training_layout = layout
        assert first_training_layout is not None
        interface = _resolve_interface(
            dataset_spec.model_interface,
            interface_spec,
            first_metadata,
            config.global_training,
            first_training_layout,
        )
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
                if aligned_metadata.width_for_split(index) < required_aligned_width:
                    raise ValueError(
                        f"Prediction aligned split {index} for {ref!r} requires "
                        f"window width at least {required_aligned_width}."
                    )
            normalized_splits = [
                normalize_path(path, config.project_root)
                for path in aligned_metadata.split_paths
            ]
            aligned_paths = {
                normalized_splits[index]
                for index in aligned_metadata.prediction_aligned_splits
            }
            used_paths = {resolved_part.training_data_path}
            if ref in evaluated and resolved_part.validation_data_path is not None:
                used_paths.add(resolved_part.validation_data_path)
            if not aligned_paths & used_paths:
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

        for target, criterion in dataset_spec.criterion.items():
            if (
                criterion == "AccurateCrossEntropyLoss"
                and interface.target_column_types[target] != "categorical"
            ):
                raise ValueError(
                    f"AccurateCrossEntropyLoss requires categorical target {target!r}."
                )

        if dataset_spec.class_weights is not None:
            for column, weights in dataset_spec.class_weights.items():
                if column in interface.categorical_hashing:
                    raise ValueError(
                        f"class_weights[{column!r}] describes canonical classes, but hash heads "
                        "predict buckets; no unambiguous class-weight conversion is available."
                    )
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

        for column in interface.categorical_hashing:
            if column in interface.target_columns and dataset_spec.criterion[
                column
            ] not in {"CrossEntropyLoss", "AccurateCrossEntropyLoss"}:
                raise ValueError(
                    f"Hashed target {column!r} requires CrossEntropyLoss or "
                    "AccurateCrossEntropyLoss with class-index labels."
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
            freeze=dataset_spec.freeze,
        )

    phases = resolve_supervised_phases(config.training_plan.phases)
    resolved = ResolvedSequifierConfig(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        global_training=config.global_training,
        model=config.model,
        dataset_training=resolved_datasets,
        training_plan=phases,
        curriculum_training=config.training_plan.curriculum_training,
        curriculum_column=config.training_plan.curriculum_column,
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
    "depth_layouts",
    "image_sources",
    "preconditioning_image",
    "tensor_payload_version",
    "metadata_by_part",
    "column_data_types",
    "column_types",
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
    "stored_window_layout_version",
    "storage_layout",
    "split_paths",
    "prediction_aligned_splits",
    "prediction_alignment_steps",
    "prediction_length",
    "target_offset",
}


@beartype
def _override_part(config: SequifierConfig) -> str:
    if len(config.dataset_training) != 1:
        raise ValueError(
            "Dataset-sensitive legacy CLI overrides require exactly one dataset"
        )
    dataset_name, dataset = next(iter(config.dataset_training.items()))
    if len(dataset.parts) == 1:
        return f"{dataset_name}.{next(iter(dataset.parts))}"
    refs = {
        source.source
        for phase in config.training_plan.phases
        for source in phase.sources
        if source.source.startswith(f"{dataset_name}.")
    }
    refs.update(
        source.source
        for phase in config.training_plan.phases
        if phase.algorithm == "supervised" and phase.evaluation is not None
        for source in phase.evaluation.sources
        if source.source.startswith(f"{dataset_name}.")
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
        window_length = layout.window_length
        max_target_offset = layout.max_target_offset
        layout_version = layout.version
    elif isinstance(layout, dict):
        window_length = layout.get("window_length")
        max_target_offset = layout.get("max_target_offset", 1)
        layout_version = layout.get("version", 2)
    else:
        window_length = values.get("window_length")
        max_target_offset = values.get("max_target_offset", 1)
        layout_version = values.get("stored_window_layout_version", 2)
    if window_length is None:
        if global_spec.context_length is None:
            raise ValueError("inline metadata requires context_length or window_length")
        window_length = global_spec.context_length + max(1, global_spec.target_offset)
    return DatasetMetadata.model_validate(
        {
            "depth_layouts": values.get("depth_layouts", {}),
            "tensor_payload_version": values.get("tensor_payload_version", 1),
            "split_paths": values.get("split_paths", []),
            "column_data_types": values.get(
                "column_data_types", values.get("column_types", {})
            ),
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
    from sequifier.config.composition import merge_config_fragments

    overrides = {
        key: args.pop(key)
        for key in ("global_training", "training_plan")
        if key in args
    }
    raw = merge_config_fragments((raw, overrides))
    if args:
        raise ValueError(f"Unsupported training CLI overrides: {sorted(args)}")

    config = try_catch_excess_keys(config_path, SequifierConfig, raw)
    if (
        config.initialize_from is not None
        or config.policy_bindings
        or config.policy_binding is not None
        or any(phase.algorithm != "supervised" for phase in config.training_plan.phases)
    ):
        if sensitive or metadata_inline_values or skip_metadata:
            raise ValueError(
                "policy/mixed plans require source declarations instead of inline metadata overrides"
            )
        from sequifier.config.training_resolution import resolve_training_run

        return LoadedTrainConfig(
            config=config, resolved=resolve_training_run(config), metadata={}
        )
    part_overrides: dict[str, dict[str, str]] = {}
    if sensitive:
        selected_part = _override_part(config)
        dataset_name = selected_part.partition(".")[0]
        if "input_columns" in sensitive:
            interface_name = config.dataset_training[dataset_name].model_interface
            config.model.interfaces[interface_name].input_columns = sensitive.pop(
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
        for dataset_name, dataset in config.dataset_training.items()
        for part_name, part in dataset.parts.items()
    ]
    if skip_metadata:
        if len(all_part_refs) != 1 and "metadata_by_part" not in metadata_inline_values:
            raise ValueError(
                "skip_metadata with multiple parts requires metadata_by_part"
            )
        if "metadata_by_part" in metadata_inline_values:
            for ref, values in metadata_inline_values["metadata_by_part"].items():
                metadata_by_part[ref] = _inline_metadata(values, config.global_training)
        else:
            dataset_name, part_name, _ = all_part_refs[0]
            metadata_by_part[f"{dataset_name}.{part_name}"] = _inline_metadata(
                metadata_inline_values, config.global_training
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
    global_training: GlobalTrainingSpecModel
    training_objective: str
    input_columns: list[str]
    static_columns: list[str]
    static_features_path: str | None
    image_sources: dict[str, ImageSourceConfig]
    preconditioning_image: PreconditioningImageConfig | None
    preconditioning_image_encoder: ImageEncoderConfig | None
    target_columns: list[str]
    categorical_hashing: dict[str, CategoricalHashingConfig]
    categorical_hash_contracts: dict[str, dict[str, Any]]
    target_column_types: dict[str, str]
    column_data_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    categorical_decoder_special_tokens: dict[str, list[str]]
    feature_layout: Optional[FeatureLayoutRegistryModel]
    depth_layouts: DepthLayoutRegistryModel
    tensor_payload_version: int
    n_classes: dict[str, int]
    id_maps: dict[str, dict[str | int, int]]
    special_token_ids: dict[str, int]
    target_decoder_ids: dict[str, list[int]]
    target_n_classes: dict[str, int]
    target_global_to_decoder: dict[str, list[int]]
    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    model: SelectedModelComponents


@dataclass(frozen=True)
class SelectedDatasetPartConfig(SelectedInterfaceConfig):
    """Explicit dataset-part configuration used by data loaders."""

    batch_size: int
    dataset_training: ResolvedDatasetTrainingSpec
    part: ResolvedDatasetPart
    curriculum_training: bool
    curriculum_column: Optional[str]
    epochs: int
    read_format: str
    window_stride: Optional[int]
    data_path: str
    validation_data_path: Optional[str]
    metadata_config_path: str


@beartype
def interface_build_view(
    config: ResolvedSequifierConfig, interface: ResolvedModelInterface
) -> SelectedInterfaceConfig:
    """Select one interface for model and objective construction."""

    return SelectedInterfaceConfig(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        global_training=config.global_training,
        training_objective=config.global_training.training_objective,
        static_columns=(
            interface.static_conditioning.columns
            if interface.static_conditioning
            else []
        ),
        **{
            name: getattr(interface, name)
            for name in SelectedInterfaceConfig.__dataclass_fields__
            if name in ResolvedModelInterface.model_fields
        },
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
) -> SelectedDatasetPartConfig:
    """Select one dataset part for data loading."""

    dataset = config.dataset_training[dataset_name]
    part = dataset.parts[part_name]
    interface_config = interface_build_view(config, dataset.interface)
    interface_values = vars(interface_config).copy()
    interface_values["static_features_path"] = part.metadata.static_features_path
    selected_sources = {
        name: part.metadata.image_sources[name]
        for name in interface_config.image_sources
    }
    selected_prefix = (
        part.metadata.preconditioning_image
        if interface_config.preconditioning_image_encoder is not None
        else None
    )
    interface_values["image_sources"], interface_values["preconditioning_image"] = (
        bind_image_paths(
            selected_sources,
            selected_prefix,
            part.image_path,
            part.preconditioning_image_path,
        )
    )

    return SelectedDatasetPartConfig(
        batch_size=config.training_plan[0].batch_size
        if config.training_plan
        else config.global_training.inference_batch_size,
        **interface_values,
        dataset_training=dataset,
        part=part,
        curriculum_training=config.curriculum_training,
        curriculum_column=config.curriculum_column,
        epochs=sum(phase.epochs for phase in config.training_plan),
        read_format=config.global_training.read_format,
        window_stride=config.global_training.window_stride,
        data_path=part.training_data_path,
        validation_data_path=part.validation_data_path,
        metadata_config_path=part.metadata_config_path,
    )


TrainModel = ResolvedSequifierConfig
