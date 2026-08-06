import copy
import json
import math
import os
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, Union, cast

from beartype import beartype
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sequifier.artifacts.backbone_repository import architecture_fingerprint
from sequifier.config.composition import load_composed_yaml_config
from sequifier.config.initialization_config import (
    ModelInitializationConfig,
    ModelInitializationSamplingConfig,
)
from sequifier.config.metadata import DatasetMetadata
from sequifier.config.probabilities import ProbabilityDistribution
from sequifier.config.train_config import (
    BERTSpecModel,
    DecodingSpecConfig,
    DotDict,
    FeatureLayoutRegistryModel,
    IngestionMergeConfig,
    IngestionSpecConfig,
    ModelSpecModel,
    NextOccurrenceConfigModel,
    ReplacementDistribution,
    ResolvedSequifierConfig,
    ResumeConfig,
    SequifierConfig,
    TrainingSpecModel,
    resolve_sequifier_config,
)
from sequifier.helpers import (
    StoredWindowLayout,
    normalize_path,
    stored_window_layout_from_metadata,
    try_catch_excess_keys,
)
from sequifier.model.embedding import DEFAULT_EMBEDDING_LAYER_NAMES
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    BERTObjective,
    NextOccurrenceObjective,
    forward_objective_names,
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids


class FloatDistribution(BaseModel):
    """Optuna float range with optional step/log sampling."""

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    step: Optional[float] = None
    log: bool = False

    @model_validator(mode="after")
    def validate_step_and_log(self):
        if self.low > self.high:
            raise ValueError(
                f"distribution low must be <= high, got {self.low} > {self.high}"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"distribution step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log distributions require low > 0, got low={self.low}")
        if self.log and self.step is not None:
            raise ValueError(
                f"Optuna does not support setting step when log=True. "
                f"Got step={self.step} and log={self.log}."
            )
        return self


class IntDistribution(BaseModel):
    """Optuna integer range with step/log sampling."""

    model_config = ConfigDict(extra="forbid")

    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def validate_step_and_log(self):
        if self.low > self.high:
            raise ValueError(
                f"distribution low must be <= high, got {self.low} > {self.high}"
            )
        if self.step <= 0:
            raise ValueError(f"distribution step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log distributions require low > 0, got low={self.low}")
        if self.log and self.step != 1:
            raise ValueError(
                f"Optuna does not support setting step != 1 when log=True. "
                f"Got step={self.step} and log={self.log}."
            )
        return self


OptunaFloat = Union[list[float], FloatDistribution]
OptunaInt = Union[list[int], IntDistribution]
OptionalOptunaInt = Union[list[Optional[int]], IntDistribution]
OptionalOptunaFloat = Union[float, list[Optional[float]], FloatDistribution, None]
_DEFAULT_KV_HEADS = object()


def _validation_space_values(
    name: str,
    space: Union[list, FloatDistribution, IntDistribution],
) -> list:
    """Return every categorical value or the extrema of a numeric range."""
    if isinstance(space, list):
        if not space:
            raise ValueError(f"{name} candidates cannot be empty")
        return space
    if isinstance(space, IntDistribution):
        step = space.step
        if step is None:
            raise ValueError(f"{name} integer distribution step cannot be None")
        sampled_high = space.low + ((space.high - space.low) // step) * step
        return [space.low] if space.low == sampled_high else [space.low, sampled_high]
    if isinstance(space, FloatDistribution):
        sampled_high = space.high
        if space.step is not None:
            low = Decimal(str(space.low))
            high = Decimal(str(space.high))
            step = Decimal(str(space.step))
            sampled_high = float(low + ((high - low) // step) * step)
        return [space.low] if space.low == sampled_high else [space.low, sampled_high]
    raise TypeError(f"Unsupported hyperparameter search space for {name}: {space}")


def sample_param(
    trial: Any,
    name: str,
    space: Union[list, FloatDistribution, IntDistribution],
):
    if isinstance(space, list):
        return trial.suggest_categorical(name, space)
    if isinstance(space, FloatDistribution):
        return trial.suggest_float(
            name, space.low, space.high, step=space.step, log=space.log
        )
    if isinstance(space, IntDistribution):
        return trial.suggest_int(
            name, space.low, space.high, step=space.step, log=space.log
        )
    raise TypeError(f"Unsupported hyperparameter search space for {name}: {space}")


def grid_space_size(
    name: str, space: Union[list, FloatDistribution, IntDistribution]
) -> int:
    """Return the number of discrete values in an Optuna search space."""
    if isinstance(space, list):
        return len(space)
    if isinstance(space, IntDistribution):
        step = space.step
        if step is None:
            raise ValueError(f"{name}.step cannot be null for an integer distribution.")
        return int((space.high - space.low) // step + 1)
    if isinstance(space, FloatDistribution):
        if space.step is None:
            raise ValueError(
                f"{name}.step must be configured for grid search because an "
                "unstepped float distribution has infinitely many combinations."
            )
        low = Decimal(str(space.low))
        high = Decimal(str(space.high))
        step = Decimal(str(space.step))
        return int((high - low) // step) + 1
    raise TypeError(f"Unsupported hyperparameter search space for {name}: {space}")


class BERTSpecHyperparameterSampling(BaseModel):
    """Search space for BERT objective masking parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    masking_probability: OptunaFloat
    replacement_distribution: list[ReplacementDistribution]
    span_masking: list[ProbabilityDistribution]

    @model_validator(mode="after")
    def validate_concrete_candidates(self):
        masking_probabilities = _validation_space_values(
            "bert_spec.masking_probability",
            self.masking_probability,
        )
        if not self.replacement_distribution:
            raise ValueError("bert_spec.replacement_distribution cannot be empty")
        if not self.span_masking:
            raise ValueError("bert_spec.span_masking cannot be empty")

        replacement_distribution = self.replacement_distribution[0]
        span_masking = self.span_masking[0]
        for masking_probability in masking_probabilities:
            try:
                BERTSpecModel(
                    masking_probability=masking_probability,
                    replacement_distribution=replacement_distribution,
                    span_masking=span_masking,
                )
            except ValidationError as error:
                raise ValueError(
                    "bert_spec can sample an invalid configuration for "
                    f"masking_probability={masking_probability!r}:\n{error}"
                ) from error
        return self

    def validation_model(self) -> BERTSpecModel:
        """Build one already-validated representative BERT specification."""
        return BERTSpecModel(
            masking_probability=_validation_space_values(
                "bert_spec.masking_probability",
                self.masking_probability,
            )[0],
            replacement_distribution=self.replacement_distribution[0],
            span_masking=self.span_masking[0],
        )

    def grid_size(self) -> int:
        """Return the number of BERT-specific grid combinations."""
        return (
            grid_space_size("bert_spec.masking_probability", self.masking_probability)
            * len(self.replacement_distribution)
            * len(self.span_masking)
        )

    def sample_trial(self, trial: Any) -> BERTSpecModel:
        masking_probability = sample_param(
            trial, "bert_masking_probability", self.masking_probability
        )
        replacement_distribution_index = trial.suggest_categorical(
            "bert_replacement_distribution_index",
            list(range(len(self.replacement_distribution))),
        )
        span_masking_index = trial.suggest_categorical(
            "bert_span_masking_index", list(range(len(self.span_masking)))
        )

        replacement_distribution = self.replacement_distribution[
            replacement_distribution_index
        ].model_copy(deep=True)  # type: ignore
        span_masking = self.span_masking[span_masking_index].model_copy(deep=True)  # type: ignore

        logger.info(
            f"{masking_probability = } - {replacement_distribution = } - {span_masking = }"
        )

        return BERTSpecModel(
            masking_probability=masking_probability,
            replacement_distribution=replacement_distribution,
            span_masking=span_masking,
        )


@beartype
def load_hyperparameter_search_config(
    config_path: str, skip_metadata: bool
) -> "HyperparameterSearchConfig":
    """Load hyperparameter-search YAML plus optional metadata-derived fields."""
    config_values = load_composed_yaml_config(config_path)

    if "overrides" in config_values:
        if not config_values.get("base_config_path"):
            raise ValueError(
                f"Hyperparameter search config '{config_path}' uses the override "
                "format because it contains 'overrides', but 'base_config_path' "
                "is missing or empty."
            )
        # Import lazily: the partial compiler derives its registry from the
        # concrete legacy HPS models defined in this module.
        from sequifier.config.partial_hyperparameter_search_config import (
            compile_hyperparameter_search_override_config,
        )

        return compile_hyperparameter_search_override_config(
            config_path,
            config_values,
            skip_metadata,
        )

    return _load_legacy_hyperparameter_search_config(
        config_path,
        config_values,
        skip_metadata,
    )


def _load_legacy_hyperparameter_search_config(
    config_path: str,
    config_values: dict[str, Any],
    skip_metadata: bool,
) -> "HyperparameterSearchConfig":
    """Load the original self-contained hyperparameter-search format."""
    config_values = copy.deepcopy(config_values)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")
        if not isinstance(metadata_config_path, str) or not metadata_config_path:
            raise ValueError(
                f"Hyperparameter search config '{config_path}' must define a "
                "non-empty metadata_config_path when metadata loading is enabled."
            )

        with open(
            normalize_path(metadata_config_path, config_values["project_root"]), "r"
        ) as f:
            metadata_config = json.loads(f.read())

        config_values["special_token_ids"] = validate_special_token_ids(
            metadata_config.get(
                "special_token_ids",
                SPECIAL_TOKEN_IDS.ids_by_label,
            ),
            source=f"metadata config '{metadata_config_path}'",
        )

        config_values["column_data_types"] = config_values.get(
            "column_data_types", [metadata_config["column_data_types"]]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = [
                list(config_vals.keys())
                for config_vals in config_values["column_data_types"]
            ]

        config_values["categorical_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_data_types"].items()
                if "int" in type_.lower() and col in input_columns
            ]
            for input_columns in config_values["input_columns"]
        ]

        config_values["real_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_data_types"].items()
                if "float" in type_.lower() and col in input_columns
            ]
            for input_columns in config_values["input_columns"]
        ]

        config_values["n_classes"] = config_values.get(
            "n_classes", metadata_config["n_classes"]
        )

        storage_layout = stored_window_layout_from_metadata(metadata_config)
        if storage_layout.version != 2:
            raise ValueError(
                "Hyperparameter search requires metadata stored_window_layout_version=2, "
                f"got {storage_layout.version}."
            )

        config_values["storage_layout"] = storage_layout

        config_values["data_path"] = normalize_path(
            config_values.get("data_path", metadata_config["split_paths"][0]),
            config_values["project_root"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                metadata_config["split_paths"][
                    min(1, len(metadata_config["split_paths"]) - 1)
                ],
            ),
            config_values["project_root"],
        )

        config_values["id_maps"] = metadata_config["id_maps"]

    return try_catch_excess_keys(config_path, HyperparameterSearchConfig, config_values)


@dataclass(frozen=True)
class SampledTrainingConfig:
    """A concrete training spec plus its top-level objective and device."""

    training_spec: TrainingSpecModel
    training_objective: str
    device: str
    backbone_dropout: float
    ingestion_dropout: Optional[float] = None


class TrainingSpecHyperparameterSampling(BaseModel):
    """Training-spec search space with paired LR/scheduler candidates."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device: str
    epochs: list[int]
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
    save_latest_interval_minutes: Optional[float] = None
    save_interval_minutes: Optional[float] = None
    save_interval_val_loss: bool = True
    save_interval_batches: Optional[int] = None
    calculate_validation_loss_on_initialization: bool = False

    training_objective: list[str] = Field(default_factory=lambda: ["causal"])
    batch_size: OptunaInt
    learning_rate: list[float]  # Kept as list to preserve coupling with epochs
    bert_spec: Optional[BERTSpecHyperparameterSampling] = None
    next_occurrence_config: Optional[NextOccurrenceConfigModel] = None
    criterion: dict[str, str]
    class_weights: Optional[dict[str, list[float]]] = None
    accumulation_steps: OptionalOptunaInt
    gradient_clip: OptionalOptunaFloat = None
    dropout: OptunaFloat = [0.0]
    ingestion_dropout: Optional[OptunaFloat] = None

    loss_weights: Optional[dict[str, float]] = None
    optimizer: list[DotDict] = Field(
        default_factory=lambda: [DotDict({"name": "Adam"})]
    )
    scheduler_step_on: str = "epoch"
    scheduler: list[DotDict] = Field(
        default_factory=lambda: [
            DotDict({"name": "StepLR", "step_size": 1, "gamma": 0.99})
        ]
    )
    resume: Optional[ResumeConfig] = None
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: Union[int, float] = 16
    device_max_concat_length: int = 12
    world_size: int = 1
    num_workers: int = 0
    backend: str = "nccl"
    layer_type_dtypes: Optional[dict[str, str]] = None
    layer_autocast: Optional[bool] = False
    data_parallelism: Optional[str] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: str = "outer"
    float32_matmul_precision: str = "highest"

    def _build_training_spec(
        self,
        *,
        schedule_index: int,
        optimizer_index: int,
        training_objective: str,
        batch_size: int,
        dropout: float,
        ingestion_dropout: Optional[float],
        accumulation_steps: Optional[int],
        gradient_clip: Optional[float],
        bert_spec: Optional[BERTSpecModel] = None,
    ) -> SampledTrainingConfig:
        objective_class = get_objective_class(training_objective)
        next_occurrence_config = (
            self.next_occurrence_config
            if issubclass(objective_class, NextOccurrenceObjective)
            else None
        )
        training_spec = TrainingSpecModel(
            epochs=self.epochs[schedule_index],
            log_interval=self.log_interval,
            class_share_log_columns=self.class_share_log_columns,
            early_stopping_epochs=self.early_stopping_epochs,
            save_interval_epochs=self.save_interval_epochs,
            save_latest_interval_minutes=self.save_latest_interval_minutes,
            save_interval_minutes=self.save_interval_minutes,
            save_interval_batches=self.save_interval_batches,
            save_interval_val_loss=self.save_interval_val_loss,
            calculate_validation_loss_on_initialization=self.calculate_validation_loss_on_initialization,
            batch_size=batch_size,
            learning_rate=self.learning_rate[schedule_index],
            criterion=self.criterion,
            class_weights=self.class_weights,
            bert_spec=bert_spec,
            next_occurrence_config=next_occurrence_config,
            accumulation_steps=accumulation_steps,
            gradient_clip=gradient_clip,
            loss_weights=self.loss_weights,
            optimizer=self.optimizer[optimizer_index],
            scheduler=self.scheduler[schedule_index],
            resume=self.resume,
            enforce_determinism=True,
            scheduler_step_on=self.scheduler_step_on,
            distributed=self.distributed,
            load_full_data_to_ram=self.load_full_data_to_ram,
            max_ram_gb=self.max_ram_gb,
            device_max_concat_length=self.device_max_concat_length,
            world_size=self.world_size,
            num_workers=self.num_workers,
            backend=self.backend,
            layer_type_dtypes=self.layer_type_dtypes,
            layer_autocast=self.layer_autocast,
            data_parallelism=self.data_parallelism,
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            torch_compile=self.torch_compile,
            float32_matmul_precision=self.float32_matmul_precision,
        )
        return SampledTrainingConfig(
            training_spec=training_spec,
            training_objective=training_objective,
            device=self.device,
            backbone_dropout=dropout,
            ingestion_dropout=ingestion_dropout,
        )

    def validation_model(
        self,
        *,
        schedule_index: int = 0,
        optimizer_index: int = 0,
        training_objective: Optional[str] = None,
    ) -> SampledTrainingConfig:
        """Build a representative concrete training specification."""
        objective = training_objective or self.training_objective[0]
        objective_class = get_objective_class(objective)
        bert_spec = (
            self.bert_spec.validation_model()
            if (
                issubclass(objective_class, BERTObjective)
                and self.bert_spec is not None
            )
            else None
        )
        gradient_clip = self.gradient_clip
        if isinstance(gradient_clip, (list, FloatDistribution)):
            gradient_clip = _validation_space_values(
                "gradient_clip",
                gradient_clip,
            )[0]
        return self._build_training_spec(
            schedule_index=schedule_index,
            optimizer_index=optimizer_index,
            training_objective=objective,
            batch_size=_validation_space_values("batch_size", self.batch_size)[0],
            dropout=_validation_space_values("dropout", self.dropout)[0],
            ingestion_dropout=(
                _validation_space_values("ingestion_dropout", self.ingestion_dropout)[0]
                if self.ingestion_dropout is not None
                else None
            ),
            accumulation_steps=_validation_space_values(
                "accumulation_steps",
                self.accumulation_steps,
            )[0],
            gradient_clip=gradient_clip,
            bert_spec=bert_spec,
        )

    def grid_size(self) -> int:
        """Return the number of training-spec grid combinations."""
        gradient_clip_combinations = (
            grid_space_size("gradient_clip", self.gradient_clip)
            if isinstance(self.gradient_clip, (list, FloatDistribution))
            else 1
        )
        objective_combinations = sum(
            self.bert_spec.grid_size()
            if (
                issubclass(get_objective_class(objective_name), BERTObjective)
                and self.bert_spec is not None
            )
            else 1
            for objective_name in self.training_objective
        )
        ingestion_dropout_combinations = (
            grid_space_size("ingestion_dropout", self.ingestion_dropout)
            if self.ingestion_dropout is not None
            else 1
        )
        return (
            len(self.learning_rate)
            * len(self.optimizer)
            * objective_combinations
            * grid_space_size("batch_size", self.batch_size)
            * grid_space_size("dropout", self.dropout)
            * ingestion_dropout_combinations
            * grid_space_size("accumulation_steps", self.accumulation_steps)
            * gradient_clip_combinations
        )

    def __init__(self, **kwargs):
        """Normalize optimizer/scheduler dicts before Pydantic validation."""
        normalized_kwargs = dict(kwargs)
        if "optimizer" in normalized_kwargs:
            normalized_kwargs["optimizer"] = [
                DotDict(optimizer_config)
                for optimizer_config in normalized_kwargs["optimizer"]
            ]
        if "scheduler" in normalized_kwargs:
            normalized_kwargs["scheduler"] = [
                DotDict(scheduler_config)
                for scheduler_config in normalized_kwargs["scheduler"]
            ]
        super().__init__(**normalized_kwargs)

    @field_validator("training_objective", mode="before")
    @classmethod
    def normalize_training_objective(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("training_objective")
    @classmethod
    def validate_training_objective(cls, v):
        invalid = set(v).difference(ALLOWED_OBJECTIVE_NAMES)
        if invalid:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {invalid}"
            )
        return v

    @model_validator(mode="after")
    def validate_objective_specific_config(self):
        objective_classes = [
            get_objective_class(objective_name)
            for objective_name in self.training_objective
        ]
        has_bert_objective = any(
            issubclass(objective_class, BERTObjective)
            for objective_class in objective_classes
        )
        has_next_occurrence_objective = any(
            issubclass(objective_class, NextOccurrenceObjective)
            for objective_class in objective_classes
        )
        if has_bert_objective and self.bert_spec is None:
            raise ValueError(
                "If 'bert' is in training_objective, bert_spec must be configured."
            )
        if not has_bert_objective and self.bert_spec is not None:
            raise ValueError(
                "bert_spec should only be configured if 'bert' is in "
                "training_objective."
            )
        if has_next_occurrence_objective and self.next_occurrence_config is None:
            raise ValueError(
                "If 'next_occurrence' is in training_objective, "
                "next_occurrence_config must be configured."
            )
        if (
            not has_next_occurrence_objective
            and self.next_occurrence_config is not None
        ):
            raise ValueError(
                "next_occurrence_config should only be configured if "
                "'next_occurrence' is in training_objective."
            )
        return self

    @model_validator(mode="after")
    def validate_concrete_candidates(self):
        if not self.epochs:
            raise ValueError("epochs candidates cannot be empty")
        if not self.learning_rate:
            raise ValueError("learning_rate candidates cannot be empty")
        if not self.optimizer:
            raise ValueError("optimizer candidates cannot be empty")
        if not self.scheduler:
            raise ValueError("scheduler candidates cannot be empty")
        if not self.training_objective:
            raise ValueError("training_objective candidates cannot be empty")

        _validation_space_values("batch_size", self.batch_size)
        _validation_space_values("dropout", self.dropout)
        if self.ingestion_dropout is not None:
            _validation_space_values("ingestion_dropout", self.ingestion_dropout)
        _validation_space_values("accumulation_steps", self.accumulation_steps)
        if isinstance(self.gradient_clip, (list, FloatDistribution)):
            _validation_space_values("gradient_clip", self.gradient_clip)

        candidates: list[tuple[int, int, str]] = [
            (schedule_index, 0, self.training_objective[0])
            for schedule_index in range(len(self.epochs))
        ]
        candidates.extend(
            (0, optimizer_index, self.training_objective[0])
            for optimizer_index in range(len(self.optimizer))
        )
        candidates.extend((0, 0, objective) for objective in self.training_objective)
        for schedule_index, optimizer_index, objective in candidates:
            try:
                self.validation_model(
                    schedule_index=schedule_index,
                    optimizer_index=optimizer_index,
                    training_objective=objective,
                )
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    "training_hyperparameter_sampling can produce an invalid "
                    "TrainingSpecModel for "
                    f"schedule_index={schedule_index}, "
                    f"optimizer_index={optimizer_index}, "
                    f"training_objective={objective!r}:\n{error}"
                ) from error
        return self

    @field_validator("layer_type_dtypes")
    @classmethod
    def validate_layer_type_dtypes(cls, v):
        expected_keys = ["embedding", "linear", "conv", "norm", "decoder"]
        allowed_types = [
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "float8_e4m3fn",
            "float8_e5m2",
        ]
        bad_keys, bad_types = [], []
        if v:
            for k, vv in v.items():
                if k not in expected_keys:
                    bad_keys.append(k)
                if vv not in allowed_types:
                    bad_types.append(vv)

            if len(bad_keys) > 0:
                raise ValueError(
                    f"The following keys are invalid: {bad_keys}. Allowed keys are: {expected_keys}"
                )

            if len(bad_types) > 0:
                raise ValueError(
                    f"The following layer types are invalid: {bad_types}. Allowed types are: {allowed_types}"
                )

        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_model_spec(cls, v, info):
        if not (len(info.data.get("epochs")) == len(v)):
            raise ValueError(
                "learning_rate and epochs must have the same number of candidate values, that are paired"
            )

        return v

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler_config(cls, v, info_dict):
        learning_rate = info_dict.data.get("learning_rate")
        if learning_rate is not None and len(learning_rate) != len(v):
            raise ValueError(
                "learning_rate and scheduler must have the same number of "
                f"paired candidates, got {len(learning_rate)} and {len(v)}"
            )
        for i, scheduler_config in enumerate(v):
            if "total_steps" in scheduler_config:
                if info_dict.data.get("scheduler_step_on") == "epoch":
                    epochs = info_dict.data.get("epochs")[i]
                    if not scheduler_config["total_steps"] == epochs:
                        raise ValueError(
                            f"scheduler total steps: {scheduler_config['total_steps']} != {epochs}: total epochs"
                        )
                else:
                    logger.warning(
                        f"{scheduler_config['total_steps']} scheduler steps at {info_dict.data.get('epochs')[i]} epochs implies {scheduler_config['total_steps']/info_dict.data.get('epochs')[i]:.2f} batches. Does this seem correct?"
                    )
        return v

    def sample_trial(self, trial: Any) -> SampledTrainingConfig:
        """Sample training hyperparameters for one Optuna trial."""
        lr_sched_index = trial.suggest_categorical(
            "lr_sched_index", list(range(len(self.learning_rate)))
        )
        learning_rate = self.learning_rate[lr_sched_index]

        opt_index = trial.suggest_categorical(
            "optimizer_index", list(range(len(self.optimizer)))
        )
        optimizer = self.optimizer[opt_index]

        training_objective = trial.suggest_categorical(
            "training_objective", self.training_objective
        )
        objective_class = get_objective_class(training_objective)
        bert_spec = (
            self.bert_spec.sample_trial(trial)
            if (
                issubclass(objective_class, BERTObjective)
                and self.bert_spec is not None
            )
            else None
        )
        batch_size = sample_param(trial, "batch_size", self.batch_size)
        dropout = sample_param(trial, "dropout", self.dropout)
        ingestion_dropout = (
            sample_param(trial, "ingestion_dropout", self.ingestion_dropout)
            if self.ingestion_dropout is not None
            else None
        )
        accumulation_steps = sample_param(
            trial, "accumulation_steps", self.accumulation_steps
        )
        gradient_clip = (
            sample_param(trial, "gradient_clip", self.gradient_clip)
            if isinstance(self.gradient_clip, (list, FloatDistribution))
            else self.gradient_clip
        )

        logger.info(
            f"{training_objective = } - {learning_rate = } - {batch_size = } - "
            f"{dropout = } - {ingestion_dropout = } - {gradient_clip = } - "
            f"{optimizer = }"
        )

        return self._build_training_spec(
            schedule_index=lr_sched_index,
            optimizer_index=opt_index,
            training_objective=training_objective,
            batch_size=batch_size,
            dropout=dropout,
            ingestion_dropout=ingestion_dropout,
            accumulation_steps=accumulation_steps,
            gradient_clip=gradient_clip,
            bert_spec=bert_spec,
        )


class ModelSpecHyperparameterSampling(BaseModel):
    """Model-architecture search space with paired width choices."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dim_model: list[int]
    max_context_length: int = Field(2048, gt=0)
    backbone_id: str = "hyperparameter-search"
    ingestion_spec: Optional[Union[IngestionSpecConfig, list[IngestionSpecConfig]]] = (
        None
    )
    ingestion_merge: Optional[
        Union[IngestionMergeConfig, list[IngestionMergeConfig]]
    ] = None
    allow_shared_ingestion_columns: bool = False
    allow_unused_input_columns: bool = False
    auxiliary_input_columns: list[str] = Field(default_factory=list)
    initialization: ModelInitializationSamplingConfig = Field(
        default_factory=ModelInitializationSamplingConfig
    )
    ingestion_initialization: Optional[ModelInitializationSamplingConfig] = None
    backbone_initialization: Optional[ModelInitializationSamplingConfig] = None
    decoder_initialization: Optional[ModelInitializationSamplingConfig] = None
    n_head: list[int]

    dim_feedforward: OptunaInt
    num_layers: OptunaInt
    prediction_length: int
    decoding_support: Union[int, OptunaInt] = 1
    decoding_spec: Optional[Union[DecodingSpecConfig, list[DecodingSpecConfig]]] = None

    activation_fn: list[str]
    normalization: list[str]
    positional_encoding: list[str]
    positional_encoding_scope: list[str] = Field(
        default_factory=lambda: ["per_feature"]
    )
    attention_type: list[str]
    attention_output_projection: list[bool] = Field(default_factory=lambda: [True])

    norm_first: list[bool]
    shared_layer_groups: list[list[int]] = Field(default_factory=list)
    n_kv_heads: list[Optional[int]]
    rope_theta: OptunaFloat

    def _ingestion_spec_for_width(self, width_index: int):
        if isinstance(self.ingestion_spec, list):
            return self.ingestion_spec[width_index]
        return self.ingestion_spec

    def _ingestion_merge_for_width(self, width_index: int):
        if isinstance(self.ingestion_merge, list):
            return self.ingestion_merge[width_index]
        return self.ingestion_merge

    def _valid_kv_heads_for_width(self, width_index: int) -> list[Optional[int]]:
        n_head = self.n_head[width_index]
        valid_kv_heads = [
            kv
            for kv in self.n_kv_heads
            if kv is None or (n_head % kv == 0 and kv <= n_head)
        ]
        return valid_kv_heads or [None]

    def _decoding_spec_for_index(
        self,
        decoding_spec_index: int,
    ) -> Optional[DecodingSpecConfig]:
        if isinstance(self.decoding_spec, list):
            return self.decoding_spec[decoding_spec_index]
        return cast(Optional[DecodingSpecConfig], self.decoding_spec)

    def _validation_initializations(
        self,
    ) -> tuple[
        ModelInitializationConfig,
        ModelInitializationConfig,
        ModelInitializationConfig,
    ]:
        fallback = self.initialization.validation_config()
        return tuple(
            (
                sampling.validation_config()
                if sampling is not None
                else fallback.model_copy(deep=True)
            )
            for sampling in (
                self.ingestion_initialization,
                self.backbone_initialization,
                self.decoder_initialization,
            )
        )  # type: ignore[return-value]

    def _sample_initializations(
        self, trial: Any
    ) -> tuple[
        ModelInitializationConfig,
        ModelInitializationConfig,
        ModelInitializationConfig,
    ]:
        component_sampling = (
            ("ingestion", self.ingestion_initialization),
            ("backbone", self.backbone_initialization),
            ("decoder", self.decoder_initialization),
        )
        fallback = (
            self.initialization.sample_trial(trial)
            if any(sampling is None for _, sampling in component_sampling)
            else None
        )
        return tuple(
            (
                sampling.sample_trial(
                    trial, parameter_prefix=f"{component}_initialization"
                )
                if sampling is not None
                else cast(ModelInitializationConfig, fallback).model_copy(deep=True)
            )
            for component, sampling in component_sampling
        )  # type: ignore[return-value]

    def _build_model_spec(
        self,
        *,
        width_index: int,
        dim_feedforward: int,
        num_layers: int,
        decoding_support: int,
        decoding_spec: Optional[DecodingSpecConfig],
        activation_fn: str,
        normalization: str,
        positional_encoding: str,
        positional_encoding_scope: str,
        attention_type: str,
        attention_output_projection: bool,
        norm_first: bool,
        n_kv_heads: Optional[int],
        rope_theta: float,
        ingestion_initialization: ModelInitializationConfig,
        backbone_initialization: ModelInitializationConfig,
        decoder_initialization: ModelInitializationConfig,
    ) -> ModelSpecModel:
        dim_model = self.dim_model[width_index]
        transformer_input_width = dim_model - int(positional_encoding == "range_concat")
        ingestion_spec = self._ingestion_spec_for_width(width_index)
        ingestion_merge = self._ingestion_merge_for_width(width_index)
        if ingestion_spec is None:
            ingestion: dict[str, Any] = {
                "type": "direct_embed",
                "output_dim": transformer_input_width,
            }
        elif isinstance(ingestion_spec, dict):
            ingestion = {
                "type": "composite",
                "branches": {
                    name: branch.model_dump(mode="python")
                    for name, branch in ingestion_spec.items()
                },
                "merge": (
                    ingestion_merge.model_dump(mode="python")
                    if ingestion_merge is not None
                    else {"type": "concat"}
                ),
            }
        else:
            ingestion = ingestion_spec.model_dump(mode="python")
        ingestion.update(
            {
                "allow_shared_columns": self.allow_shared_ingestion_columns,
                "allow_unused_input_columns": self.allow_unused_input_columns,
                "auxiliary_input_columns": self.auxiliary_input_columns,
                "initialization": ingestion_initialization.model_dump(mode="python"),
            }
        )

        if decoding_spec is None:
            decoder: dict[str, Any] = {"type": "linear"}
        elif isinstance(decoding_spec, dict):
            decoder = {
                "type": "composite",
                "branches": {
                    name: branch.model_dump(mode="python")
                    for name, branch in decoding_spec.items()
                },
            }
        else:
            decoder = decoding_spec.model_dump(mode="python")
        decoder.update(
            {
                "prediction_length": self.prediction_length,
                "support": decoding_support,
                "initialization": decoder_initialization.model_dump(mode="python"),
            }
        )

        architecture_key = (
            f"d{dim_model}-l{num_layers}-h{self.n_head[width_index]}-"
            f"ff{dim_feedforward}-{attention_type}-{positional_encoding}"
        )
        model_spec = ModelSpecModel.model_validate(
            {
                "ingestion": ingestion,
                "backbone": {
                    "architecture": {
                        "dim_model": dim_model,
                        "max_context_length": self.max_context_length,
                        "num_layers": num_layers,
                        "attention": {
                            "type": attention_type,
                            "n_heads": self.n_head[width_index],
                            "n_kv_heads": n_kv_heads,
                            "output_projection": attention_output_projection,
                        },
                        "feed_forward": {
                            "dim": dim_feedforward,
                            "activation": activation_fn,
                        },
                        "normalization": {
                            "type": normalization,
                            "norm_first": norm_first,
                        },
                        "position_encoding": {
                            "type": positional_encoding,
                            "theta": rope_theta,
                        },
                        "positional_encoding_scope": positional_encoding_scope,
                        "dropout": 0.0,
                        "shared_layer_groups": self.shared_layer_groups,
                    },
                    "repository": {
                        "backbone_id": f"{self.backbone_id}-{architecture_key}",
                        "path": (
                            "checkpoints/backbones/hyperparameter-search/"
                            + architecture_key
                        ),
                        "load_policy": "if_exists",
                        "publish": False,
                        "conflict_policy": "compare_and_swap",
                    },
                    "initialization": backbone_initialization.model_dump(mode="python"),
                },
                "decoder": decoder,
            }
        )
        fingerprint = architecture_fingerprint(model_spec.backbone.architecture)
        repository_template = model_spec.backbone.repository
        if repository_template is None:
            raise RuntimeError(
                "Hyperparameter-search model construction did not create its "
                "backbone repository template."
            )
        repository = repository_template.model_copy(
            update={
                "path": (
                    "checkpoints/backbones/hyperparameter-search/" + fingerprint[:16]
                ),
                "backbone_id": f"{self.backbone_id}-{fingerprint[:16]}",
            }
        )
        return model_spec.model_copy(
            update={
                "backbone": model_spec.backbone.model_copy(
                    update={"repository": repository}
                )
            }
        )

    def validation_model(
        self,
        *,
        width_index: int = 0,
        dim_feedforward: Optional[int] = None,
        num_layers: Optional[int] = None,
        decoding_support: Optional[int] = None,
        decoding_spec_index: int = 0,
        activation_fn: Optional[str] = None,
        normalization: Optional[str] = None,
        positional_encoding: Optional[str] = None,
        positional_encoding_scope: Optional[str] = None,
        attention_type: Optional[str] = None,
        attention_output_projection: Optional[bool] = None,
        norm_first: Optional[bool] = None,
        n_kv_heads: object = _DEFAULT_KV_HEADS,
        rope_theta: Optional[float] = None,
    ) -> ModelSpecModel:
        """Build a representative concrete model specification."""
        selected_positional_encoding = (
            self.positional_encoding[0]
            if positional_encoding is None
            else positional_encoding
        )
        sampled_scope = (
            self.positional_encoding_scope[0]
            if positional_encoding_scope is None
            else positional_encoding_scope
        )
        selected_scope = (
            "global"
            if selected_positional_encoding in {"range", "range_concat", "sinusoidal"}
            else sampled_scope
        )
        if n_kv_heads is _DEFAULT_KV_HEADS:
            selected_kv_heads = self._valid_kv_heads_for_width(width_index)[0]
        elif n_kv_heads is None or isinstance(n_kv_heads, int):
            selected_kv_heads = n_kv_heads
        else:
            raise TypeError(
                f"n_kv_heads must be an integer or null, got {n_kv_heads!r}"
            )
        selected_decoding_support = decoding_support
        if selected_decoding_support is None:
            selected_decoding_support = (
                self.decoding_support
                if isinstance(self.decoding_support, int)
                else _validation_space_values(
                    "decoding_support",
                    self.decoding_support,
                )[0]
            )
        (
            ingestion_initialization,
            backbone_initialization,
            decoder_initialization,
        ) = self._validation_initializations()
        return self._build_model_spec(
            width_index=width_index,
            dim_feedforward=(
                _validation_space_values(
                    "dim_feedforward",
                    self.dim_feedforward,
                )[0]
                if dim_feedforward is None
                else dim_feedforward
            ),
            num_layers=(
                _validation_space_values("num_layers", self.num_layers)[0]
                if num_layers is None
                else num_layers
            ),
            decoding_support=selected_decoding_support,
            decoding_spec=self._decoding_spec_for_index(decoding_spec_index),
            activation_fn=activation_fn or self.activation_fn[0],
            normalization=normalization or self.normalization[0],
            positional_encoding=selected_positional_encoding,
            positional_encoding_scope=selected_scope,
            attention_type=attention_type or self.attention_type[0],
            attention_output_projection=(
                self.attention_output_projection[0]
                if attention_output_projection is None
                else attention_output_projection
            ),
            norm_first=self.norm_first[0] if norm_first is None else norm_first,
            n_kv_heads=selected_kv_heads,
            rope_theta=(
                _validation_space_values("rope_theta", self.rope_theta)[0]
                if rope_theta is None
                else rope_theta
            ),
            ingestion_initialization=ingestion_initialization,
            backbone_initialization=backbone_initialization,
            decoder_initialization=decoder_initialization,
        )

    def grid_size(self) -> int:
        """Return the number of model-spec grid combinations."""
        width_and_kv_head_combinations = sum(
            max(
                1,
                sum(
                    kv_head is None or (n_head % kv_head == 0 and kv_head <= n_head)
                    for kv_head in self.n_kv_heads
                ),
            )
            for n_head in self.n_head
        )
        decoding_support_combinations = (
            1
            if isinstance(self.decoding_support, int)
            else grid_space_size("decoding_support", self.decoding_support)
        )
        decoding_spec_combinations = (
            len(self.decoding_spec) if isinstance(self.decoding_spec, list) else 1
        )
        component_initialization_combinations = math.prod(
            sampling.grid_size()
            for sampling in (
                self.ingestion_initialization,
                self.backbone_initialization,
                self.decoder_initialization,
            )
            if sampling is not None
        )
        fallback_initialization_combinations = (
            self.initialization.grid_size()
            if any(
                sampling is None
                for sampling in (
                    self.ingestion_initialization,
                    self.backbone_initialization,
                    self.decoder_initialization,
                )
            )
            else 1
        )
        return (
            width_and_kv_head_combinations
            * decoding_support_combinations
            * decoding_spec_combinations
            * grid_space_size("dim_feedforward", self.dim_feedforward)
            * grid_space_size("num_layers", self.num_layers)
            * grid_space_size("rope_theta", self.rope_theta)
            * len(self.activation_fn)
            * len(self.normalization)
            * len(self.positional_encoding)
            * len(self.positional_encoding_scope)
            * len(self.attention_type)
            * len(self.attention_output_projection)
            * len(self.norm_first)
            * component_initialization_combinations
            * fallback_initialization_combinations
        )

    @field_validator("decoding_support")
    @classmethod
    def validate_decoding_support(cls, v):
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("decoding_support must be positive")
            return v
        if isinstance(v, list):
            if not v:
                raise ValueError("decoding_support candidates cannot be empty")
            invalid_values = [support for support in v if support <= 0]
            if invalid_values:
                raise ValueError(
                    "decoding_support candidates must be positive: " f"{invalid_values}"
                )
            return v
        if v.low <= 0:
            raise ValueError("decoding_support distribution must be positive")
        return v

    @field_validator("decoding_spec")
    @classmethod
    def validate_decoding_spec_candidates(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("decoding_spec candidates cannot be empty")
        return v

    @field_validator("attention_type")
    @classmethod
    def validate_attention_type_candidates(cls, v):
        invalid_attention_types = [
            attention_type
            for attention_type in v
            if attention_type not in ["mha", "mqa", "gqa"]
        ]
        if invalid_attention_types:
            raise ValueError(
                "Invalid attention_type candidates: " f"{invalid_attention_types}"
            )
        return v

    @field_validator("n_head")
    @classmethod
    def validate_model_spec(cls, v, info):
        dim_model = info.data.get("dim_model")
        if dim_model is None:
            return v
        invalid_dim_models = [value for value in dim_model or [] if value <= 0]
        if invalid_dim_models:
            raise ValueError(
                f"dim_model candidates must be positive: {invalid_dim_models}"
            )
        invalid_n_heads = [value for value in v if value <= 0]
        if invalid_n_heads:
            raise ValueError(f"n_head candidates must be positive: {invalid_n_heads}")

        ingestion_spec = info.data.get("ingestion_spec")
        if isinstance(ingestion_spec, list):
            if len(dim_model) != len(ingestion_spec):
                raise ValueError(
                    "dim_model and ingestion_spec must have the same number of candidate values, that are paired"
                )

        ingestion_merge = info.data.get("ingestion_merge")
        if isinstance(ingestion_merge, list):
            if len(dim_model) != len(ingestion_merge):
                raise ValueError(
                    "dim_model and ingestion_merge must have the same number of candidate values, that are paired"
                )

        if not (len(dim_model) == len(v)):
            raise ValueError(
                "dim_model and n_head must have the same number of candidate values, that are paired"
            )

        return v

    @field_validator("n_kv_heads")
    @classmethod
    def validate_n_kv_head_candidates(cls, v):
        invalid_values = [value for value in v if value is not None and value <= 0]
        if invalid_values:
            raise ValueError(
                f"n_kv_heads candidates must be positive or null: {invalid_values}"
            )
        return v

    @model_validator(mode="after")
    def validate_concrete_candidates(self):
        list_fields = {
            "dim_model": self.dim_model,
            "n_head": self.n_head,
            "activation_fn": self.activation_fn,
            "normalization": self.normalization,
            "positional_encoding": self.positional_encoding,
            "positional_encoding_scope": self.positional_encoding_scope,
            "attention_type": self.attention_type,
            "attention_output_projection": self.attention_output_projection,
            "norm_first": self.norm_first,
            "n_kv_heads": self.n_kv_heads,
        }
        empty_fields = [name for name, values in list_fields.items() if not values]
        if empty_fields:
            raise ValueError(
                "model hyperparameter candidate lists cannot be empty: "
                f"{empty_fields}"
            )

        dim_feedforward_values = _validation_space_values(
            "dim_feedforward",
            self.dim_feedforward,
        )
        num_layer_values = _validation_space_values("num_layers", self.num_layers)
        rope_theta_values = _validation_space_values("rope_theta", self.rope_theta)
        decoding_support_values = (
            [self.decoding_support]
            if isinstance(self.decoding_support, int)
            else _validation_space_values(
                "decoding_support",
                self.decoding_support,
            )
        )
        decoding_specs = (
            self.decoding_spec
            if isinstance(self.decoding_spec, list)
            else [self.decoding_spec]
        )

        candidates: list[tuple[str, dict[str, Any]]] = []
        candidates.extend(
            (f"width_index={width_index}", {"width_index": width_index})
            for width_index in range(len(self.dim_model))
        )
        candidates.extend(
            (f"dim_feedforward={value!r}", {"dim_feedforward": value})
            for value in dim_feedforward_values
        )
        candidates.extend(
            (f"num_layers={value!r}", {"num_layers": value})
            for value in num_layer_values
        )
        candidates.extend(
            (f"decoding_support={value!r}", {"decoding_support": value})
            for value in decoding_support_values
        )
        candidates.extend(
            (
                f"decoding_spec_index={index}",
                {"decoding_spec_index": index},
            )
            for index in range(len(decoding_specs))
        )
        for field_name, values in (
            ("activation_fn", self.activation_fn),
            ("normalization", self.normalization),
            ("attention_output_projection", self.attention_output_projection),
            ("norm_first", self.norm_first),
        ):
            candidates.extend(
                (f"{field_name}={value!r}", {field_name: value}) for value in values
            )
        candidates.extend(
            (f"rope_theta={value!r}", {"rope_theta": value})
            for value in rope_theta_values
        )

        for width_index in range(len(self.dim_model)):
            for positional_encoding in self.positional_encoding:
                for positional_encoding_scope in self.positional_encoding_scope:
                    candidates.append(
                        (
                            "width/positional encoding combination "
                            f"({width_index}, {positional_encoding!r}, "
                            f"{positional_encoding_scope!r})",
                            {
                                "width_index": width_index,
                                "positional_encoding": positional_encoding,
                                "positional_encoding_scope": (
                                    positional_encoding_scope
                                ),
                            },
                        )
                    )
            for attention_type in self.attention_type:
                for n_kv_heads in self._valid_kv_heads_for_width(width_index):
                    candidates.append(
                        (
                            "width/attention/KV-head combination "
                            f"({width_index}, {attention_type!r}, {n_kv_heads!r})",
                            {
                                "width_index": width_index,
                                "attention_type": attention_type,
                                "n_kv_heads": n_kv_heads,
                            },
                        )
                    )

        for description, candidate_kwargs in candidates:
            try:
                self.validation_model(**candidate_kwargs)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    "model_hyperparameter_sampling can produce an invalid "
                    f"ModelSpecModel for {description}:\n{error}"
                ) from error
        return self

    def sample_trial(self, trial: Any) -> ModelSpecModel:
        """Sample architecture hyperparameters for one Optuna trial."""
        dim_model_idx = trial.suggest_categorical(
            "dim_model_idx", list(range(len(self.dim_model)))
        )

        dim_model = self.dim_model[dim_model_idx]
        n_head = self.n_head[dim_model_idx]
        decoding_support = (
            self.decoding_support
            if isinstance(self.decoding_support, int)
            else sample_param(trial, "decoding_support", self.decoding_support)
        )
        if isinstance(self.decoding_spec, list):
            decoding_spec_idx = trial.suggest_categorical(
                "decoding_spec_idx", list(range(len(self.decoding_spec)))
            )
            decoding_spec = self._decoding_spec_for_index(decoding_spec_idx)
        else:
            decoding_spec = self._decoding_spec_for_index(0)

        dim_feedforward = sample_param(trial, "dim_feedforward", self.dim_feedforward)
        num_layers = sample_param(trial, "num_layers", self.num_layers)
        rope_theta = sample_param(trial, "rope_theta", self.rope_theta)
        (
            ingestion_initialization,
            backbone_initialization,
            decoder_initialization,
        ) = self._sample_initializations(trial)

        activation_fn = trial.suggest_categorical("activation_fn", self.activation_fn)
        normalization = trial.suggest_categorical("normalization", self.normalization)
        positional_encoding = trial.suggest_categorical(
            "positional_encoding", self.positional_encoding
        )
        sampled_positional_encoding_scope = trial.suggest_categorical(
            "positional_encoding_scope", self.positional_encoding_scope
        )
        positional_encoding_scope = (
            "global"
            if positional_encoding in {"range", "range_concat", "sinusoidal"}
            else sampled_positional_encoding_scope
        )
        attention_type = trial.suggest_categorical(
            "attention_type", self.attention_type
        )
        attention_output_projection = trial.suggest_categorical(
            "attention_output_projection", self.attention_output_projection
        )
        norm_first = trial.suggest_categorical("norm_first", self.norm_first)

        valid_kv_heads = [
            kv
            for kv in self.n_kv_heads
            if kv is None or (n_head % kv == 0 and kv <= n_head)
        ]

        if not valid_kv_heads:
            logger.warning(
                f"No valid n_kv_heads found in config for n_head={n_head}. Defaulting to None (MHA)."
            )
            n_kv_heads = None
        else:
            n_kv_heads = trial.suggest_categorical("n_kv_heads", valid_kv_heads)

        logger.info(
            f"{dim_model = } - {dim_feedforward = } - {num_layers = } - "
            f"{activation_fn = } - {normalization = } - "
            f"{positional_encoding = } - {positional_encoding_scope = } - "
            f"{attention_type = } - {attention_output_projection = } - "
            f"{norm_first = } - {n_kv_heads = } - {rope_theta = } "
        )

        return self._build_model_spec(
            width_index=dim_model_idx,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            decoding_support=decoding_support,
            decoding_spec=decoding_spec,
            activation_fn=activation_fn,
            normalization=normalization,
            positional_encoding=positional_encoding,
            positional_encoding_scope=positional_encoding_scope,
            attention_type=attention_type,
            attention_output_projection=attention_output_projection,
            norm_first=norm_first,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
            ingestion_initialization=ingestion_initialization,
            backbone_initialization=backbone_initialization,
            decoder_initialization=decoder_initialization,
        )


class HyperparameterSearchConfig(BaseModel):
    """Top-level Optuna search config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    metadata_config_path: str
    hp_search_name: str
    search_strategy: str = "bayesian"
    global_seed: Optional[int] = None
    seed: Optional[list[int]] = None
    n_trials: Optional[int] = Field(None, alias="n_samples")
    prune_trials: Optional[bool] = True
    pruning_warmup_epochs: Optional[int] = Field(default=None, ge=0)
    pruning_warmup_batches: Optional[int] = Field(default=None, ge=0)
    model_config_write_path: str
    data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    input_columns: list[list[str]]
    column_data_types: list[dict[str, str]]
    categorical_columns: list[list[str]]
    real_columns: list[list[str]]
    target_columns: list[str]
    target_column_types: dict[str, str]
    id_maps: dict[str, dict[str | int, int]]
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: SPECIAL_TOKEN_IDS.ids_by_label
    )
    categorical_decoder_special_tokens: dict[str, list[str]] = Field(
        default_factory=dict
    )

    context_length: list[int]
    target_offset: int = Field(default=1, ge=0)
    storage_layout: StoredWindowLayout
    model_window_stride: Optional[int] = Field(default=None, gt=0)
    n_classes: dict[str, int]
    inference_batch_size: int

    export_generative_model: bool
    export_embedding_model: bool
    embedding_layer_names: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EMBEDDING_LAYER_NAMES),
        min_length=1,
    )
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    feature_layout: Optional[FeatureLayoutRegistryModel] = None

    evaluation_inference_config: Optional[str] = None
    evaluation_script: Optional[str] = None
    evaluation_metric_directions: Optional[list[str]] = None
    evaluation_metrics: Optional[list[str]] = None

    model_hyperparameter_sampling: ModelSpecHyperparameterSampling
    training_hyperparameter_sampling: TrainingSpecHyperparameterSampling

    override_input: bool = False

    def _build_train_model(
        self,
        *,
        model_spec: ModelSpecModel,
        training_config: SampledTrainingConfig,
        input_columns_index: int,
        context_length: int,
        seed: int,
        run_index: int,
    ) -> ResolvedSequifierConfig:
        authored = self._build_authored_config(
            model_spec=model_spec,
            training_config=training_config,
            input_columns_index=input_columns_index,
            context_length=context_length,
            seed=seed,
            run_index=run_index,
        )
        metadata = DatasetMetadata(
            split_paths=[self.data_path, self.validation_data_path],
            column_data_types=self.column_data_types[input_columns_index],
            n_classes=self.n_classes,
            id_maps=self.id_maps,
            special_token_ids=self.special_token_ids,
            stored_context_width=self.storage_layout.stored_context_width,
            max_target_offset=self.storage_layout.max_target_offset,
            stored_window_layout_version=self.storage_layout.version,
        )
        return resolve_sequifier_config(authored, metadata)

    def _build_authored_config(
        self,
        *,
        model_spec: ModelSpecModel,
        training_config: SampledTrainingConfig,
        input_columns_index: int,
        context_length: int,
        seed: int,
        run_index: int,
    ) -> SequifierConfig:
        training_spec = training_config.training_spec
        training_objective = training_config.training_objective
        objective_class = get_objective_class(training_objective)
        architecture = model_spec.backbone.architecture.model_copy(
            update={"dropout": training_config.backbone_dropout}
        )
        ingestion = model_spec.ingestion
        if training_config.ingestion_dropout is not None:
            ingestion_updates: dict[str, Any] = {
                "dropout": training_config.ingestion_dropout
            }
            if ingestion.type == "composite":
                ingestion_updates["branches"] = {
                    name: branch.model_copy(
                        update={"dropout": training_config.ingestion_dropout}
                    )
                    for name, branch in ingestion.branches.items()
                }
            ingestion = ingestion.model_copy(update=ingestion_updates)
        repository_template = model_spec.backbone.repository
        if repository_template is None:
            repository = None
        else:
            fingerprint = architecture_fingerprint(architecture)
            repository = repository_template.model_copy(
                update={
                    "backbone_id": (
                        repository_template.backbone_id.rsplit("-", 1)[0]
                        + f"-{fingerprint[:16]}"
                    ),
                    "path": os.path.join(
                        os.path.dirname(repository_template.path),
                        fingerprint[:16],
                    ),
                }
            )
        model_spec = model_spec.model_copy(
            update={
                "ingestion": ingestion,
                "backbone": model_spec.backbone.model_copy(
                    update={
                        "architecture": architecture,
                        "repository": repository,
                    }
                ),
            }
        )
        if not objective_class.forward_looking:
            model_spec = model_spec.model_copy(
                update={
                    "decoder": model_spec.decoder.model_copy(
                        update={"prediction_length": context_length}
                    )
                }
            )

        return SequifierConfig(
            project_root=self.project_root,
            metadata_config_path=self.metadata_config_path,
            model_name=f"{self.hp_search_name}-run-{run_index}",
            training_objective=training_objective,
            device=training_config.device,
            data_path=self.data_path,
            validation_data_path=self.validation_data_path,
            read_format=self.read_format,
            input_columns=self.input_columns[input_columns_index],
            column_data_types=self.column_data_types[input_columns_index],
            target_columns=self.target_columns,
            target_column_types=self.target_column_types,
            categorical_decoder_special_tokens=self.categorical_decoder_special_tokens,
            context_length=context_length,
            target_offset=target_offset_for_objective(
                training_objective,
                self.target_offset,
            ),
            model_window_stride=self.model_window_stride,
            inference_batch_size=self.inference_batch_size,
            seed=seed,
            export_embedding_model=self.export_embedding_model,
            export_generative_model=self.export_generative_model,
            embedding_layer_names=self.embedding_layer_names,
            export_onnx=self.export_onnx,
            export_pt=self.export_pt,
            export_with_dropout=self.export_with_dropout,
            feature_layout=self.feature_layout,
            model_spec=model_spec,
            training_spec=training_spec,
        )

    @field_validator("special_token_ids")
    @classmethod
    def validate_special_token_ids_match_runtime(cls, v):
        return validate_special_token_ids(v, source="HyperparameterSearchConfig")

    @model_validator(mode="after")
    def validate_sequence_layout(self):
        for cl in self.context_length:
            if (
                cl + self.storage_layout.max_target_offset
                > self.storage_layout.stored_context_width
            ):
                raise ValueError(
                    f"Window capacity mismatch: context_length ({cl}) + max_target_offset "
                    f"({self.storage_layout.max_target_offset}) > stored_context_width ({self.storage_layout.stored_context_width}). "
                    "Model inputs cannot exceed the preprocessed sequence length."
                )
        forward_objectives = forward_objective_names()
        if (
            set(self.training_hyperparameter_sampling.training_objective)
            & forward_objectives
        ):
            if self.storage_layout.max_target_offset < 1:
                raise ValueError(
                    "The hyperparameter search space includes a forward-looking "
                    "objective ('causal', 'final_value', or 'next_occurrence'), "
                    "but the preprocessed dataset has max_target_offset=0. "
                    "Causal, final_value, and next_occurrence modeling require "
                    "max_target_offset >= 1."
                )
        for objective_name in self.training_hyperparameter_sampling.training_objective:
            target_offset = target_offset_for_objective(
                objective_name,
                self.target_offset,
            )
            if target_offset > self.storage_layout.max_target_offset:
                raise ValueError(
                    f"Hyperparameter search target_offset={target_offset} for "
                    f"training objective {objective_name!r} exceeds the stored "
                    "dataset's max_target_offset="
                    f"{self.storage_layout.max_target_offset}."
                )
        return self

    @model_validator(mode="after")
    def validate_prune_trials(self):
        if self.prune_trials and self.training_hyperparameter_sampling.distributed:
            warnings.warn(
                "Trial pruning in distributed training settings is in beta mode."
            )
        return self

    @model_validator(mode="after")
    def validate_pruning_warmup(self):
        if (
            self.pruning_warmup_epochs is not None
            and self.pruning_warmup_batches is not None
        ):
            raise ValueError(
                "Only one of pruning_warmup_epochs and pruning_warmup_batches "
                "can be set."
            )
        return self

    @model_validator(mode="after")
    def validate_grid_n_trials_matches_combinations(self):
        """Require an explicit grid trial limit to cover the complete grid."""
        if self.search_strategy != "grid" or self.n_trials is None:
            return self

        seed_combinations = 1 if self.seed is None else len(self.seed)
        configured_combinations = (
            self.model_hyperparameter_sampling.grid_size()
            * self.training_hyperparameter_sampling.grid_size()
            * seed_combinations
            * len(self.input_columns)
            * len(self.context_length)
        )
        if self.n_trials != configured_combinations:
            raise ValueError(
                "For search_strategy='grid', n_samples must equal the number of "
                f"configured combinations ({configured_combinations}), got "
                f"{self.n_trials}. Remove n_samples to let the grid run until "
                "exhaustion, or set it to the configured combination count."
            )
        return self

    @model_validator(mode="after")
    def validate_concrete_train_models(self):
        candidate_lengths = {
            "input_columns": len(self.input_columns),
            "column_data_types": len(self.column_data_types),
            "categorical_columns": len(self.categorical_columns),
            "real_columns": len(self.real_columns),
        }
        if not self.input_columns:
            raise ValueError("input_columns candidates cannot be empty")
        if len(set(candidate_lengths.values())) != 1:
            raise ValueError(
                "input_columns, column_data_types, categorical_columns, and "
                "real_columns must have the same number of paired candidates; "
                + ", ".join(
                    f"{name}={length}" for name, length in candidate_lengths.items()
                )
            )
        if not self.context_length:
            raise ValueError("context_length candidates cannot be empty")
        if self.seed is not None and not self.seed:
            raise ValueError("seed candidates cannot be empty")

        model_sampling = self.model_hyperparameter_sampling
        training_sampling = self.training_hyperparameter_sampling
        baseline_seed = 101 if self.seed is None else self.seed[0]
        baseline_context = self.context_length[0]
        baseline_model = model_sampling.validation_model()
        baseline_training = training_sampling.validation_model()

        candidates: list[
            tuple[str, ModelSpecModel, SampledTrainingConfig, int, int]
        ] = [
            (
                "the baseline candidate",
                baseline_model,
                baseline_training,
                0,
                baseline_context,
            )
        ]

        for input_columns_index in range(len(self.input_columns)):
            for width_index in range(len(model_sampling.dim_model)):
                candidates.append(
                    (
                        "input/model-width combination "
                        f"({input_columns_index}, {width_index})",
                        model_sampling.validation_model(width_index=width_index),
                        baseline_training,
                        input_columns_index,
                        baseline_context,
                    )
                )

        decoding_specs = (
            model_sampling.decoding_spec
            if isinstance(model_sampling.decoding_spec, list)
            else [model_sampling.decoding_spec]
        )
        for decoding_spec_index in range(len(decoding_specs)):
            candidates.append(
                (
                    f"decoding_spec_index={decoding_spec_index}",
                    model_sampling.validation_model(
                        decoding_spec_index=decoding_spec_index
                    ),
                    baseline_training,
                    0,
                    baseline_context,
                )
            )

        decoding_support_values = (
            [model_sampling.decoding_support]
            if isinstance(model_sampling.decoding_support, int)
            else _validation_space_values(
                "decoding_support",
                model_sampling.decoding_support,
            )
        )
        for context_length in self.context_length:
            for objective in training_sampling.training_objective:
                training_config = training_sampling.validation_model(
                    training_objective=objective
                )
                for decoding_support in decoding_support_values:
                    candidates.append(
                        (
                            "context/objective/decoding-support combination "
                            f"({context_length}, {objective!r}, "
                            f"{decoding_support})",
                            model_sampling.validation_model(
                                decoding_support=decoding_support
                            ),
                            training_config,
                            0,
                            context_length,
                        )
                    )

        for (
            description,
            model_spec,
            training_config,
            input_index,
            context,
        ) in candidates:
            try:
                self._build_train_model(
                    model_spec=model_spec,
                    training_config=training_config,
                    input_columns_index=input_index,
                    context_length=context,
                    seed=baseline_seed,
                    run_index=0,
                )
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    "Hyperparameter search can produce an invalid SequifierConfig for "
                    f"{description}:\n{error}"
                ) from error
        return self

    @field_validator("evaluation_metrics")
    @classmethod
    def validate_evaluation_metrics(cls, v, info):
        if v is not None and info.data.get("evaluation_script") is None:
            raise ValueError(
                "evaluation_script must be provided if evaluation_metrics is defined."
            )
        if v is not None:
            if info.data.get("evaluation_metric_directions") is None:
                raise ValueError(
                    "evaluation_metric_directions must be provided if evaluation_metrics is defined."
                )
            else:
                evaluation_metric_directions = info.data.get(
                    "evaluation_metric_directions"
                )
                if len(v) != len(evaluation_metric_directions):
                    raise ValueError(
                        f"evaluation_metrics and evaluation_metric_directions must have the same number of values, len(evaluation_metrics) = {len(v)}, {len(evaluation_metric_directions) = }"
                    )
        if v is not None and info.data.get("evaluation_inference_config") is None:
            warnings.warn(
                "Please provide evaluation_inference_config if your evaluation_script requires inference outputs"
            )
        return v

    @field_validator("evaluation_metric_directions")
    @classmethod
    def validate_evaluation_metric_directions(cls, v):
        if v is not None:
            allowed_vals = {"minimize", "maximize"}
            diff = set(v).difference(allowed_vals)
            if len(diff):
                raise ValueError(
                    f"In evaluation_metric_directions, only 'minimize' and 'maximize' are allowed, found: {diff}"
                )
        return v

    @field_validator("evaluation_script")
    @classmethod
    def validate_evaluation_script(cls, v, info):
        if v is not None:
            project_root = info.data.get("project_root")
            if not os.path.exists(os.path.join(project_root, v)):
                raise ValueError(
                    f"evaluation_script '{v}' does not exist at '{project_root}'"
                )
        return v

    @field_validator("evaluation_inference_config")
    @classmethod
    def validate_evaluation_inference_config(cls, v, info):
        if v is not None:
            if not os.path.exists(v):
                raise ValueError(f"evaluation_inference_config '{v}' does not exist")
        return v

    @field_validator("column_data_types")
    @classmethod
    def validate_model_spec(cls, v, info):
        input_columns = info.data.get("input_columns")
        if input_columns is not None and len(input_columns) != len(v):
            raise ValueError(
                "input_columns and column_data_types must have the same number of candidate values, that are paired"
            )
        return v

    @field_validator("search_strategy")
    @classmethod
    def validate_search_strategy(cls, v: str) -> str:
        allowed = ["sample", "grid", "bayesian"]
        if v not in allowed:
            raise ValueError(f"search_strategy must be one of {allowed}, got '{v}'")
        return v

    def sample_trial(self, trial: Any, run_index: int) -> SequifierConfig:
        """Sample and validate one concrete authored training config."""
        model_spec = self.model_hyperparameter_sampling.sample_trial(trial)

        seed = (
            101 if self.seed is None else trial.suggest_categorical("seed", self.seed)
        )
        input_columns_index = trial.suggest_categorical(
            "input_columns_index", list(range(len(self.input_columns)))
        )
        context_length = trial.suggest_categorical(
            "context_length", self.context_length
        )
        training_config = self.training_hyperparameter_sampling.sample_trial(trial)
        logger.info(f"{seed = } - {input_columns_index = } - {context_length = }")

        return self._build_authored_config(
            model_spec=model_spec,
            training_config=training_config,
            input_columns_index=input_columns_index,
            context_length=context_length,
            seed=seed,
            run_index=run_index,
        )


_PARTIAL_CONFIG_EXPORTS = {
    "BERTSpecHyperparameterSamplingOverride",
    "HyperparameterSearchOverrides",
    "ModelSpecHyperparameterSamplingOverride",
    "PartialHyperparameterSearchConfig",
    "TrainingSpecHyperparameterSamplingOverride",
    "compile_hyperparameter_search_override_config",
}


def __getattr__(name: str):
    """Lazily preserve imports moved to the partial-config compiler module."""
    if name not in _PARTIAL_CONFIG_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from sequifier.config import partial_hyperparameter_search_config

    return getattr(partial_hyperparameter_search_config, name)
