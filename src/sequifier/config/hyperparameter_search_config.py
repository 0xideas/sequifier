import json
import os
import warnings
from typing import Any, Optional, Union

import yaml
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.probabilities import ProbabilityDistribution
from sequifier.config.train_config import (
    BERTSpecModel,
    DotDict,
    FeatureLayoutRegistryModel,
    IngestionMergeConfig,
    IngestionSpecConfig,
    ModelSpecModel,
    NextOccurrenceConfigModel,
    ReplacementDistribution,
    TrainingSpecModel,
    TrainModel,
)
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    normalize_path,
    resolve_window_view,
    stored_window_layout_from_metadata,
    try_catch_excess_keys,
)
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    BERTObjective,
    NextOccurrenceObjective,
    forward_objective_names,
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.special_tokens import validate_special_token_ids


class FloatDistribution(BaseModel):
    """Optuna float range with optional step/log sampling."""

    low: float
    high: float
    step: Optional[float] = None
    log: bool = False

    @model_validator(mode="after")
    def validate_step_and_log(self):
        if self.log and self.step is not None:
            raise ValueError(
                f"Optuna does not support setting step when log=True. "
                f"Got step={self.step} and log={self.log}."
            )
        return self


class IntDistribution(BaseModel):
    """Optuna integer range with step/log sampling."""

    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def validate_step_and_log(self):
        if self.log and self.step != 1:
            raise ValueError(
                f"Optuna does not support setting step != 1 when log=True. "
                f"Got step={self.step} and log={self.log}."
            )
        return self


OptunaFloat = Union[list[float], FloatDistribution]
OptunaInt = Union[list[int], IntDistribution]


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


class BERTSpecHyperparameterSampling(BaseModel):
    """Search space for BERT objective masking parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    masking_probability: OptunaFloat
    replacement_distribution: list[ReplacementDistribution]
    span_masking: list[ProbabilityDistribution]

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
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_root"]), "r"
        ) as f:
            metadata_config = json.loads(f.read())

        validate_special_token_ids(
            metadata_config["special_token_ids"],
            source=f"metadata config '{metadata_config_path}'",
        )

        config_values["column_types"] = config_values.get(
            "column_types", [metadata_config["column_types"]]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = [
                list(config_vals.keys())
                for config_vals in config_values["column_types"]
            ]

        config_values["categorical_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_types"].items()
                if "int" in type_.lower() and col in input_columns
            ]
            for input_columns in config_values["input_columns"]
        ]

        config_values["real_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_types"].items()
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

        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", metadata_config["split_paths"][0]),
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
    accumulation_steps: OptunaInt
    dropout: OptunaFloat = [0.0]

    loss_weights: Optional[dict[str, float]] = None
    optimizer: list[DotDict] = Field(
        default_factory=lambda: [DotDict({"name": "Adam"})]
    )
    scheduler: list[DotDict] = Field(
        default_factory=lambda: [
            DotDict({"name": "StepLR", "step_size": 1, "gamma": 0.99})
        ]
    )
    continue_training: bool
    scheduler_step_on: str = "epoch"
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: Union[int, float] = 16
    device_max_concat_length: int = 12
    world_size: int = 1
    num_workers: int = 0
    backend: str = "nccl"
    layer_type_dtypes: Optional[dict[str, str]] = None
    layer_autocast: Optional[bool] = True
    data_parallelism: Optional[str] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: str = "outer"
    float32_matmul_precision: str = "highest"

    def __init__(self, **kwargs):
        """Normalize optimizer/scheduler dicts after Pydantic validation."""
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        self.optimizer = [
            DotDict(optimizer_config) for optimizer_config in kwargs["optimizer"]
        ]
        if not len(self.learning_rate) == len(kwargs["scheduler"]):
            raise ValueError(
                f"{len(self.learning_rate) = } != {len(kwargs['scheduler']) = }"
            )

        self.scheduler = [
            DotDict(scheduler_config) for scheduler_config in kwargs["scheduler"]
        ]

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

    def sample_trial(self, trial: Any) -> TrainingSpecModel:
        """Sample training hyperparameters for one Optuna trial."""
        lr_sched_index = trial.suggest_categorical(
            "lr_sched_index", list(range(len(self.learning_rate)))
        )
        epochs = self.epochs[lr_sched_index]
        learning_rate = self.learning_rate[lr_sched_index]
        scheduler = self.scheduler[lr_sched_index]

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
        next_occurrence_config = (
            self.next_occurrence_config
            if issubclass(objective_class, NextOccurrenceObjective)
            else None
        )

        batch_size = sample_param(trial, "batch_size", self.batch_size)
        dropout = sample_param(trial, "dropout", self.dropout)
        accumulation_steps = sample_param(
            trial, "accumulation_steps", self.accumulation_steps
        )

        logger.info(
            f"{training_objective = } - {learning_rate = } - {batch_size = } - {dropout = } - {optimizer = }"
        )

        return TrainingSpecModel(
            training_objective=training_objective,
            device=self.device,
            epochs=epochs,
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
            learning_rate=learning_rate,
            criterion=self.criterion,
            class_weights=self.class_weights,
            bert_spec=bert_spec,
            next_occurrence_config=next_occurrence_config,
            accumulation_steps=accumulation_steps,
            dropout=dropout,
            loss_weights=self.loss_weights,
            optimizer=optimizer,
            scheduler=scheduler,
            continue_training=self.continue_training,
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


class ModelSpecHyperparameterSampling(BaseModel):
    """Model-architecture search space with paired width choices."""

    dim_model: list[int]
    ingestion_spec: Optional[Union[IngestionSpecConfig, list[IngestionSpecConfig]]] = (
        None
    )
    ingestion_merge: Optional[
        Union[IngestionMergeConfig, list[IngestionMergeConfig]]
    ] = None
    allow_shared_ingestion_columns: bool = False
    allow_unused_input_columns: bool = False
    auxiliary_input_columns: list[str] = Field(default_factory=list)
    n_head: list[int]

    dim_feedforward: OptunaInt
    num_layers: OptunaInt
    prediction_length: int

    activation_fn: list[str]
    normalization: list[str]
    positional_encoding: list[str]
    attention_type: list[str]

    norm_first: list[bool]
    n_kv_heads: list[Optional[int]]
    rope_theta: OptunaFloat

    @field_validator("n_head")
    @classmethod
    def validate_model_spec(cls, v, info):
        ingestion_spec = info.data.get("ingestion_spec")
        if isinstance(ingestion_spec, list):
            if len(info.data.get("dim_model")) != len(ingestion_spec):
                raise ValueError(
                    "dim_model and ingestion_spec must have the same number of candidate values, that are paired"
                )

        ingestion_merge = info.data.get("ingestion_merge")
        if isinstance(ingestion_merge, list):
            if len(info.data.get("dim_model")) != len(ingestion_merge):
                raise ValueError(
                    "dim_model and ingestion_merge must have the same number of candidate values, that are paired"
                )

        if not (len(info.data.get("dim_model")) == len(v)):
            raise ValueError(
                "dim_model and n_head must have the same number of candidate values, that are paired"
            )

        return v

    @model_validator(mode="after")
    def validate_fixed_single_ingestion_matches_dim_model(self):
        if self.ingestion_spec is None or isinstance(self.ingestion_spec, dict):
            return self

        if isinstance(self.ingestion_spec, list):
            mismatched_candidates = [
                (index, dim_model, ingestion_spec.output_dim)
                for index, (dim_model, ingestion_spec) in enumerate(
                    zip(self.dim_model, self.ingestion_spec)
                )
                if not isinstance(ingestion_spec, dict)
                and dim_model != ingestion_spec.output_dim
            ]
            if mismatched_candidates:
                raise ValueError(
                    "model_hyperparameter_sampling.ingestion_spec list candidates "
                    "must match their paired dim_model values for single-branch "
                    f"ingestions. Mismatches: {mismatched_candidates}"
                )
            return self

        mismatched_dim_models = [
            dim_model
            for dim_model in self.dim_model
            if dim_model != self.ingestion_spec.output_dim
        ]
        if mismatched_dim_models:
            raise ValueError(
                "model_hyperparameter_sampling.ingestion_spec.output_dim must "
                "match every dim_model candidate when a fixed single-branch "
                "ingestion_spec is provided. Provide ingestion_spec as a list "
                "paired with dim_model for variable widths."
            )

        return self

    def sample_trial(self, trial: Any) -> ModelSpecModel:
        """Sample architecture hyperparameters for one Optuna trial."""
        dim_model_idx = trial.suggest_categorical(
            "dim_model_idx", list(range(len(self.dim_model)))
        )

        dim_model = self.dim_model[dim_model_idx]
        n_head = self.n_head[dim_model_idx]
        if isinstance(self.ingestion_spec, list):
            ingestion_spec = self.ingestion_spec[dim_model_idx]
        else:
            ingestion_spec = self.ingestion_spec
        if isinstance(self.ingestion_merge, list):
            ingestion_merge = self.ingestion_merge[dim_model_idx]
        else:
            ingestion_merge = self.ingestion_merge

        dim_feedforward = sample_param(trial, "dim_feedforward", self.dim_feedforward)
        num_layers = sample_param(trial, "num_layers", self.num_layers)
        rope_theta = sample_param(trial, "rope_theta", self.rope_theta)

        activation_fn = trial.suggest_categorical("activation_fn", self.activation_fn)
        normalization = trial.suggest_categorical("normalization", self.normalization)
        positional_encoding = trial.suggest_categorical(
            "positional_encoding", self.positional_encoding
        )
        attention_type = trial.suggest_categorical(
            "attention_type", self.attention_type
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
            f"{dim_model = } - {dim_feedforward = } - {num_layers = } - {activation_fn = } - {normalization = } - {positional_encoding = } - {attention_type = } - {norm_first = } - {n_kv_heads = } - {rope_theta = } "
        )

        model_spec_kwargs = {
            "dim_model": dim_model,
            "n_head": n_head,
            "dim_feedforward": dim_feedforward,
            "num_layers": num_layers,
            "activation_fn": activation_fn,
            "normalization": normalization,
            "positional_encoding": positional_encoding,
            "attention_type": attention_type,
            "norm_first": norm_first,
            "n_kv_heads": n_kv_heads,
            "rope_theta": rope_theta,
            "prediction_length": self.prediction_length,
        }
        if ingestion_spec is not None:
            model_spec_kwargs["ingestion_spec"] = ingestion_spec
        if ingestion_merge is not None:
            model_spec_kwargs["ingestion_merge"] = ingestion_merge
        model_spec_kwargs["allow_shared_ingestion_columns"] = (
            self.allow_shared_ingestion_columns
        )
        model_spec_kwargs["allow_unused_input_columns"] = (
            self.allow_unused_input_columns
        )
        model_spec_kwargs["auxiliary_input_columns"] = self.auxiliary_input_columns

        return ModelSpecModel(**model_spec_kwargs)


class HyperparameterSearchConfig(BaseModel):
    """Top-level Optuna search config."""

    project_root: str
    metadata_config_path: str
    hp_search_name: str
    search_strategy: str = "bayesian"
    seed: Optional[int] = None
    n_trials: Optional[int] = Field(None, alias="n_samples")
    prune_trials: Optional[bool] = True
    model_config_write_path: str
    training_data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    input_columns: list[list[str]]
    column_types: list[dict[str, str]]
    categorical_columns: list[list[str]]
    real_columns: list[list[str]]
    target_columns: list[str]
    target_column_types: dict[str, str]
    id_maps: dict[str, dict[str | int, int]]

    context_length: list[int]
    storage_layout: StoredWindowLayout
    n_classes: dict[str, int]
    inference_batch_size: int

    export_generative_model: bool
    export_embedding_model: bool
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
        return self

    @model_validator(mode="after")
    def validate_prune_trials(self):
        if self.prune_trials and self.training_hyperparameter_sampling.distributed:
            warnings.warn(
                "Trial pruning in distributed training settings is in beta mode."
            )
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

    @field_validator("column_types")
    @classmethod
    def validate_model_spec(cls, v, info):
        if v is not None:
            if not (len(info.data.get("input_columns")) == len(v)):
                raise ValueError(
                    "input_columns and column_types must have the same number of candidate values, that are paired"
                )
        return v

    @field_validator("search_strategy")
    @classmethod
    def validate_search_strategy(cls, v: str) -> str:
        allowed = ["sample", "grid", "bayesian"]
        if v not in allowed:
            raise ValueError(f"search_strategy must be one of {allowed}, got '{v}'")
        return v

    def sample_trial(self, trial: Any, run_index: int) -> TrainModel:
        """Sample a concrete TrainModel for one trial/run index."""
        model_spec = self.model_hyperparameter_sampling.sample_trial(trial)

        input_columns_index = trial.suggest_categorical(
            "input_columns_index", list(range(len(self.input_columns)))
        )
        context_length = trial.suggest_categorical(
            "context_length", self.context_length
        )
        training_spec = self.training_hyperparameter_sampling.sample_trial(trial)
        objective_class = get_objective_class(training_spec.training_objective)
        if not objective_class.forward_looking:
            model_spec = model_spec.model_copy(
                update={"prediction_length": context_length}
            )

        window_view = ModelWindowView(
            context_length=context_length,
            objective=training_spec.training_objective,
            target_offset=target_offset_for_objective(
                training_spec.training_objective,
                1,
            ),
        )
        resolve_window_view(self.storage_layout, window_view)

        logger.info(f"{input_columns_index = } - {context_length = }")

        return TrainModel(
            project_root=self.project_root,
            metadata_config_path=self.metadata_config_path,
            model_name=f"{self.hp_search_name}-run-{run_index}",
            training_data_path=self.training_data_path,
            validation_data_path=self.validation_data_path,
            read_format=self.read_format,
            input_columns=self.input_columns[input_columns_index],
            column_types=self.column_types[input_columns_index],
            categorical_columns=self.categorical_columns[input_columns_index],
            real_columns=self.real_columns[input_columns_index],
            target_columns=self.target_columns,
            target_column_types=self.target_column_types,
            id_maps=self.id_maps,
            storage_layout=self.storage_layout,
            window_view=window_view,
            n_classes=self.n_classes,
            inference_batch_size=self.inference_batch_size,
            seed=101,
            export_embedding_model=self.export_embedding_model,
            export_generative_model=self.export_generative_model,
            export_onnx=self.export_onnx,
            export_pt=self.export_pt,
            export_with_dropout=self.export_with_dropout,
            feature_layout=self.feature_layout,
            model_spec=model_spec,
            training_spec=training_spec,
        )
