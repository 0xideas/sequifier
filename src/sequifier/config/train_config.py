import copy
import json
import math
import os
import warnings
from itertools import product
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
import torch
import torch_optimizer
import yaml
from beartype import beartype
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

import sequifier
from sequifier.config.probabilities import ProbabilityDistribution
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
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS, validate_special_token_ids

AnyType = str | int | float
NextOccurrenceTargetValue = StrictInt | StrictStr


class CartesianLayoutModel(BaseModel):
    """Reusable coordinate annotation for flat feature columns."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cartesian"]
    axis_order: list[str] = Field(..., min_length=1)
    axes: dict[str, list[AnyType]] = Field(..., min_length=1)
    columns: dict[str, dict[str, AnyType]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_cartesian(self):
        if self.axis_order != list(self.axes.keys()):
            raise ValueError("axis_order must exactly match axes keys")

        for axis, values in self.axes.items():
            if not values:
                raise ValueError(
                    f"Layout axis {axis!r} must contain at least one value"
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Layout axis {axis!r} cannot contain duplicate values"
                )

        coordinate_tuples = set()
        for column_name, coordinates in self.columns.items():
            if set(coordinates) != set(self.axis_order):
                raise ValueError(
                    f"Layout column {column_name!r} must define every axis"
                )

            coordinate_tuple = tuple(coordinates[axis] for axis in self.axis_order)
            if coordinate_tuple in coordinate_tuples:
                raise ValueError(
                    f"Duplicate cartesian coordinate tuple: {coordinate_tuple!r}"
                )
            coordinate_tuples.add(coordinate_tuple)

            for axis, value in coordinates.items():
                if value not in self.axes[axis]:
                    raise ValueError(
                        f"Layout column {column_name!r} has value {value!r} "
                        f"outside axis {axis!r}"
                    )

        expected_tuples = set(product(*(self.axes[axis] for axis in self.axis_order)))
        if coordinate_tuples != expected_tuples:
            raise ValueError("cartesian layouts must contain every coordinate")

        return self


class FeatureLayoutRegistryModel(BaseModel):
    """Top-level registry of reusable feature layouts."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    layouts: dict[str, CartesianLayoutModel] = Field(..., min_length=1)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        if v != 1:
            raise ValueError("Only feature_layout version 1 is supported")
        return v


class DirectEmbedIngestionConfig(BaseModel):
    """Use the existing flat-column embedding path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["direct_embed"] = "direct_embed"
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: Optional[int] = Field(default=None, gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "direct_embed ingestion columns")
        return v


class PassThroughIngestionConfig(BaseModel):
    """Pass real-valued columns through without per-feature encoders."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pass_through"]
    columns: Optional[list[str]] = Field(default=None, min_length=1)
    output_dim: Optional[int] = Field(default=None, gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        if v is not None:
            _validate_column_list_unique(v, "pass_through ingestion columns")
        return v


class FeaturePoolIngestionConfig(BaseModel):
    """Encode columns as feature tokens before pooling to one time token."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["feature_pool"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        _validate_column_list_unique(v, "feature_pool ingestion columns")
        return v


class GroupedIngestionConfig(BaseModel):
    """Encode configured column groups and merge them within one branch."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["grouped"]
    output_dim: int = Field(..., gt=0)
    groups: dict[str, list[str]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_groups(self):
        grouped_columns = []
        for group_name, columns in self.groups.items():
            if not columns:
                raise ValueError(
                    f"grouped ingestion group {group_name!r} must contain columns"
                )
            _validate_column_list_unique(
                columns, f"grouped ingestion group {group_name!r}"
            )
            grouped_columns.extend(columns)

        _validate_column_list_unique(grouped_columns, "grouped ingestion columns")
        return self


class SiameseIngestionConfig(BaseModel):
    """Apply one shared scalar encoder across the branch columns."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["siamese"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v):
        _validate_column_list_unique(v, "siamese ingestion columns")
        return v


class TemporalConvIngestionConfig(BaseModel):
    """Encode columns, then apply Conv1D over the time axis."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["temporal_conv"]
    columns: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    kernel_size: int = Field(3, gt=0)
    dilation: int = Field(1, gt=0)
    num_layers: int = Field(1, gt=0)
    causal: bool = True
    activation_fn: Literal["relu", "gelu", "silu"] = "gelu"
    dropout: float = Field(0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_temporal_conv(self):
        _validate_column_list_unique(self.columns, "temporal_conv ingestion columns")
        if not self.causal and self.kernel_size % 2 == 0:
            raise ValueError(
                "temporal_conv kernel_size must be odd when causal is false"
            )
        return self


class AxisProjectionBlockModel(BaseModel):
    """Flatten configured cartesian axes and project them with a linear layer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_projection"]
    axes: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        return self


class AxisConvBlockModel(BaseModel):
    """Sweep a native 1D/2D/3D convolution over configured cartesian axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_conv"]
    axes: list[str] = Field(..., min_length=1, max_length=3)
    output_dim: int = Field(..., gt=0)
    kernel_size: int = Field(3, gt=0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        if self.kernel_size % 2 == 0:
            raise ValueError("axis_conv kernel_size must be odd to preserve axis sizes")
        return self


class AxisAttentionBlockModel(BaseModel):
    """Apply self-attention over one or more configured cartesian axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_attention"]
    axes: list[str] = Field(..., min_length=1)
    output_dim: int = Field(..., gt=0)
    n_head: int = Field(1, gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    unshared_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        _validate_axis_list_unique(self.unshared_axes, "unshared_axes")
        if self.output_dim % self.n_head != 0:
            raise ValueError("axis_attention output_dim must be divisible by n_head")
        return self


class AxisPoolBlockModel(BaseModel):
    """Reduce configured cartesian axes without changing the channel dimension."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["axis_pool"]
    axes: list[str] = Field(..., min_length=1)
    mode: Literal["mean", "sum", "max"] = "mean"

    @model_validator(mode="after")
    def validate_axes_unique(self):
        _validate_axis_list_unique(self.axes, "axes")
        return self


StructuredProcessingBlock = Annotated[
    Union[
        AxisProjectionBlockModel,
        AxisConvBlockModel,
        AxisAttentionBlockModel,
        AxisPoolBlockModel,
    ],
    Field(discriminator="type"),
]


def _validate_axis_list_unique(axes: list[str], field_name: str) -> None:
    if len(axes) != len(set(axes)):
        raise ValueError(f"{field_name} cannot contain duplicate axes")


def _validate_column_list_unique(columns: list[str], field_name: str) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError(f"{field_name} cannot contain duplicate columns")


class AxisEmbeddingModel(BaseModel):
    """Optional positional encoding for cartesian layout axes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "learned", "rope"] = "none"
    axes: list[str] = Field(default_factory=list)
    rope_theta: float = Field(10000.0, gt=0.0)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v):
        if v is None:
            return "none"
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def validate_axes(self):
        _validate_axis_list_unique(self.axes, "axis_embeddings.axes")
        if self.type == "none" and self.axes:
            raise ValueError("axis_embeddings.axes must be empty when type is 'none'")
        if self.type != "none" and not self.axes:
            raise ValueError(
                "axis_embeddings.axes must contain at least one axis unless type is 'none'"
            )
        return self


class StructuredIngestionConfig(BaseModel):
    """Consume a top-level cartesian layout."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["structured"]
    layout: str
    output_dim: int = Field(..., gt=0)
    cell_dim: Optional[int] = Field(default=None, gt=0)
    axis_embeddings: AxisEmbeddingModel = Field(default_factory=AxisEmbeddingModel)
    processing_blocks: list[StructuredProcessingBlock] = Field(default_factory=list)

    @field_validator("axis_embeddings", mode="before")
    @classmethod
    def normalize_axis_embeddings(cls, v):
        if v is None:
            return {"type": "none", "axes": []}
        if isinstance(v, list):
            return {"type": "learned", "axes": v}
        return v


BranchIngestionConfig = Annotated[
    Union[
        DirectEmbedIngestionConfig,
        PassThroughIngestionConfig,
        FeaturePoolIngestionConfig,
        GroupedIngestionConfig,
        SiameseIngestionConfig,
        TemporalConvIngestionConfig,
        StructuredIngestionConfig,
    ],
    Field(discriminator="type"),
]


class IngestionMergeConfig(BaseModel):
    """How composite branch outputs are merged."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["concat", "sum", "gated", "attention"] = "concat"
    output_dim: int = Field(..., gt=0)


IngestionLayerConfig = BranchIngestionConfig | dict[str, BranchIngestionConfig]


def _validate_class_share_log_columns(config_values: dict[str, Any]) -> None:
    training_spec = config_values.get("training_spec", {})

    for col in training_spec.get("class_share_log_columns", []):
        if col not in config_values["target_columns"]:
            raise ValueError(f"Class-share column {col!r} must be a target column.")
        if config_values["target_column_types"].get(col) != "categorical":
            raise ValueError(
                f"Class-share column {col!r} must be a categorical target column."
            )
        if col not in config_values["n_classes"]:
            raise ValueError(
                f"Class-share column {col!r} has no configured class count."
            )
        if col not in config_values["id_maps"]:
            raise ValueError(
                f"Class-share column {col!r} has no index map for logging."
            )


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> "TrainModel":
    """Load train YAML plus CLI overrides and optional metadata-derived fields."""
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    config_values["seed"] = config_values.get("seed", 1010)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_root"]), "r"
        ) as f:
            metadata_config = json.loads(f.read())

        storage_layout = stored_window_layout_from_metadata(metadata_config)
        if storage_layout.version != 2:
            raise ValueError(
                "Training requires metadata stored_window_layout_version=2, "
                f"got {storage_layout.version}."
            )
        training_objective = config_values["training_spec"]["training_objective"]
        target_offset = target_offset_for_objective(
            training_objective,
            int(config_values.pop("target_offset", 1)),
        )
        window_view = ModelWindowView(
            context_length=int(config_values.pop("context_length")),
            objective=training_objective,
            target_offset=target_offset,
        )
        resolve_window_view(storage_layout, window_view)
        config_values["storage_layout"] = storage_layout
        config_values["window_view"] = window_view
        for key in (
            "stored_context_width",
            "max_target_offset",
            "stored_window_layout_version",
        ):
            config_values.pop(key, None)
        for key in (
            "target_offset",
            "stored_context_width",
            "max_target_offset",
            "stored_window_layout_version",
        ):
            config_values.pop(key, None)
        for key in ("target_offset", "stored_context_width", "max_target_offset"):
            config_values.get("training_spec", {}).pop(key, None)

        split_paths = metadata_config["split_paths"]

        config_values["column_types"] = config_values.get(
            "column_types", metadata_config["column_types"]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = list(config_values["column_types"].keys())

        config_values["categorical_columns"] = [
            col
            for col, type_ in metadata_config["column_types"].items()
            if "int" in type_.lower() and col in config_values["input_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in metadata_config["column_types"].items()
            if "float" in type_.lower() and col in config_values["input_columns"]
        ]
        if not (
            len(config_values["real_columns"] + config_values["categorical_columns"])
            > 0
        ):
            raise ValueError("No columns found in config_values")
        config_values["n_classes"] = config_values.get(
            "n_classes", metadata_config["n_classes"]
        )
        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", split_paths[0]),
            config_values["project_root"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                split_paths[min(1, len(split_paths) - 1)],
            ),
            config_values["project_root"],
        )

        config_values["id_maps"] = metadata_config["id_maps"]
        config_values["special_token_ids"] = validate_special_token_ids(
            metadata_config.get(
                "special_token_ids",
                SPECIAL_TOKEN_IDS.ids_by_label,
            ),
            source=f"metadata config '{metadata_config_path}'",
        )

        _validate_class_share_log_columns(config_values)

    return try_catch_excess_keys(config_path, TrainModel, config_values)


class DotDict(dict):
    """Dot notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class ReplacementDistribution(BaseModel):
    masked: float = Field(..., ge=0.0, le=1.0)
    random: float = Field(..., ge=0.0, le=1.0)
    identical: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.masked + self.random + self.identical
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Replacement distribution probabilities must sum to 1.0, got {total}"
            )
        return self


class BERTSpecModel(BaseModel):
    masking_probability: float = Field(..., gt=0.0, le=1.0)
    replacement_distribution: ReplacementDistribution
    span_masking: ProbabilityDistribution


class NextOccurrenceConfigModel(BaseModel):
    column_name: str
    target_values: list[NextOccurrenceTargetValue] = Field(..., min_length=1)


class TrainingSpecModel(BaseModel):
    """Training loop, optimization, precision, and distribution settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    training_objective: str
    device: str
    device_max_concat_length: int = 12
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
    save_latest_interval_minutes: Optional[float] = None
    save_interval_minutes: Optional[float] = None
    save_interval_batches: Optional[int] = None
    save_interval_val_loss: bool = True
    calculate_validation_loss_on_initialization: bool = True
    batch_size: int
    learning_rate: float
    criterion: dict[str, str]
    class_weights: Optional[dict[str, list[float]]] = None
    accumulation_steps: Optional[int] = None
    dropout: float = 0.0
    loss_weights: Optional[dict[str, float]] = None
    optimizer: DotDict = Field(default_factory=lambda: DotDict({"name": "Adam"}))
    scheduler: DotDict = Field(
        default_factory=lambda: DotDict(
            {"name": "StepLR", "step_size": 1, "gamma": 0.99}
        )
    )
    scheduler_step_on: str = "epoch"
    bert_spec: Optional[BERTSpecModel] = None
    next_occurrence_config: Optional[NextOccurrenceConfigModel] = None

    continue_training: bool = True
    enforce_determinism: bool = False
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: Union[int, float] = 16
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
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        self.validate_optimizer_config(kwargs["optimizer"])
        self.optimizer = DotDict(kwargs["optimizer"])
        self.validate_scheduler_config(kwargs["scheduler"], kwargs)
        self.scheduler = DotDict(kwargs["scheduler"])

    @field_serializer("optimizer", "scheduler")
    def serialize_dotdict(self, value: DotDict) -> dict[str, Any]:
        return dict(value)

    @field_validator("layer_type_dtypes")
    @classmethod
    def validate_layer_type_dtypes(cls, v):
        expected_keys = ["embedding", "linear", "norm", "decoder"]
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

    @field_validator("float32_matmul_precision")
    @classmethod
    def validate_float32_matmul_precision(cls, v):
        allowed_precisions = ["highest", "high", "medium"]
        if v not in allowed_precisions:
            raise ValueError(
                f"float32_matmul_precision must be one of {allowed_precisions}, got '{v}'"
            )
        return v

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, v):
        for vv in v.values():
            if not hasattr(torch.nn, vv):
                raise ValueError(f"{vv} not in torch.nn")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer_config(cls, v):
        if "name" not in v:
            raise ValueError("optimizer dict must specify 'name' field")
        if (
            not hasattr(torch.optim, v["name"])
            and not hasattr(torch_optimizer, v["name"])
            and not hasattr(sequifier.optimizers, v["name"])  # type: ignore
        ):
            raise ValueError(f"{v['name']} not in torch.optim or in torch_optimizer")
        return v

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler_config(cls, v, info_dict):
        if "name" not in v:
            raise ValueError("scheduler dict must specify 'name' field")
        if not hasattr(torch.optim.lr_scheduler, v["name"]):
            raise ValueError(f"{v} not in torch.optim.lr_scheduler")
        if "total_steps" in v:
            if info_dict.get("scheduler_step_on") == "epoch":
                if not v["total_steps"] == info_dict.get("epochs"):
                    raise ValueError(
                        f"scheduler total steps: {v['total_steps']} != {info_dict.get('epochs')}: total epochs"
                    )
            else:
                logger.info(
                    f"[WARNING] {v['total_steps']} scheduler steps at {info_dict.get('epochs')} epochs implies {v['total_steps']/info_dict.get('epochs'):.2f} batches. Does this seem correct?"
                )
        return v

    @field_validator("scheduler_step_on")
    @classmethod
    def validate_scheduler_step_on(cls, v):
        if v not in ["epoch", "batch"]:
            raise ValueError(
                f"scheduler_step_on must be in ['epoch', 'batch'], {v} isn't"
            )
        return v

    @field_validator("training_objective")
    @classmethod
    def validate_training_objective(cls, v):
        if v not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {v}")
        return v

    @model_validator(mode="after")
    def validate_objective_specific_config(self):
        objective_class = get_objective_class(self.training_objective)
        is_bert_objective = issubclass(objective_class, BERTObjective)
        is_next_occurrence_objective = issubclass(
            objective_class,
            NextOccurrenceObjective,
        )
        if self.bert_spec is not None and not is_bert_objective:
            raise ValueError(
                "The BERT hyperparameters should only be configured if the training objective is 'bert'"
            )
        if self.bert_spec is None and is_bert_objective:
            raise ValueError(
                "If the training_objective is 'bert', the BERT hyperparameters must be set"
            )
        if self.next_occurrence_config is not None and not is_next_occurrence_objective:
            raise ValueError(
                "next_occurrence_config should only be configured if the training objective is 'next_occurrence'"
            )
        if self.next_occurrence_config is None and is_next_occurrence_objective:
            raise ValueError(
                "If the training_objective is 'next_occurrence', next_occurrence_config must be set"
            )
        return self

    @field_validator("data_parallelism")
    @classmethod
    def validate_data_parallelism(cls, v):
        if v is not None and v not in ["DDP", "FSDP"]:
            raise ValueError(
                f"data_parallelism must be None, or 'DDP' or 'FSDP', got '{v}'"
            )
        return v


class ModelSpecModel(BaseModel):
    """Transformer architecture settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    initial_embedding_dim: int
    feature_embedding_dims: Optional[dict[str, int]] = None
    ingestion_layer_config: IngestionLayerConfig = Field(
        default_factory=DirectEmbedIngestionConfig
    )
    ingestion_merge: Optional[IngestionMergeConfig] = None
    allow_shared_ingestion_columns: bool = False
    dim_model: int
    n_head: int
    dim_feedforward: int
    num_layers: int

    activation_fn: str = "swiglu"  # Options: "relu", "gelu", "swiglu"
    normalization: str = "rmsnorm"  # Options: "layer_norm", "rmsnorm"
    positional_encoding: str = "learned"  # Options: "learned", "rope" (Rotary)
    attention_type: str = (
        "mha"  # Options: "mha" (Multi-Head), "mqa" (Multi-Query), "gqa" (Grouped-Query)
    )

    norm_first: bool = True
    n_kv_heads: Optional[int] = None
    rope_theta: float = 10000.0

    prediction_length: int

    @field_validator("dim_model")
    @classmethod
    def validate_dim_model(cls, v, info):
        initial_embedding_dim = info.data.get("initial_embedding_dim")
        ingestion_layer_config = info.data.get("ingestion_layer_config")
        dim_model = v

        if isinstance(ingestion_layer_config, dict):
            return v

        if (
            ingestion_layer_config is not None
            and getattr(ingestion_layer_config, "output_dim", None) is not None
        ):
            return v

        if isinstance(ingestion_layer_config, PassThroughIngestionConfig):
            return v

        if not v == initial_embedding_dim:
            raise ValueError(
                "If ingestion_layer_config.output_dim is not configured, "
                "dim_model must be equal to initial_embedding_dim, "
                f"{dim_model = } != {initial_embedding_dim = }"
            )

        return v

    @model_validator(mode="after")
    def validate_ingestion_layer_output_dim(self):
        if isinstance(self.ingestion_layer_config, dict):
            if self.ingestion_merge is None:
                self.ingestion_merge = IngestionMergeConfig(output_dim=self.dim_model)
            if self.ingestion_merge.output_dim != self.dim_model:
                raise ValueError(
                    "model_spec.ingestion_merge.output_dim must equal dim_model"
                )
        elif (
            getattr(self.ingestion_layer_config, "output_dim", None) is not None
            and self.ingestion_layer_config.output_dim != self.dim_model
        ):
            raise ValueError(
                "model_spec.ingestion_layer_config.output_dim must equal dim_model"
            )
        elif self.ingestion_merge is not None:
            raise ValueError(
                "model_spec.ingestion_merge is only valid when "
                "ingestion_layer_config defines multiple named ingestions"
            )

        return self

    @field_validator("activation_fn")
    @classmethod
    def validate_activation(cls, v):
        if v not in ["relu", "gelu", "swiglu"]:
            raise ValueError(f"Invalid activation_fn: {v}")
        return v

    @field_validator("normalization")
    @classmethod
    def validate_normalization(cls, v):
        if v not in ["layer_norm", "rmsnorm"]:
            raise ValueError(f"Invalid normalization: {v}")
        return v

    @field_validator("positional_encoding")
    @classmethod
    def validate_pos_encoding(cls, v):
        if v not in ["learned", "rope"]:
            raise ValueError(f"Invalid positional_encoding: {v}")
        return v

    @field_validator("attention_type")
    @classmethod
    def validate_attention_type(cls, v):
        if v not in ["mha", "mqa", "gqa"]:
            raise ValueError(f"Invalid attention_type: {v}")
        return v

    @field_validator("feature_embedding_dims")
    @classmethod
    def validate_feature_embedding_dims(cls, v, info):
        initial_embedding_dim = info.data.get("initial_embedding_dim")
        if (
            v is not None
            and initial_embedding_dim
            and sum(v.values()) != initial_embedding_dim
        ):
            raise ValueError(
                f"Sum of feature_embedding_dims {sum(v.values())} != initial_embedding_dim {initial_embedding_dim}"
            )
        return v

    @field_validator("n_head")
    @classmethod
    def validate_n_head(cls, v, info):
        dim_model = info.data.get("dim_model")
        if v is None:
            raise ValueError("n_heads is None")
        if dim_model is None:
            raise ValueError("dim_model is None")
        if dim_model % v != 0:
            raise ValueError(f"dim_model {dim_model} not divisible by n_head {v}")
        return v

    @field_validator("n_kv_heads")
    @classmethod
    def validate_n_kv_heads(cls, v, info):
        n_head = info.data.get("n_head")
        attn_type = info.data.get("attention_type")

        if v is not None:
            if n_head and n_head % v != 0:
                raise ValueError(f"n_head {n_head} not divisible by n_kv_heads {v}")
            if n_head and v > n_head:
                raise ValueError(f"n_kv_heads {v} > n_head {n_head}")

            if attn_type == "mqa" and v != 1:
                raise ValueError(f"n_kv_heads must be 1 for mqa, got {v}")
            if attn_type == "mha" and v != n_head:
                raise ValueError(f"n_kv_heads must equal n_head for mha, got {v}")
        else:
            if attn_type in ["gqa", "mqa"]:
                raise ValueError(f"n_kv_heads must be specified for {attn_type}")

        return v


class TrainModel(BaseModel):
    """Top-level training config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project_root: str
    metadata_config_path: str
    model_name: str
    training_data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    input_columns: list[str]
    column_types: dict[str, str]
    categorical_columns: list[str]
    real_columns: list[str]
    target_columns: list[str]
    target_column_types: dict[str, str]
    id_maps: dict[str, dict[str | int, int]]
    special_token_ids: dict[str, int] = Field(
        default_factory=lambda: SPECIAL_TOKEN_IDS.ids_by_label
    )

    storage_layout: StoredWindowLayout
    window_view: ModelWindowView
    n_classes: dict[str, int]
    inference_batch_size: int
    seed: int

    export_generative_model: bool
    export_embedding_model: bool
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    feature_layout: Optional[FeatureLayoutRegistryModel] = None
    model_spec: ModelSpecModel
    training_spec: TrainingSpecModel

    @field_validator("special_token_ids")
    @classmethod
    def validate_special_token_ids_match_runtime(cls, v):
        return validate_special_token_ids(v, source="TrainModel")

    @model_validator(mode="after")
    def validate_bert_prediction_length_matches_context_length(self):
        if self.window_view.objective != self.training_spec.training_objective:
            raise ValueError(
                "window_view objective must match training_spec.training_objective "
                f"({self.window_view.objective} != {self.training_spec.training_objective})."
            )
        resolve_window_view(self.storage_layout, self.window_view)
        get_objective_class(
            self.training_spec.training_objective
        ).validate_prediction_length(
            self.model_spec.prediction_length,
            self.window_view.context_length,
            usage="training",
        )
        return self

    @model_validator(mode="after")
    def validate_next_occurrence_config_matches_targets(self):
        objective_class = get_objective_class(self.training_spec.training_objective)
        if issubclass(objective_class, NextOccurrenceObjective):
            next_occurrence_config = self.training_spec.next_occurrence_config
            if next_occurrence_config is None:
                raise ValueError(
                    "next_occurrence_config must be set for next_occurrence training."
                )

            column_name = next_occurrence_config.column_name
            if column_name not in self.target_columns:
                raise ValueError(
                    "next_occurrence_config.column_name must be one of target_columns, "
                    f"got {column_name!r}."
                )
            if self.target_column_types.get(column_name) != "categorical":
                raise ValueError(
                    "next_occurrence_config.column_name must refer to a categorical target column."
                )
            if column_name not in self.id_maps:
                raise ValueError(
                    "next_occurrence_config.column_name must have a preprocessing id_map, "
                    f"got {column_name!r}."
                )

            id_map = self.id_maps[column_name]
            missing_values = [
                value
                for value in next_occurrence_config.target_values
                if value not in id_map
            ]
            if missing_values:
                raise ValueError(
                    "next_occurrence_config.target_values must match keys in "
                    f"id_maps[{column_name!r}] exactly, missing {missing_values!r}."
                )
        return self

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not "embedding" not in v:
            raise ValueError("model_name cannot contain 'embedding'")
        return v

    @field_validator("target_column_types")
    @classmethod
    def validate_target_column_types(cls, v, info):
        if not all(vv in ["categorical", "real"] for vv in v.values()):
            raise ValueError(
                f"Invalid target_column_types found: {[vv not in ['categorical', 'real'] for vv in v.values()]}. Only 'categorical' and 'real' are allowed."
            )
        if not (list(v.keys()) == info.data.get("target_columns")):
            raise ValueError(
                "target_columns and target_column_types must contain the same values/keys in the same order"
            )
        return v

    @field_validator("read_format")
    @classmethod
    def validate_read_format(cls, v):
        if v not in [
            "csv",
            "parquet",
            "pt",
        ]:
            raise ValueError("Currently only 'csv', 'parquet' and 'pt' are supported")
        return v

    @field_validator("training_spec")
    @classmethod
    def validate_training_spec(cls, v, info):
        if not set(info.data.get("target_columns")) == set(v.criterion.keys()):
            raise ValueError(
                "target_columns and criterion must contain the same values/keys"
            )

        if v.distributed:
            if info.data.get("read_format") not in ["pt", "parquet"]:
                raise ValueError(
                    "If distributed is set to 'true', the format must be 'pt' or 'parquet' representing a folder dataset."
                )
            if info.data.get("read_format") == "parquet":
                warnings.warn(
                    "Training on distributed data in parquet format takes significantly more CPU per GPU than with 'pt'."
                )

        if (
            v.save_latest_interval_minutes is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_latest_interval_minutes == 0
        ):
            raise ValueError("save_latest_interval_minutes must be larger than 0")

        if (
            v.save_interval_minutes is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_interval_minutes == 0
        ):
            raise ValueError("save_interval_minutes must be larger than 0")

        if (
            v.save_interval_batches is not None
            and not os.getenv("SEQUIFIER_TESTING", "0") == "1"
            and v.save_interval_batches == 0
        ):
            raise ValueError("save_interval_batches must be larger than 0")

        if v.torch_compile not in ["outer", "inner", "none"]:
            raise ValueError(
                f'torch_compile {v.torch_compile} invalid, must be one of ["outer", "inner", "none"]'
            )

        if v.data_parallelism == "FSDP":
            if v.layer_type_dtypes is not None:
                raise ValueError(
                    "FSDP does not support manual layer pre-casting. Please set "
                    "'layer_type_dtypes' to null when using FSDP, and rely on "
                    "'layer_autocast' (MixedPrecisionPolicy) instead."
                )
            if v.fsdp_cpu_offload is None:
                raise ValueError(
                    "If data_parallelism == 'FSDP', fsdp_cpu_offload cannot be None"
                )

        if v.data_parallelism == "FSDP" and v.torch_compile == "outer":
            raise ValueError(
                "If data_parallelism is set to 'FSDP' then torch_compile must be one of 'none' and 'inner'"
            )

        if v.data_parallelism == "DDP" and v.torch_compile == "inner":
            raise ValueError(
                "If data_parallelism is set to 'DDP' then torch_compile must be one of 'none' and 'outer'"
            )

        if v.data_parallelism is None or v.data_parallelism != "FSDP":
            if v.fsdp_cpu_offload is not None:
                raise ValueError(
                    "If data_parallelism != 'FSDP', fsdp_cpu_offload must be None"
                )
        if v.data_parallelism == "FSDP":
            if v.fsdp_cpu_offload is None:
                raise ValueError(
                    "If data_parallelism == 'FSDP', fsdp_cpu_offload cannot be None"
                )

        if v.distributed and v.data_parallelism is None:
            raise ValueError(
                "If 'distributed' is True, data_parallelism cannot be 'None'"
            )

        export_generative_model = info.data.get("export_generative_model")
        export_embedding_model = info.data.get("export_embedding_model")
        if (
            not export_generative_model
            and not export_embedding_model
            and os.getenv("SEQUIFIER_PREVENT_EXPORT") is None
        ):
            raise ValueError(
                "At least one of 'export_generative_model' and 'export_embedding_model' must be true. If you want to override this, set the env variable 'SEQUIFIER_PREVENT_EXPORT' to any value"
            )

        return v

    @field_validator("column_types")
    @classmethod
    def validate_column_types(cls, v, info):
        target_columns = info.data.get("target_columns", [])
        column_ordered = list(v.keys())
        columns_ordered_filtered = [c for c in column_ordered if c in target_columns]
        if not (columns_ordered_filtered == target_columns):
            raise ValueError(f"{columns_ordered_filtered = } != {target_columns = }")
        return v

    @field_validator("model_spec")
    @classmethod
    def validate_model_spec(cls, v, info):
        input_columns = info.data.get("input_columns")
        categorical_columns = info.data.get("categorical_columns", [])
        real_columns = info.data.get("real_columns", [])
        categorical_set = set(categorical_columns)
        real_set = set(real_columns)

        if not (
            input_columns is None
            or (v.feature_embedding_dims is None)
            or np.all(
                np.array(list(v.feature_embedding_dims.keys()))
                == np.array(list(input_columns))
            )
        ):
            raise ValueError(
                "If feature_embedding_dims is not None, dimensions must be specified for all input columns"
            )

        direct_embed_column_groups: list[tuple[list[str], int]] = []
        if isinstance(v.ingestion_layer_config, DirectEmbedIngestionConfig):
            direct_embed_column_groups.append(
                (
                    v.ingestion_layer_config.columns or input_columns or [],
                    v.initial_embedding_dim,
                )
            )
        elif isinstance(v.ingestion_layer_config, TemporalConvIngestionConfig):
            direct_embed_column_groups.append(
                (v.ingestion_layer_config.columns, v.ingestion_layer_config.output_dim)  # type: ignore
            )
        elif isinstance(v.ingestion_layer_config, dict):
            for branch in v.ingestion_layer_config.values():
                if isinstance(
                    branch,
                    (DirectEmbedIngestionConfig, TemporalConvIngestionConfig),
                ):
                    if branch.columns is None:
                        continue
                    direct_embed_column_groups.append(
                        (
                            branch.columns,
                            branch.output_dim or v.initial_embedding_dim,
                        )
                    )

        for direct_embed_columns, embedding_size in direct_embed_column_groups:
            n_categorical = len(
                [col for col in direct_embed_columns if col in categorical_set]
            )
            n_real = len([col for col in direct_embed_columns if col in real_set])

            if n_categorical > 0 and n_real > 0 and v.feature_embedding_dims is None:
                raise ValueError(
                    "If both real and categorical variables are present, 'feature_embedding_dims' in 'model_spec' must be set explicitly."
                )

            if n_real > 0 and n_categorical == 0 and v.feature_embedding_dims is None:
                if embedding_size < n_real:
                    raise ValueError(
                        f"initial_embedding_dim ({embedding_size}) must be at least the number of real variables ({n_real})."
                    )

            if n_categorical > 0 and n_real == 0 and v.feature_embedding_dims is None:
                if embedding_size % n_categorical != 0:
                    raise ValueError(
                        f"If only categorical variables are included and feature_embedding_dims is not set, "
                        f"initial_embedding_dim ({embedding_size}) must be a multiple of the number of categorical variables ({n_categorical}: {categorical_columns})."
                    )

        return v

    @model_validator(mode="after")
    def validate_feature_layout_columns(self):
        if self.feature_layout is None:
            return self

        allowed_columns = set(self.input_columns) | set(self.column_types)
        for layout_name, layout in self.feature_layout.layouts.items():
            missing_columns = set(layout.columns) - allowed_columns
            if missing_columns:
                raise ValueError(
                    f"feature_layout {layout_name!r} references unknown columns: "
                    f"{sorted(missing_columns)}"
                )

        return self

    @model_validator(mode="after")
    def validate_ingestion_layer_branches(self):
        ingestion_layer_config = self.model_spec.ingestion_layer_config
        if not isinstance(ingestion_layer_config, dict):
            columns = self._branch_columns(
                ingestion_layer_config,
                default_columns=self.input_columns,
            )
            self._validate_ingestion_columns(
                "model_spec.ingestion_layer_config",
                columns,
            )
            if isinstance(ingestion_layer_config, StructuredIngestionConfig):
                layout = self._layout_for_branch(ingestion_layer_config)
                self._validate_structured_axis_embeddings(
                    ingestion_layer_config, layout
                )
                self._validate_structured_processing_blocks(
                    ingestion_layer_config, layout
                )
            if isinstance(ingestion_layer_config, PassThroughIngestionConfig):
                self._validate_pass_through_ingestion(
                    columns,
                    ingestion_layer_config,
                    usage="model_spec.ingestion_layer_config",
                    require_model_width=True,
                )
            return self

        ingestion_merge = self.model_spec.ingestion_merge
        if ingestion_merge is None:
            raise ValueError(
                "model_spec.ingestion_merge must be configured for multiple ingestions"
            )

        if ingestion_merge.output_dim != self.model_spec.dim_model:
            raise ValueError("Multiple-ingestion merge output_dim must equal dim_model")

        used_columns: dict[str, str] = {}
        for branch_name, ingestion in ingestion_layer_config.items():
            columns = self._branch_columns(ingestion)
            if not columns:
                raise ValueError(
                    f"Ingestion branch {branch_name!r} must resolve to at "
                    "least one input column"
                )
            self._validate_ingestion_columns(branch_name, columns)

            if not self.model_spec.allow_shared_ingestion_columns:
                overlapping_columns = [
                    column for column in columns if column in used_columns
                ]
                if overlapping_columns:
                    raise ValueError(
                        "Ingestion branches cannot share columns unless "
                        "allow_shared_ingestion_columns is true: "
                        f"{sorted(overlapping_columns)}"
                    )
                for column in columns:
                    used_columns[column] = branch_name

            if isinstance(ingestion, StructuredIngestionConfig):
                layout = self._layout_for_branch(ingestion)
                self._validate_structured_axis_embeddings(ingestion, layout)
                self._validate_structured_processing_blocks(ingestion, layout)
            if isinstance(ingestion, PassThroughIngestionConfig):
                self._validate_pass_through_ingestion(
                    columns,
                    ingestion,
                    usage=f"Composite ingestion branch {branch_name!r}",
                    require_model_width=False,
                )
        return self

    def _validate_ingestion_columns(self, usage: str, columns: list[str]) -> None:
        missing_columns = set(columns) - set(self.input_columns)
        if missing_columns:
            raise ValueError(
                f"{usage} references unknown input columns: {sorted(missing_columns)}"
            )

    def _layout_for_branch(
        self,
        ingestion: StructuredIngestionConfig,
    ) -> CartesianLayoutModel:
        if self.feature_layout is None:
            raise ValueError(
                f"Ingestion layout {ingestion.layout!r} requires top-level feature_layout"
            )
        if ingestion.layout not in self.feature_layout.layouts:
            raise ValueError(f"Unknown feature_layout {ingestion.layout!r}")
        return self.feature_layout.layouts[ingestion.layout]

    def _validate_structured_axis_embeddings(
        self,
        ingestion: StructuredIngestionConfig,
        layout: CartesianLayoutModel,
    ) -> None:
        unknown_axes = [
            axis
            for axis in ingestion.axis_embeddings.axes
            if axis not in layout.axis_order
        ]
        if unknown_axes:
            raise ValueError(
                "Structured ingestion axis_embeddings references unavailable axes: "
                f"{unknown_axes}"
            )

        if (
            ingestion.axis_embeddings.type == "rope"
            and (ingestion.cell_dim or ingestion.output_dim) % 2 != 0
        ):
            raise ValueError(
                "Structured ingestion axis_embeddings type 'rope' requires an even "
                "cell_dim/output_dim"
            )

    def _validate_structured_processing_blocks(
        self,
        ingestion: StructuredIngestionConfig,
        layout: CartesianLayoutModel,
    ) -> None:
        active_axes = list(layout.axis_order)
        channel_dim = ingestion.cell_dim or ingestion.output_dim

        for block in ingestion.processing_blocks:
            unknown_axes = [axis for axis in block.axes if axis not in active_axes]
            if unknown_axes:
                raise ValueError(
                    f"Structured ingestion block references unavailable axes: "
                    f"{unknown_axes}"
                )

            if isinstance(
                block,
                (
                    AxisProjectionBlockModel,
                    AxisConvBlockModel,
                    AxisAttentionBlockModel,
                ),
            ):
                available_unshared_axes = [
                    axis for axis in active_axes if axis not in block.axes
                ]
                invalid_unshared_axes = [
                    axis
                    for axis in block.unshared_axes
                    if axis not in available_unshared_axes
                ]
                if invalid_unshared_axes:
                    raise ValueError(
                        "Structured ingestion block unshared_axes must be a subset "
                        "of non-swept active axes: "
                        f"{invalid_unshared_axes}"
                    )

                channel_dim = block.output_dim

            if isinstance(block, (AxisProjectionBlockModel, AxisPoolBlockModel)):
                active_axes = [axis for axis in active_axes if axis not in block.axes]

        if channel_dim != ingestion.output_dim:
            raise ValueError(
                "Structured ingestion processing_blocks must produce output_dim "
                f"{ingestion.output_dim}, got {channel_dim}"
            )

    def _branch_columns(
        self,
        ingestion: BranchIngestionConfig,
        default_columns: Optional[list[str]] = None,
    ) -> list[str]:
        if isinstance(ingestion, StructuredIngestionConfig):
            layout = self._layout_for_branch(ingestion)
            return list(layout.columns)

        if isinstance(ingestion, GroupedIngestionConfig):
            return [
                column
                for group_columns in ingestion.groups.values()
                for column in group_columns
            ]

        if (
            isinstance(
                ingestion,
                (DirectEmbedIngestionConfig, PassThroughIngestionConfig),
            )
            and ingestion.columns is None
        ):
            if default_columns is None:
                raise ValueError(
                    f"{ingestion.type} ingestion branches must configure columns"
                )
            return default_columns

        return ingestion.columns  # type: ignore

    def _validate_pass_through_ingestion(
        self,
        columns: list[str],
        ingestion: PassThroughIngestionConfig,
        *,
        usage: str,
        require_model_width: bool,
    ) -> None:
        categorical_columns = [
            col for col in columns if col in self.categorical_columns
        ]
        real_columns = [col for col in columns if col in self.real_columns]
        if categorical_columns:
            raise ValueError(
                f"{usage} type 'pass_through' only supports real columns; "
                f"got categorical columns {categorical_columns}"
            )
        if not real_columns:
            raise ValueError(f"{usage} type 'pass_through' requires real columns")

        output_dim = ingestion.output_dim or len(real_columns)
        if require_model_width and output_dim != self.model_spec.dim_model:
            raise ValueError(
                f"{usage} output_dim ({output_dim}) must equal dim_model "
                f"({self.model_spec.dim_model})"
            )
