import copy
import json
from typing import Any, Optional, Union

import numpy as np
import torch
import torch_optimizer
import yaml
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

import sequifier
from sequifier.helpers import normalize_path, try_catch_excess_keys

AnyType = str | int | float


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], skip_metadata: bool
) -> "TrainModel":
    """
    Load training configuration from a YAML file and update it with args_config.

    Args:
        config_path: Path to the YAML configuration file.
        args_config: Dictionary containing additional configuration arguments.
        skip_metadata: Flag indicating whether to process the configuration or not.

    Returns:
        TrainModel instance with loaded configuration.
    """
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

    return try_catch_excess_keys(config_path, TrainModel, config_values)


class DotDict(dict):
    """Dot notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


class TrainingSpecModel(BaseModel):
    """Pydantic model for training specifications.

    Attributes:
        device: The torch.device to train the model on (e.g., 'cuda', 'cpu', 'mps').
        device_max_concat_length: Maximum sequence length for concatenation on device.
        epochs: The total number of epochs to train for.
        log_interval: The interval in batches for logging.
        class_share_log_columns: A list of column names for which to log the class share of predictions.
        early_stopping_epochs: Number of epochs to wait for validation loss improvement before stopping.
        save_interval_epochs: The interval in epochs for checkpointing the model.
        batch_size: The training batch size.
        learning_rate: The learning rate.
        criterion: A dictionary mapping each target column to a loss function.
        class_weights: A dictionary mapping categorical target columns to a list of class weights.
        accumulation_steps: The number of gradient accumulation steps.
        dropout: The dropout value for the transformer model.
        loss_weights: A dictionary mapping columns to specific loss weights.
        optimizer: The optimizer configuration.
        scheduler: The learning rate scheduler configuration.
        scheduler_step_on: The time of the .step() call on the scheduler, either 'epoch' or 'batch'
        continue_training: If True, continue training from the latest checkpoint.
        distributed: If True, enables distributed training.
        load_full_data_to_ram: If True, loads the entire dataset into RAM.
        world_size: The number of processes for distributed training.
        num_workers: The number of worker threads for data loading.
        backend: The distributed training backend (e.g., 'nccl').
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device: str
    device_max_concat_length: int = 12
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
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
    continue_training: bool = True
    enforce_determinism: bool = False
    distributed: bool = False
    load_full_data_to_ram: bool = True
    max_ram_gb: Union[int, float] = 16
    world_size: int = 1
    num_workers: int = 0
    backend: str = "nccl"

    def __init__(self, **kwargs):
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        self.validate_optimizer_config(kwargs["optimizer"])
        self.optimizer = DotDict(kwargs["optimizer"])
        self.validate_scheduler_config(kwargs["scheduler"], kwargs)
        self.scheduler = DotDict(kwargs["scheduler"])

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


class ModelSpecModel(BaseModel):
    """Pydantic model for model specifications.

    Attributes:
        initial_embedding_dim: The size of the input embedding. Must be equal to dim_model if joint_embedding_dim is None.
        feature_embedding_dims: The embedding dimensions for each input column. Must sum to initial_embedding_dim.
        joint_embedding_dim: Joint embedding layer after initial embedding. Must be equal to dim_model if specified.
        n_head: The number of heads in the multi-head attention models.
        dim_feedforward: The dimension of the feedforward network model.
        num_layers: The number of layers in the transformer model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    initial_embedding_dim: int
    feature_embedding_dims: Optional[dict[str, int]] = None
    joint_embedding_dim: Optional[int] = None
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
        joint_embedding_dim = info.data.get("joint_embedding_dim")
        dim_model = v

        if joint_embedding_dim is None:
            if not v == initial_embedding_dim:
                raise ValueError(
                    f"If no joint_embedding_dim is configured, dim_model must be equal to initial_embedding_dim, {dim_model = } != {initial_embedding_dim = }"
                )
        else:
            if not v == joint_embedding_dim:
                raise ValueError(
                    f"If joint_embedding_dim is configured it must be equal to dim_model, {dim_model = } != {joint_embedding_dim = }"
                )

        return v

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
    """Pydantic model for training configuration.

    Attributes:
        project_root: The path to the sequifier project directory.
        metadata_config_path: The path to the data-driven configuration file.
        model_name: The name of the model being trained.
        training_data_path: The path to the training data.
        validation_data_path: The path to the validation data.
        read_format: The file format of the input data (e.g., 'csv', 'parquet').
        input_columns: The list of input columns to be used for training.
        column_types: A dictionary mapping each column to its numeric type ('int64' or 'float64').
        categorical_columns: A list of columns that are categorical.
        real_columns: A list of columns that are real-valued.
        target_columns: The list of target columns for model training.
        target_column_types: A dictionary mapping target columns to their types ('categorical' or 'real').
        id_maps: For each categorical column, a map from distinct values to their indexed representation.
        seq_length: The sequence length of the model's input.
        n_classes: The number of classes for each categorical column.
        inference_batch_size: The batch size to be used for inference after model export.
        seed: The random seed for numpy and PyTorch.
        export_generative_model: If True, exports the generative model.
        export_embedding_model: If True, exports the embedding model.
        export_onnx: If True, exports the model in ONNX format.
        export_pt: If True, exports the model using torch.save.
        export_with_dropout: If True, exports the model with dropout enabled.
        model_spec: The specification of the transformer model architecture.
        training_spec: The specification of the training run configuration.
    """

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

    seq_length: int
    n_classes: dict[str, int]
    inference_batch_size: int
    seed: int

    export_generative_model: bool
    export_embedding_model: bool
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    model_spec: ModelSpecModel
    training_spec: TrainingSpecModel

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
            if not (info.data.get("read_format") == "pt"):
                raise ValueError(
                    "If distributed is set to 'true', the format has to be 'pt'"
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
        # Original validation: consistent columns
        if not (
            info.data.get("input_columns") is None
            or (v.feature_embedding_dims is None)
            or np.all(
                np.array(list(v.feature_embedding_dims.keys()))
                == np.array(list(info.data.get("input_columns")))
            )
        ):
            raise ValueError(
                "If feature_embedding_dims is not None, dimensions must be specified for all input columns"
            )

        # Additional validation based on constraints in src/sequifier/train.py
        categorical_columns = info.data.get("categorical_columns", [])
        real_columns = info.data.get("real_columns", [])
        n_categorical = len(categorical_columns)
        n_real = len(real_columns)

        # Constraint 1: Mixed Data Types
        # If both real and categorical variables are present, feature_embedding_dims must be set.
        if n_categorical > 0 and n_real > 0:
            if v.feature_embedding_dims is None:
                raise ValueError(
                    "If both real and categorical variables are present, 'feature_embedding_dims' in 'model_spec' must be set explicitly."
                )

        # Constraint 2: Categorical Divisibility
        # If only categorical variables are included and auto-calculation is used,
        # max(dim_model, n_head) must be divisible by the number of categorical variables.
        if n_categorical > 0 and n_real == 0 and v.feature_embedding_dims is None:
            embedding_size = max(v.dim_model, v.n_head)
            if embedding_size % n_categorical != 0:
                raise ValueError(
                    f"If only categorical variables are included and feature_embedding_dims is not set, "
                    f"max(dim_model, n_head) ({embedding_size}) must be a multiple of the number of categorical variables ({n_categorical})."
                )

        return v
