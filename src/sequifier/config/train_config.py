import copy
import json
from typing import Any, Optional

import numpy as np
import yaml
from beartype import beartype
from pydantic import BaseModel, Field, validator

from sequifier.helpers import normalize_path

AnyType = str | int | float

VALID_LOSS_FUNCTIONS = [
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

VALID_OPTIMIZERS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "AdEMAMix",
    "A2GradUni",
    "A2GradInc",
    "A2GradExp",
    "AccSGD",
    "AdaBelief",
    "AdaBound",
    "Adafactor",
    "Adahessian",
    "AdaMod",
    "AdamP",
    "AggMo",
    "Apollo",
    "DiffGrad",
    "Lamb",
    "LARS",
    "Lion",
    "Lookahead",
    "MADGRAD",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDP",
    "SGDW",
    "Shampoo",
    "SWATS",
    "Yogi",
]

VALID_SCHEDULERS = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "PolynomialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "SequentialLR",
    "ReduceLROnPlateau",
    "CyclicLR",
    "OneCycleLR",
    "CosineAnnealingWarmRestarts",
]


@beartype
def load_train_config(
    config_path: str, args_config: dict[str, Any], on_unprocessed: bool
) -> "TrainModel":
    """
    Load training configuration from a YAML file and update it with args_config.

    Args:
        config_path: Path to the YAML configuration file.
        args_config: Dictionary containing additional configuration arguments.
        on_unprocessed: Flag indicating whether to process the configuration or not.

    Returns:
        TrainModel instance with loaded configuration.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    config_values.update(args_config)

    if not on_unprocessed:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_path"]), "r"
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
        assert (
            len(config_values["real_columns"] + config_values["categorical_columns"])
            > 0
        )
        config_values["n_classes"] = config_values.get(
            "n_classes", metadata_config["n_classes"]
        )
        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", split_paths[0]),
            config_values["project_path"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                split_paths[min(1, len(split_paths) - 1)],
            ),
            config_values["project_path"],
        )

        config_values["id_maps"] = metadata_config["id_maps"]

    return TrainModel(**config_values)


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
        checkpoint_frequency: The interval in epochs for checkpointing the model.
        batch_size: The training batch size.
        lr: The learning rate.
        criterion: A dictionary mapping each target column to a loss function.
        class_weights: A dictionary mapping categorical target columns to a list of class weights.
        accumulation_steps: The number of gradient accumulation steps.
        dropout: The dropout value for the transformer model.
        loss_weights: A dictionary mapping columns to specific loss weights.
        optimizer: The optimizer configuration.
        scheduler: The learning rate scheduler configuration.
        scheduler_step_iter: The time of the .step() call on the scheduler, either 'epoch' or 'batch'
        continue_training: If True, continue training from the latest checkpoint.
        distributed: If True, enables distributed training.
        load_full_data_to_ram: If True, loads the entire dataset into RAM.
        world_size: The number of processes for distributed training.
        num_workers: The number of worker threads for data loading.
        backend: The distributed training backend (e.g., 'nccl').
    """

    device: str
    device_max_concat_length: int = 12
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    checkpoint_frequency: int
    batch_size: int
    lr: float
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
    scheduler_step_iter: str = "epoch"
    continue_training: bool = True
    distributed: bool = False
    load_full_data_to_ram: bool = True
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

    @validator("criterion")
    def validate_criterion(cls, v):
        for vv in v.values():
            if vv not in VALID_LOSS_FUNCTIONS:
                raise ValueError(
                    f"criterion must be in {VALID_LOSS_FUNCTIONS}, {vv} isn't"
                )
        return v

    @validator("optimizer")
    def validate_optimizer_config(cls, v):
        if "name" not in v:
            raise ValueError("optimizer dict must specify 'name' field")
        if v["name"] not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer not valid as not found in {VALID_OPTIMIZERS}")
        return v

    @validator("scheduler")
    def validate_scheduler_config(cls, v, values):
        if "name" not in v:
            raise ValueError("scheduler dict must specify 'name' field")
        if v["name"] not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler not valid as not found in {VALID_SCHEDULERS}")
        if "total_steps" in v:
            if values["scheduler_step_iter"] == "epoch":
                if not v["total_steps"] == values["epochs"]:
                    raise ValueError(
                        f"scheduler total steps: {v['total_steps']} != {values['epochs']}: total epochs"
                    )
            else:
                print(
                    f"[WARNING] {v['total_steps']} scheduler steps at {values['epochs']} epochs implies {v['total_steps']/values['epochs']:.2f} batches. Does this seem correct?"
                )
        return v

    @validator("scheduler_step_iter")
    def validate_scheduler_step_iter(cls, v):
        if v not in ["epoch", "batch"]:
            raise ValueError(
                f"scheduler_step_iter must be in ['epoch', 'batch'], {v} isn't"
            )
        return v


class ModelSpecModel(BaseModel):
    """Pydantic model for model specifications.

    Attributes:
        d_model: The number of expected features in the input.
        d_model_by_column: The embedding dimensions for each input column. Must sum to d_model.
        nhead: The number of heads in the multi-head attention models.
        d_hid: The dimension of the feedforward network model.
        nlayers: The number of layers in the transformer model.
    """

    d_model: int
    d_model_by_column: Optional[dict[str, int]]
    nhead: int
    d_hid: int
    nlayers: int
    inference_size: int

    @validator("d_model_by_column")
    def validate_d_model_by_column(cls, v, values):
        assert (
            v is None or np.sum(list(v.values())) == values["d_model"]
        ), f'{values["d_model"]} is not the sum of the d_model_by_column values'

        return v


class TrainModel(BaseModel):
    """Pydantic model for training configuration.

    Attributes:
        project_path: The path to the sequifier project directory.
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

    project_path: str
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

    @validator("model_name")
    def validate_model_name(cls, v):
        assert "embedding" not in v, "model_name cannot contain 'embedding'"
        return v

    @validator("target_column_types")
    def validate_target_column_types(cls, v, values):
        assert all(vv in ["categorical", "real"] for vv in v.values())
        assert (
            list(v.keys()) == values["target_columns"]
        ), "target_columns and target_column_types must contain the same values/keys in the same order"
        return v

    @validator("read_format")
    def validate_read_format(cls, v):
        assert v in [
            "csv",
            "parquet",
            "pt",
        ], "Currently only 'csv', 'parquet' and 'pt' are supported"
        return v

    @validator("training_spec")
    def validate_training_spec(cls, v, values):
        assert set(values["target_columns"]) == set(
            v.criterion.keys()
        ), "target_columns and criterion must contain the same values/keys"

        if v.distributed:
            assert (
                values["read_format"] == "pt"
            ), "If distributed is set to 'true', the format has to be 'pt'"
        return v

    @validator("column_types")
    def validate_column_types(cls, v, values):
        target_columns = values.get("target_columns", [])
        column_ordered = list(v.keys())
        columns_ordered_filtered = [c for c in column_ordered if c in target_columns]
        assert (
            columns_ordered_filtered == target_columns
        ), f"{columns_ordered_filtered = } != {target_columns = }"
        return v

    @validator("model_spec")
    def validate_model_spec(cls, v, values):
        assert (
            values["input_columns"] is None
            or (v.d_model_by_column is None)
            or np.all(
                np.array(list(v.d_model_by_column.keys()))
                == np.array(list(values["input_columns"]))
            )
        )
        return v
