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
        ddconfig_path = config_values.get("ddconfig_path")

        with open(
            normalize_path(ddconfig_path, config_values["project_path"]), "r"
        ) as f:
            dd_config = json.loads(f.read())

        config_values["column_types"] = config_values.get(
            "column_types", dd_config["column_types"]
        )

        if config_values["selected_columns"] is None:
            config_values["selected_columns"] = list(
                config_values["column_types"].keys()
            )

        config_values["categorical_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "int64" and col in config_values["selected_columns"]
        ]
        config_values["real_columns"] = [
            col
            for col, type_ in dd_config["column_types"].items()
            if type_ == "float64" and col in config_values["selected_columns"]
        ]
        config_values["n_classes"] = config_values.get(
            "n_classes", dd_config["n_classes"]
        )
        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", dd_config["split_paths"][0]),
            config_values["project_path"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                dd_config["split_paths"][min(1, len(dd_config["split_paths"]) - 1)],
            ),
            config_values["project_path"],
        )

        config_values["id_maps"] = dd_config["id_maps"]

    return TrainModel(**config_values)


class DotDict(dict):
    """Dot notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


class TrainingSpecModel(BaseModel):
    """Pydantic model for training specifications."""

    device: str
    epochs: int
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    iter_save: int
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
    continue_training: bool = True

    def __init__(self, **kwargs):
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        self.validate_optimizer_config(kwargs["optimizer"])
        self.optimizer = DotDict(kwargs["optimizer"])
        self.validate_scheduler_config(kwargs["scheduler"])
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
    def validate_scheduler_config(cls, v):
        if "name" not in v:
            raise ValueError("scheduler dict must specify 'name' field")
        if v["name"] not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler not valid as not found in {VALID_SCHEDULERS}")
        return v


class ModelSpecModel(BaseModel):
    """Pydantic model for model specifications."""

    d_model: int
    d_model_by_column: Optional[dict[str, int]]
    nhead: int
    d_hid: int
    nlayers: int

    @validator("d_model_by_column")
    def validate_d_model_by_column(cls, v, values):
        assert (
            v is None or np.sum(list(v.values())) == values["d_model"]
        ), f'{values["d_model"]} is not the sum of the d_model_by_column values'

        return v


class TrainModel(BaseModel):
    """Pydantic model for training configuration."""

    project_path: str
    ddconfig_path: str
    model_name: str
    training_data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    selected_columns: list[str]
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

    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    model_spec: ModelSpecModel
    training_spec: TrainingSpecModel

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
        ], "Currently only 'csv' and 'parquet' are supported"
        return v

    @validator("training_spec")
    def validate_training_spec(cls, v, values):
        assert set(values["target_columns"]) == set(
            v.criterion.keys()
        ), "target_columns and criterion must contain the same values/keys"
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
            values["selected_columns"] is None
            or (v.d_model_by_column is None)
            or np.all(
                np.array(list(v.d_model_by_column.keys()))
                == np.array(list(values["selected_columns"]))
            )
        )
        return v
