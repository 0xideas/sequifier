import json
import os
import warnings
from typing import Any, Optional, Union

import yaml
from beartype import beartype
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sequifier.config.train_config import (
    DotDict,
    ModelSpecModel,
    TrainingSpecModel,
    TrainModel,
)
from sequifier.helpers import normalize_path, try_catch_excess_keys


class FloatDistribution(BaseModel):
    """Pydantic model representing a floating-point hyperparameter distribution for Optuna.

    Attributes:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        log (bool): If True, sample from the distribution in the log domain. Defaults to False.
    """

    low: float
    high: float
    log: bool = False


class IntDistribution(BaseModel):
    """Pydantic model representing an integer hyperparameter distribution for Optuna.

    Attributes:
        low (int): The lower bound of the distribution.
        high (int): The upper bound of the distribution.
        step (int): The spacing between valid integer values. Defaults to 1.
        log (bool): If True, sample from the distribution in the log domain. Defaults to False.
    """

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


@beartype
def load_hyperparameter_search_config(
    config_path: str, skip_metadata: bool
) -> "HyperparameterSearchConfig":
    """Load a hyperparameter search configuration from a YAML file.

    This function reads a YAML configuration file, processes it to include
    data-driven configurations if needed, and returns a HyperparameterSearchConfig
    object.

    Args:
        config_path: The path to the hyperparameter search configuration file.
        skip_metadata: A boolean flag indicating whether the configuration is
            for unprocessed data. If False, it will load and integrate
            data-driven configurations.

    Returns:
        An instance of the HyperparameterSearchConfig class, populated with the
        configuration from the file.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_root"]), "r"
        ) as f:
            metadata_config = json.loads(f.read())

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
    """Pydantic model for training specification hyperparameter sampling.

    Attributes:
        device: The device to train on (e.g., 'cuda', 'cpu').
        epochs: A list of possible numbers of epochs to train for.
        log_interval: The interval in batches for logging.
        class_share_log_columns: Columns for which to log class share.
        early_stopping_epochs: Number of epochs for early stopping.
        save_interval_epochs: Interval in epochs for saving model checkpoints.
        save_latest_interval_minutes: the time interval in which a checkpoint is written to the "latest" checkpoint path
        save_batch_interval_minutes: the time interval in which a checkpoint is written to a unique checkpoint path
        save_batch_interval_minutes_val_loss: calculate val loss at the moment of batch interval saving
        calculate_validation_loss_on_initialization: calculate val loss on weight initialization
        batch_size: A list of possible batch sizes.
        learning_rate: A list of possible learning rates.
        criterion: A dictionary mapping target columns to loss functions.
        class_weights: Optional dictionary mapping columns to class weights.
        accumulation_steps: A list of possible gradient accumulation steps.
        dropout: A list of possible dropout rates.
        loss_weights: Optional dictionary mapping columns to loss weights.
        optimizer: A list of possible optimizer configurations.
        scheduler: A list of possible scheduler configurations.
        continue_training: Flag to continue training from a checkpoint.
        layer_type_dtypes: Dictionary mapping layer types (linear, embedding, norm) to dtypes (bfloat16, float8_e4m3fn).
        layer_autocast: Whether to use autocast
        sampling_strategy: data sampling in distributed training: 'exact', 'oversampling' or 'undersampling'
        data_parallelism: 'DDP' or 'FSDP'
        fsdp_cpu_offload: fsdp cpu offload
        torch_compile: compile entire model ('outer') or transformer layers ('inner') with torch.compile, alternatively 'none'
        float32_matmul_precision: precision level of float32 computations. One of 'highest', 'high' and 'medium'

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device: str
    epochs: list[int]
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
    save_latest_interval_minutes: Optional[float] = None
    save_batch_interval_minutes: Optional[float] = None
    save_batch_interval_minutes_val_loss: bool = True
    calculate_validation_loss_on_initialization: bool = False

    batch_size: OptunaInt
    learning_rate: list[float]  # Kept as list to preserve coupling with epochs
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
    sampling_strategy: str = "exact"
    data_parallelism: Optional[str] = None
    fsdp_cpu_offload: Optional[bool] = None
    torch_compile: str = "outer"
    float32_matmul_precision: str = "highest"

    def __init__(self, **kwargs):
        """Initialize the TrainingSpecHyperparameterSampling instance.

        This method initializes the Pydantic BaseModel and then processes the
        optimizer and scheduler configurations from the provided keyword
        arguments, converting them into DotDict objects.

        Args:
            **kwargs: Keyword arguments that correspond to the attributes of this
                class. The 'optimizer' and 'scheduler' arguments are expected
                to be lists of dictionaries.
        """
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
        """Samples training hyperparameters using an Optuna trial.

        This method leverages the provided Optuna trial to suggest values for
        hyperparameters like batch size, dropout, and learning rate based on the
        defined search spaces (categorical lists or distributions).

        Args:
            trial (Any): The Optuna trial object used for suggesting hyperparameters.

        Returns:
            TrainingSpecModel: A populated training specification model with the sampled hyperparameters.
        """
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

        def sample_param(
            name: str, space: Union[list, FloatDistribution, IntDistribution]
        ):
            if isinstance(space, list):
                return trial.suggest_categorical(name, space)
            elif isinstance(space, FloatDistribution):
                return trial.suggest_float(name, space.low, space.high, log=space.log)
            elif isinstance(space, IntDistribution):
                return trial.suggest_int(
                    name, space.low, space.high, step=space.step, log=space.log
                )

        batch_size = sample_param("batch_size", self.batch_size)
        dropout = sample_param("dropout", self.dropout)
        accumulation_steps = sample_param("accumulation_steps", self.accumulation_steps)

        logger.info(
            f"{learning_rate = } - {batch_size = } - {dropout = } - {optimizer = }"
        )

        return TrainingSpecModel(
            device=self.device,
            epochs=epochs,
            log_interval=self.log_interval,
            class_share_log_columns=self.class_share_log_columns,
            early_stopping_epochs=self.early_stopping_epochs,
            save_interval_epochs=self.save_interval_epochs,
            save_latest_interval_minutes=self.save_latest_interval_minutes,
            save_batch_interval_minutes=self.save_batch_interval_minutes,
            save_batch_interval_minutes_val_loss=self.save_batch_interval_minutes_val_loss,
            calculate_validation_loss_on_initialization=self.calculate_validation_loss_on_initialization,
            batch_size=batch_size,
            learning_rate=learning_rate,
            criterion=self.criterion,
            class_weights=self.class_weights,
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
            sampling_strategy=self.sampling_strategy,
            data_parallelism=self.data_parallelism,
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            torch_compile=self.torch_compile,
            float32_matmul_precision=self.float32_matmul_precision,
        )


class ModelSpecHyperparameterSampling(BaseModel):
    """Pydantic model for model specification hyperparameter sampling.

    Attributes:
        initial_embedding_dim: A list of possible sizes for the initial input embedding.
        feature_embedding_dims: A list of possible dictionaries defining embedding dimensions for each input column.
        joint_embedding_dim: A list of possible sizes for the joint embedding layer projection.
        dim_model: A list of possible numbers of expected features in the input (d_model).
        n_head: A list of possible numbers of heads in the multi-head attention models.
        dim_feedforward: A list of possible dimensions of the feedforward network model.
        num_layers: A list of possible numbers of layers in the transformer model.
    """

    initial_embedding_dim: list[int]
    joint_embedding_dim: list[Optional[int]]
    dim_model: list[int]
    feature_embedding_dims: Optional[list[dict[str, int]]]
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
        dim_model_len = len(info.data.get("dim_model", []))

        if info.data.get("feature_embedding_dims") is not None:
            if not (
                len(info.data.get("dim_model"))
                == len(info.data.get("feature_embedding_dims"))
            ):
                raise ValueError(
                    "dim_model and feature_embedding_dims must have the same number of candidate values, that are paired"
                )

        if not (len(info.data.get("dim_model")) == len(v)):
            raise ValueError(
                "dim_model and n_head must have the same number of candidate values, that are paired"
            )

        if "initial_embedding_dim" in info.data:
            if len(info.data["initial_embedding_dim"]) != dim_model_len:
                raise ValueError(
                    "initial_embedding_dim must have the same number of values as dim_model"
                )

        if "joint_embedding_dim" in info.data:
            if len(info.data["joint_embedding_dim"]) != dim_model_len:
                raise ValueError(
                    "joint_embedding_dim must have the same number of values as dim_model"
                )

        return v

    def sample_trial(self, trial: Any) -> ModelSpecModel:
        """Samples model architecture hyperparameters using an Optuna trial.

        This method uses the Optuna trial to suggest structural parameters such as
        the number of layers, feedforward dimensions, and attention heads. It ensures
        that dependent dimensions (like `n_head` and `dim_model`) stay correctly paired
        and that invalid key-value head combinations are filtered out.

        Args:
            trial (Any): The Optuna trial object used for suggesting hyperparameters.

        Returns:
            ModelSpecModel: A populated model specification model with the sampled architecture parameters.
        """
        dim_model_idx = trial.suggest_categorical(
            "dim_model_idx", list(range(len(self.dim_model)))
        )

        initial_embedding_dim = self.initial_embedding_dim[dim_model_idx]
        joint_embedding_dim = self.joint_embedding_dim[dim_model_idx]
        dim_model = self.dim_model[dim_model_idx]
        n_head = self.n_head[dim_model_idx]
        feature_embedding_dims = (
            None
            if self.feature_embedding_dims is None
            else self.feature_embedding_dims[dim_model_idx]
        )

        def sample_param(
            name: str, space: Union[list, FloatDistribution, IntDistribution]
        ):
            if isinstance(space, list):
                return trial.suggest_categorical(name, space)
            elif isinstance(space, FloatDistribution):
                return trial.suggest_float(name, space.low, space.high, log=space.log)
            elif isinstance(space, IntDistribution):
                return trial.suggest_int(
                    name, space.low, space.high, step=space.step, log=space.log
                )

        dim_feedforward = sample_param("dim_feedforward", self.dim_feedforward)
        num_layers = sample_param("num_layers", self.num_layers)
        rope_theta = sample_param("rope_theta", self.rope_theta)

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
            f"{initial_embedding_dim} - {joint_embedding_dim = } - {dim_model = } - {dim_feedforward = } - {num_layers = } - {activation_fn = } - {normalization = } - {positional_encoding = } - {attention_type = } - {norm_first = } - {n_kv_heads = } - {rope_theta = } "
        )

        return ModelSpecModel(
            initial_embedding_dim=initial_embedding_dim,
            feature_embedding_dims=feature_embedding_dims,
            joint_embedding_dim=joint_embedding_dim,
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            activation_fn=activation_fn,
            normalization=normalization,
            positional_encoding=positional_encoding,
            attention_type=attention_type,
            norm_first=norm_first,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
            prediction_length=self.prediction_length,
        )


class HyperparameterSearchConfig(BaseModel):
    """Pydantic model for hyperparameter search configuration.

    Attributes:
        project_root: The path to the sequifier project directory.
        metadata_config_path: The path to the data-driven configuration file.
        hp_search_name: The name for the hyperparameter search.
        search_strategy: The search strategy, either "sample" or "grid".
        n_samples: The number of samples to draw for the search.
        model_config_write_path: The path to write the model configurations to.
        training_data_path: The path to the training data.
        validation_data_path: The path to the validation data.
        read_format: The file format of the input data.
        input_columns: A list of lists of columns to be used for training.
        column_types: A list of dictionaries mapping columns to their types.
        categorical_columns: A list of lists of categorical columns.
        real_columns: A list of lists of real-valued columns.
        target_columns: The list of target columns for model training.
        target_column_types: A dictionary mapping target columns to their types.
        id_maps: A dictionary mapping categorical values to their indexed representation.
        seq_length: A list of possible sequence lengths.
        n_classes: The number of classes for each categorical column.
        inference_batch_size: The batch size for inference.
        export_onnx: If True, exports the model in ONNX format.
        export_pt: If True, exports the model using torch.save.
        export_with_dropout: If True, exports the model with dropout enabled.
        model_hyperparameter_sampling: The sampling configuration for model hyperparameters.
        training_hyperparameter_sampling: The sampling configuration for training hyperparameters.
        evaluation_inference_config: The inference config to infer on for hyperparameter search optimization
        evaluation_script: The script that outputs the evaluation metrics, typically from the inference output
        evaluation_metrics: The evaluation metrics to optimize during hyperparameter search
        evaluation_metric_directions: The direction to optimize evaluation_metrics in. Only 'minimize' and 'maximize' are allowed
    """

    project_root: str
    metadata_config_path: str
    hp_search_name: str
    search_strategy: str = "bayesian"
    n_trials: Optional[int] = Field(None, alias="n_samples")
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

    seq_length: list[int]
    n_classes: dict[str, int]
    inference_batch_size: int

    export_generative_model: bool
    export_embedding_model: bool
    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    evaluation_inference_config: Optional[str] = None
    evaluation_script: Optional[str] = None
    evaluation_metric_directions: Optional[list[str]] = None
    evaluation_metrics: Optional[list[str]] = None

    model_hyperparameter_sampling: ModelSpecHyperparameterSampling
    training_hyperparameter_sampling: TrainingSpecHyperparameterSampling

    override_input: bool = False

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
        """Generates a complete training configuration using an Optuna trial.

        This method orchestrates the sampling of both model and training specifications,
        as well as data sequence parameters, combining them into a final configuration
        ready for model execution.

        Args:
            trial (Any): The Optuna trial object used for suggesting hyperparameters.
            run_index (int): The current run/trial index, used to assign a unique name to the model.

        Returns:
            TrainModel: A fully populated configuration instance for the current trial.
        """
        model_spec = self.model_hyperparameter_sampling.sample_trial(trial)
        training_spec = self.training_hyperparameter_sampling.sample_trial(trial)

        input_columns_index = trial.suggest_categorical(
            "input_columns_index", list(range(len(self.input_columns)))
        )
        seq_length = trial.suggest_categorical("seq_length", self.seq_length)

        logger.info(f"{input_columns_index = } - {seq_length = }")

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
            seq_length=seq_length,
            n_classes=self.n_classes,
            inference_batch_size=self.inference_batch_size,
            seed=101,
            export_embedding_model=self.export_embedding_model,
            export_generative_model=self.export_generative_model,
            export_onnx=self.export_onnx,
            export_pt=self.export_pt,
            export_with_dropout=self.export_with_dropout,
            model_spec=model_spec,
            training_spec=training_spec,
        )
