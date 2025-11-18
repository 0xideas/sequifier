import json
from itertools import product
from typing import Optional

import numpy as np
import yaml
from beartype import beartype
from pydantic import BaseModel, Field, field_validator

from sequifier.config.train_config import (
    DotDict,
    ModelSpecModel,
    TrainingSpecModel,
    TrainModel,
)
from sequifier.helpers import normalize_path, try_catch_excess_keys


@beartype
def load_hyperparameter_search_config(
    config_path: str, skip_metadata: bool
) -> "HyperparameterSearch":
    """Load a hyperparameter search configuration from a YAML file.

    This function reads a YAML configuration file, processes it to include
    data-driven configurations if needed, and returns a HyperparameterSearch
    object.

    Args:
        config_path: The path to the hyperparameter search configuration file.
        skip_metadata: A boolean flag indicating whether the configuration is
            for unprocessed data. If False, it will load and integrate
            data-driven configurations.

    Returns:
        An instance of the HyperparameterSearch class, populated with the
        configuration from the file.
    """
    with open(config_path, "r") as f:
        config_values = yaml.safe_load(f)

    if not skip_metadata:
        metadata_config_path = config_values.get("metadata_config_path")

        with open(
            normalize_path(metadata_config_path, config_values["project_path"]), "r"
        ) as f:
            metadata_config = json.loads(f.read())

        config_values["column_types"] = config_values.get(
            "column_types", [metadata_config["column_types"]]
        )

        if config_values["input_columns"] is None:
            config_values["input_columns"] = [
                list(config_values["column_types"].keys())
            ]

        config_values["categorical_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_types"].items()
                if type_ == "Int64" and col in input_columns
            ]
            for input_columns in config_values["input_columns"]
        ]

        config_values["real_columns"] = [
            [
                col
                for col, type_ in metadata_config["column_types"].items()
                if type_ == "Float64" and col in input_columns
            ]
            for input_columns in config_values["input_columns"]
        ]

        config_values["n_classes"] = config_values.get(
            "n_classes", metadata_config["n_classes"]
        )
        config_values["training_data_path"] = normalize_path(
            config_values.get("training_data_path", metadata_config["split_paths"][0]),
            config_values["project_path"],
        )
        config_values["validation_data_path"] = normalize_path(
            config_values.get(
                "validation_data_path",
                metadata_config["split_paths"][
                    min(1, len(metadata_config["split_paths"]) - 1)
                ],
            ),
            config_values["project_path"],
        )

        config_values["id_maps"] = metadata_config["id_maps"]

    return try_catch_excess_keys(config_path, HyperparameterSearch, config_values)


class TrainingSpecHyperparameterSampling(BaseModel):
    """Pydantic model for training specification hyperparameter sampling.

    Attributes:
        device: The device to train on (e.g., 'cuda', 'cpu').
        epochs: A list of possible numbers of epochs to train for.
        log_interval: The interval in batches for logging.
        class_share_log_columns: Columns for which to log class share.
        early_stopping_epochs: Number of epochs for early stopping.
        save_interval_epochs: Interval in epochs for saving model checkpoints.
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
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    device: str
    epochs: list[int]
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    save_interval_epochs: int
    batch_size: list[int]
    learning_rate: list[float]
    criterion: dict[str, str]
    class_weights: Optional[dict[str, list[float]]] = None
    accumulation_steps: list[int]
    dropout: list[float] = [0.0]
    loss_weights: Optional[dict[str, float]] = None
    optimizer: list[DotDict] = Field(
        default_factory=lambda: [DotDict({"name": "Adam"})]
    )
    scheduler: list[DotDict] = Field(
        default_factory=lambda: [
            DotDict({"name": "StepLR", "step_size": 1, "gamma": 0.99})
        ]
    )
    continue_training: bool = True

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
        self.scheduler = [
            DotDict(scheduler_config) for scheduler_config in kwargs["scheduler"]
        ]

    @field_validator("scheduler")
    @classmethod
    def validate_model_spec(cls, v, values):
        assert (
            len(values["learning_rate"]) == len(v)
        ), "learning_rate and scheduler must have the same number of candidate values, that are paired"

        assert (
            len(values["epochs"]) == len(v)
        ), "epochs and scheduler must have the same number of candidate values, that are paired"
        return v

    def random_sample(self):
        """Randomly sample a set of training hyperparameters.

        This method selects a random combination of hyperparameters from the
        defined lists of possibilities. It ensures that learning rates and
        schedulers are paired correctly.

        Returns:
            A TrainingSpecModel instance populated with a randomly sampled set of
            hyperparameters.
        """
        learning_rate_and_scheduler_index = np.random.randint(len(self.learning_rate))
        optimizer_index = np.random.randint(len(self.optimizer))
        batch_size = np.random.choice(self.batch_size)
        dropout = np.random.choice(self.dropout)
        accumulation_steps = np.random.choice(self.accumulation_steps)
        optimizer = self.optimizer[optimizer_index]
        learning_rate = self.learning_rate[learning_rate_and_scheduler_index]

        print(f"{learning_rate = } - {batch_size = } - {dropout = } - {optimizer = }")

        return TrainingSpecModel(
            device=self.device,
            epochs=self.epochs[learning_rate_and_scheduler_index],
            log_interval=self.log_interval,
            class_share_log_columns=self.class_share_log_columns,
            early_stopping_epochs=self.early_stopping_epochs,
            save_interval_epochs=self.save_interval_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            criterion=self.criterion,
            class_weights=self.class_weights,
            accumulation_steps=accumulation_steps,
            dropout=dropout,
            loss_weights=self.loss_weights,
            optimizer=optimizer,
            scheduler=self.scheduler[learning_rate_and_scheduler_index],
        )

    def grid_sample(self, i):
        """Select a set of training hyperparameters based on a grid search index.

        This method generates a grid of all possible hyperparameter combinations
        and selects the combination at the given index.

        Args:
            i: The index of the hyperparameter combination to select from the grid.

        Returns:
            A TrainingSpecModel instance populated with the selected set of
            hyperparameters.
        """
        hyperparameter_combinations = list(
            product(
                np.arange(len(self.learning_rate)),
                self.batch_size,
                self.dropout,
                self.optimizer,
                self.accumulation_steps,
            )
        )
        (
            learning_rate_and_scheduler_index,
            batch_size,
            dropout,
            optimizer,
            accumulation_steps,
        ) = hyperparameter_combinations[i]

        learning_rate = self.learning_rate[learning_rate_and_scheduler_index]

        print(f"{learning_rate = } - {batch_size = } - {dropout = } - {optimizer = }")

        return TrainingSpecModel(
            device=self.device,
            epochs=self.epochs[learning_rate_and_scheduler_index],
            log_interval=self.log_interval,
            class_share_log_columns=self.class_share_log_columns,
            early_stopping_epochs=self.early_stopping_epochs,
            save_interval_epochs=self.save_interval_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            criterion=self.criterion,
            class_weights=self.class_weights,
            accumulation_steps=accumulation_steps,
            dropout=dropout,
            loss_weights=self.loss_weights,
            optimizer=optimizer,
            scheduler=self.scheduler[learning_rate_and_scheduler_index],
        )

    def n_combinations(self):
        """Calculate the total number of hyperparameter combinations.

        This method computes the total number of unique hyperparameter sets that
        can be generated by the grid search.

        Returns:
            The total number of possible hyperparameter combinations.
        """
        return (
            len(self.learning_rate)
            * len(self.batch_size)
            * len(self.dropout)
            * len(self.optimizer)
            * len(self.accumulation_steps)
        )


class ModelSpecHyperparameterSampling(BaseModel):
    """Pydantic model for model specification hyperparameter sampling.

    Attributes:
        dim_model: A list of possible numbers of expected features in the input.
        dim_model_by_column: A list of possible embedding dimensions for each input column.
        n_head: A list of possible numbers of heads in the multi-head attention models.
        dim_feedforward: A list of possible dimensions of the feedforward network model.
        num_layers: A list of possible numbers of layers in the transformer model.
    """

    dim_model: list[int]
    dim_model_by_column: Optional[list[dict[str, int]]]
    n_head: list[int]
    dim_feedforward: list[int]
    num_layers: list[int]

    @field_validator("n_head")
    @classmethod
    def validate_model_spec(cls, v, values):
        if values["dim_model_by_column"] is not None:
            assert (
                len(values["dim_model"]) == len(values["dim_model_by_column"])
            ), "dim_model and dim_model_by_column must have the same number of candidate values, that are paired"

        assert (
            len(values["dim_model"]) == len(v)
        ), "dim_model and n_head must have the same number of candidate values, that are paired"
        return v

    def random_sample(self):
        """Randomly sample a set of model hyperparameters.

        This method selects a random combination of model hyperparameters from the
        defined lists of possibilities. It ensures that dim_model, dim_model_by_column,
        and n_head are paired correctly.

        Returns:
            A ModelSpecModel instance populated with a randomly sampled set of
            hyperparameters.
        """
        dim_model_index = np.random.randint(len(self.dim_model))
        dim_model_by_column = (
            None
            if self.dim_model_by_column is None
            else self.dim_model_by_column[dim_model_index]
        )
        dim_model = self.dim_model[dim_model_index]
        dim_feedforward = np.random.choice(self.dim_feedforward)
        num_layers = np.random.choice(self.num_layers)
        print(f"{dim_model = } - {dim_feedforward = } - {num_layers = }")

        return ModelSpecModel(
            dim_model=self.dim_model[dim_model_index],
            dim_model_by_column=dim_model_by_column,
            n_head=self.n_head[dim_model_index],
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        )

    def grid_sample(self, i):
        """Select a set of model hyperparameters based on a grid search index.

        This method generates a grid of all possible model hyperparameter
        combinations and selects the combination at the given index.

        Args:
            i: The index of the hyperparameter combination to select from the grid.

        Returns:
            A ModelSpecModel instance populated with the selected set of
            hyperparameters.
        """
        hyperparameter_combinations = list(
            product(
                np.arange(len(self.dim_model)), self.dim_feedforward, self.num_layers
            )
        )

        dim_model_index, dim_feedforward, num_layers = hyperparameter_combinations[i]
        dim_model = self.dim_model[dim_model_index]
        print(f"{dim_model = } - {dim_feedforward = } - {num_layers = }")

        dim_model_by_column = (
            None
            if self.dim_model_by_column is None
            else self.dim_model_by_column[dim_model_index]
        )

        return ModelSpecModel(
            dim_model=dim_model,
            dim_model_by_column=dim_model_by_column,
            n_head=self.n_head[dim_model_index],
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        )

    def n_combinations(self):
        """Calculate the total number of model hyperparameter combinations.

        This method computes the total number of unique model hyperparameter sets
        that can be generated by the grid search.

        Returns:
            The total number of possible model hyperparameter combinations.
        """
        return len(self.dim_model) * len(self.dim_feedforward) * len(self.num_layers)


class HyperparameterSearch(BaseModel):
    """Pydantic model for hyperparameter search configuration.

    Attributes:
        project_path: The path to the sequifier project directory.
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
    """

    project_path: str
    metadata_config_path: str
    hp_search_name: str
    search_strategy: str = "sample"  # "sample" or "grid"
    n_samples: Optional[int]
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

    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    model_hyperparameter_sampling: ModelSpecHyperparameterSampling
    training_hyperparameter_sampling: TrainingSpecHyperparameterSampling

    @field_validator("column_types")
    @classmethod
    def validate_model_spec(cls, v, values):
        if v is not None:
            assert (
                len(values["input_columns"]) == len(v)
            ), "input_columns and column_types must have the same number of candidate values, that are paired"
        return v

    def random_sample(self, i):
        """Randomly sample a full training configuration.

        This method generates a complete training configuration by randomly
        sampling model and training hyperparameters, as well as selecting a
        column set and sequence length.

        Args:
            i: The index of the sample, used to create a unique model name.

        Returns:
            A TrainModel instance populated with a randomly sampled configuration.
        """
        model_spec = self.model_hyperparameter_sampling.random_sample()
        training_spec = self.training_hyperparameter_sampling.random_sample()
        input_columns_index = np.random.randint(len(self.input_columns))
        seq_length = np.random.choice(self.seq_length)
        print(f"{input_columns_index = } - {seq_length = }")
        return TrainModel(
            project_path=self.project_path,
            metadata_config_path=self.metadata_config_path,
            model_name=self.hp_search_name + f"-run-{i}",
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
            export_onnx=self.export_onnx,
            export_pt=self.export_pt,
            export_with_dropout=self.export_with_dropout,
            model_spec=model_spec,
            training_spec=training_spec,
        )

    def grid_sample(self, i):
        """Select a full training configuration based on a grid search index.

        This method generates a grid of all possible configurations and selects
        the configuration at the given index.

        Args:
            i: The index of the configuration to select from the grid.

        Returns:
            A TrainModel instance populated with the selected configuration.
        """
        model_hyperparamter_sample = self.model_hyperparameter_sampling.n_combinations()
        training_hyperparamter_sample = (
            self.training_hyperparameter_sampling.n_combinations()
        )
        inner_combinations = model_hyperparamter_sample * training_hyperparamter_sample

        i_model = i % model_hyperparamter_sample
        i_training = (i // model_hyperparamter_sample) % training_hyperparamter_sample
        i_outer = i // inner_combinations

        model_spec = self.model_hyperparameter_sampling.grid_sample(i_model)
        training_spec = self.training_hyperparameter_sampling.grid_sample(i_training)

        hyperparameter_combinations = list(
            product(np.arange(len(self.input_columns)), self.seq_length)
        )

        input_columns_index, seq_length = hyperparameter_combinations[i_outer]

        return TrainModel(
            project_path=self.project_path,
            metadata_config_path=self.metadata_config_path,
            model_name=self.hp_search_name + f"-run-{i}",
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
            export_onnx=self.export_onnx,
            export_pt=self.export_pt,
            export_with_dropout=self.export_with_dropout,
            model_spec=model_spec,
            training_spec=training_spec,
        )

    def sample(self, i):
        """Sample a configuration based on the specified search strategy.

        This method delegates to either random_sample or grid_sample based on
        the `search_strategy` attribute.

        Args:
            i: The index of the sample or grid combination to generate.

        Returns:
            A TrainModel instance with a generated configuration.

        Raises:
            Exception: If the search_strategy is not 'sample' or 'grid'.
        """
        if self.search_strategy == "sample":
            return self.random_sample(i)
        elif self.search_strategy == "grid":
            return self.grid_sample(i)
        else:
            raise Exception(f"{self.search_strategy} invalid")

    def n_combinations(self):
        """Calculate the total number of possible configurations.

        This method computes the total number of unique configurations that can be
        generated by a grid search over all defined hyperparameters.

        Returns:
            The total number of possible hyperparameter configurations.
        """
        return (
            len(self.input_columns)
            * len(self.seq_length)
            * self.model_hyperparameter_sampling.n_combinations()
            * self.training_hyperparameter_sampling.n_combinations()
        )
