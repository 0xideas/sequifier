from itertools import product
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from train_config import DotDict, ModelSpecModel


class TrainingSpecHyperparameterSampling(BaseModel):
    """Pydantic model for training specifications."""

    device: str
    epochs: list[int]
    log_interval: int = 10
    class_share_log_columns: list[str] = Field(default_factory=list)
    early_stopping_epochs: Optional[int] = None
    iter_save: int
    batch_size: list[int]
    lr: list[float]
    criterion: dict[str, str]
    class_weights: Optional[list[dict[str, list[float]]]] = None
    accumulation_steps: Optional[int] = None
    dropout: list[float] = [0.0]
    loss_weights: Optional[list[dict[str, float]]] = None
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
        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in ["optimizer", "scheduler"]}
        )

        self.optimizer = [DotDict(kwargs["optimizer"])]
        self.scheduler = [DotDict(kwargs["scheduler"])]


class ModelSpecHyperparameterSampling(BaseModel):
    """Pydantic model for model specifications."""

    d_model: list[int]
    d_model_by_column: Optional[list[dict[str, int]]]
    nhead: list[int]
    d_hid: list[int]
    nlayers: list[int]

    hyperparamter_combinations = None

    def random_sample(self):
        d_model_index = np.random.choice(np.arange(len(self.d_model)))
        d_model_by_column = (
            None
            if self.d_model_by_column is None
            else self.d_model_by_column[d_model_index]
        )
        return ModelSpecModel(
            d_model=self.d_model[d_model_index],
            d_model_by_column=d_model_by_column,
            nhead=self.nhead[d_model_index],
            d_hid=np.random.choice(self.d_hid),
            nlayers=np.random.choice(self.nlayers),
        )

    def grid_sample(self, i):
        if self.hyperparamter_combinations is None:
            self.hyperparamter_combinations = list(
                product(np.arange(len(self.d_model)), self.d_hid, self.nlayers)
            )

        hyperparameter_combination = self.hyperparamter_combinations[i]

        d_model_index = hyperparameter_combination[0]
        d_model_by_column = (
            None
            if self.d_model_by_column is None
            else self.d_model_by_column[d_model_index]
        )

        return ModelSpecModel(
            d_model=self.d_model[d_model_index],
            d_model_by_column=d_model_by_column,
            nhead=self.nhead[d_model_index],
            d_hid=hyperparameter_combination[1],
            nlayers=hyperparameter_combination[2],
        )


class TrainModelHyperparameterSampling(BaseModel):
    """Pydantic model for training configuration."""

    project_path: str
    model_name_root: str
    search_strategy: str = "sample"  # "sample" or "grid"
    n_samples: Optional[int]
    training_data_path: str
    validation_data_path: str
    read_format: str = "parquet"

    selected_columns: list[list[str]]
    column_types: list[dict[str, str]]
    categorical_columns: list[str]
    real_columns: list[str]
    target_columns: list[list[str]]
    target_column_types: list[dict[str, str]]
    id_maps: dict[str, dict[str | int, int]]

    seq_length: list[int]
    n_classes: dict[str, int]
    inference_batch_size: int
    seed: int

    export_onnx: bool = True
    export_pt: bool = False
    export_with_dropout: bool = False

    model_spec: ModelSpecHyperparameterSampling
    training_spec: TrainingSpecHyperparameterSampling
