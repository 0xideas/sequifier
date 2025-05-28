import os
from datetime import datetime

import numpy as np
import torch
import torch._dynamo
import yaml
from beartype import beartype

torch._dynamo.config.suppress_errors = True
from sequifier.config.hyperparameter_search_config import (  # noqa: E402
    load_hyperparameter_search_config,
)
from sequifier.helpers import PANDAS_TO_TORCH_TYPES  # noqa: E402
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.helpers import read_data  # noqa: E402
from sequifier.helpers import numpy_to_pytorch, subset_to_selected_columns  # noqa: E402
from sequifier.io.yaml import TrainModelDumper  # noqa: E402
from sequifier.train import TransformerModel  # noqa: E402


@beartype
def hyperparameter_search(config_path, on_unprocessed) -> None:
    hyperparameter_search_config = load_hyperparameter_search_config(
        config_path, on_unprocessed
    )

    n_combinations = hyperparameter_search_config.n_combinations()

    print(f"Found {n_combinations} hyperparameter combinations")
    if hyperparameter_search_config.search_strategy == "sample":
        n_samples = hyperparameter_search_config.n_samples
        assert n_samples is not None
        if n_samples > hyperparameter_search_config.n_combinations():
            input(
                f"{n_samples} is above the number of combinations of hyperparameters. Press any key to continue with grid search or abort to reconfigure"
            )
            n_samples = hyperparameter_search_config.n_combinations()
            hyperparameter_search_config.search_strategy = "grid"
    elif hyperparameter_search_config.search_strategy == "grid":
        n_samples = hyperparameter_search_config.n_combinations()
        input(
            f"Found {n_samples} hyperparameter combinations. Please enter any key to confirm, or change search strategy to 'sample'"
        )
    else:
        raise Exception(
            f"search strategy {hyperparameter_search_config.search_strategy} is not valid. Allowed values are 'grid' and 'sample'"
        )

    assert n_samples is not None
    for i in range(n_samples):
        np.random.seed(int(datetime.now().timestamp() * 1e6) % (2**32))
        config = hyperparameter_search_config.sample(i)

        normalized_config_path = normalize_path(
            hyperparameter_search_config.model_config_write_path,
            hyperparameter_search_config.project_path,
        )
        with open(
            os.path.join(
                normalized_config_path,
                f"{hyperparameter_search_config.model_name_root}-run-{i}.yaml",
            ),
            "w",
        ) as f:
            f.write(
                yaml.dump(
                    config,
                    Dumper=TrainModelDumper,
                    sort_keys=False,
                    default_flow_style=False,
                )
            )

        column_types = {
            col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
            for col in config.column_types
        }

        data_train = read_data(
            normalize_path(config.training_data_path, config.project_path),
            config.read_format,
        )
        if config.selected_columns is not None:
            data_train = subset_to_selected_columns(data_train, config.selected_columns)

        X_train, y_train = numpy_to_pytorch(
            data_train,
            column_types,
            config.selected_columns,
            config.target_columns,
            config.seq_length,
            config.training_spec.device,
            to_device=False,
        )
        del data_train

        data_valid = read_data(
            normalize_path(config.validation_data_path, config.project_path),
            config.read_format,
        )
        if config.selected_columns is not None:
            data_valid = subset_to_selected_columns(data_valid, config.selected_columns)

        X_valid, y_valid = numpy_to_pytorch(
            data_valid,
            column_types,
            config.selected_columns,
            config.target_columns,
            config.seq_length,
            config.training_spec.device,
            to_device=False,
        )
        del data_valid

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        model = torch.compile(TransformerModel(config).to(config.training_spec.device))

        model.train_model(X_train, y_train, X_valid, y_valid)
