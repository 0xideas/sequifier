import os
import subprocess
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
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.io.yaml import TrainModelDumper  # noqa: E402


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
        full_config_path = os.path.join(
            normalized_config_path,
            f"{hyperparameter_search_config.model_name_root}-run-{i}.yaml",
        )
        with open(full_config_path, "w") as f:
            f.write(
                yaml.dump(
                    config,
                    Dumper=TrainModelDumper,
                    sort_keys=False,
                    default_flow_style=False,
                )
            )

        print(f"--- Starting Hyperparameter Search Run {i} ---")
        try:
            # This is the recommended way to run the command.
            # It waits for the command to complete.
            # `check=True` will raise an exception if the training fails.
            subprocess.run(
                ["sequifier", "train", f"--config-path={full_config_path}"], check=True
            )
        except subprocess.CalledProcessError as e:
            print(
                f"!!! ERROR: Run {i} failed with exit code {e.returncode}. Stopping hyperparameter search. !!!"
            )
        except Exception as e:
            raise e

        print(f"--- Finished Hyperparameter Search Run {i} ---")
