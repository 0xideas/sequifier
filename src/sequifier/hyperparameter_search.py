import glob
import os
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch._dynamo
import yaml
from beartype import beartype

torch._dynamo.config.suppress_errors = True
from sequifier.config.hyperparameter_search_config import (  # noqa: E402
    load_hyperparameter_search_config,
)
from sequifier.config.train_config import TrainModel  # noqa: E402
from sequifier.helpers import LogFile  # noqa: E402
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.io.yaml import TrainModelDumper  # noqa: E402


@beartype
def hyperparameter_search(config_path, on_unprocessed) -> None:
    hyperparameter_search_config = load_hyperparameter_search_config(
        config_path, on_unprocessed
    )

    hyperparameter_searcher = HyperparameterSearcher(hyperparameter_search_config)

    hyperparameter_searcher.hyperparameter_search()


class HyperparameterSearcher:
    def __init__(self, hyperparameter_search_config):
        self.config = hyperparameter_search_config
        self.normalized_config_path = normalize_path(
            self.config.model_config_write_path,
            self.config.project_path,
        )
        self.start_run = self._get_start_run()
        self._initialize_log_file()
        self.n_samples = self._calculate_n_samples()

    @beartype
    def _get_start_run(self) -> int:
        file_root = f"{self.config.hp_search_name}-run-"
        search_pattern = os.path.join(self.normalized_config_path, f"{file_root}*.yaml")
        files = [os.path.split(file)[1] for file in glob.glob(search_pattern)]
        files.sort(
            key=lambda filename: int(filename.replace(file_root, "").split(".")[0])
        )

        if len(files) > 0:
            last_iter = int(files[-1].split(".")[0].replace(file_root, ""))
            return last_iter + 1
        else:
            return 1

    @beartype
    def _initialize_log_file(self) -> None:
        os.makedirs(os.path.join(self.config.project_path, "logs"), exist_ok=True)
        open_mode = "w" if self.start_run == 1 else "a"
        self.log_file = LogFile(
            os.path.join(
                self.config.project_path,
                "logs",
                f"sequifier-{self.config.hp_search_name}-[NUMBER].txt",
            ),
            open_mode,
        )

    @beartype
    def _calculate_n_samples(self) -> int:
        n_combinations = self.config.n_combinations()
        print(f"Found {n_combinations} hyperparameter combinations")
        if self.config.search_strategy == "sample":
            n_samples = self.config.n_samples
            assert n_samples is not None
            if n_samples > self.config.n_combinations():
                input(
                    f"{n_samples} is above the number of combinations of hyperparameters. Press any key to continue with grid search or abort to reconfigure"
                )
                n_samples = self.config.n_combinations()
                self.config.search_strategy = "grid"
        elif self.config.search_strategy == "grid":
            n_samples = self.config.n_combinations()
            input(
                f"Found {n_samples} hyperparameter combinations. Please enter any key to confirm, or change search strategy to 'sample'"
            )
        else:
            raise Exception(
                f"search strategy {self.config.search_strategy} is not valid. Allowed values are 'grid' and 'sample'"
            )

        assert n_samples is not None

        return n_samples

    def _create_config_and_run(
        self, i: int, seed: int, config: Optional[TrainModel] = None, attempt=0
    ):
        if config is None:
            config = self.config.sample(i)
        full_config_path = os.path.join(
            self.normalized_config_path,
            f"{self.config.hp_search_name}-run-{i}.yaml",
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

        self.log_file.write(
            f"--- Starting Hyperparameter Search Run {i} with seed {seed} ---"
        )
        try:
            subprocess.run(
                [
                    "sequifier",
                    "train",
                    f"--config-path={full_config_path}",
                    f"--seed={seed}",
                ],
                check=True,
            )
            self.log_file.write(f"--- Finished Hyperparameter Search Run {i} ---")

        except subprocess.CalledProcessError as e:
            if attempt < 3:
                assert config is not None
                new_batch_size = int(config.training_spec.batch_size / 2)

                assert new_batch_size > 0
                config.training_spec.batch_size = new_batch_size
                self.log_file.write(
                    f"ERROR: Run {i} failed with exit code {e.returncode}. Halving batch size to {new_batch_size} in attempt {attempt + 1}"
                )
                self._create_config_and_run(i, seed, config, attempt=attempt + 1)
            else:
                self.log_file.write(
                    f"ERROR: Run {i} failed with exit code {e.returncode}. Stopping run {i}"
                )

    @beartype
    def hyperparameter_search(self) -> None:
        for i in range(self.start_run, self.n_samples + 1):
            seed = int(datetime.now().timestamp() * 1e6) % (2**32)
            np.random.seed(seed)
            self._create_config_and_run(i, seed=seed)
