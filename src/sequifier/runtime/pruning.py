"""Cooperative Optuna pruning at native training boundaries."""

import os
from pathlib import Path

from sequifier.logging_paths import model_log_directory


def pruning_requested(config, strategy):
    if not os.environ.get("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN"):
        return False
    requested = False
    if strategy.rank == 0:
        marker = os.environ.get("SEQUIFIER_SEARCH_PRUNE_PATH")
        path = (
            Path(marker)
            if marker
            else (
                model_log_directory(config.project_root, config.model_name)
                / f"{config.model_name}.prune"
            )
        )
        requested = path.is_file()
    return strategy.broadcast(requested) if strategy.world_size > 1 else requested
