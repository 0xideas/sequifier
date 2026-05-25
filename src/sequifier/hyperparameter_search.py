import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time

import optuna
import torch._dynamo
import yaml
from beartype import beartype

torch._dynamo.config.suppress_errors = True
from sequifier.config.hyperparameter_search_config import (  # noqa: E402
    load_hyperparameter_search_config,
)
from sequifier.io.yaml import TrainModelDumper  # noqa: E402


def get_free_port() -> int:
    """Dynamically binds to socket 0 to retrieve a free port for NCCL."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_pdeathsig():
    """Binds child process lifecycle to the parent orchestrator via Linux prctl."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1


def objective(trial: optuna.Trial, config) -> float:
    """The central objective engine bridging Optuna to pure CLI execution.

    This function handles generating the YAML configuration for the specific
    trial, dynamically allocating a port for distributed training, launching the
    training subprocess, asynchronously polling the validation metrics, and reporting
    them back to Optuna for potential pruning.

    Args:
        trial (optuna.Trial): The Optuna trial object managing the current hyperparameter combination.
        config (HyperparameterSearchConfig): The parsed hyperparameter search configuration.

    Returns:
        float: The best validation loss achieved during the trial.

    Raises:
        optuna.TrialPruned: If the trial is pruned by the Optuna orchestrator.
        RuntimeError: If the training subprocess fails or is externally preempted.
    """
    run_config = config.sample_trial(trial, trial.number)
    run_name = run_config.model_name

    # 1. YAML Generation
    config_path = os.path.join(
        config.project_root, config.model_config_write_path, f"{run_name}.yaml"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(run_config, f, Dumper=TrainModelDumper, sort_keys=False)

    # 2. Dynamic Port Allocation
    env = os.environ.copy()
    env["MASTER_PORT"] = str(get_free_port())

    # 3. Subprocess Launch (Worker Isolation)
    cmd = ["sequifier", "train", f"--config-path={config_path}"]
    process = subprocess.Popen(
        cmd,
        env=env,
        preexec_fn=set_pdeathsig if sys.platform.startswith("linux") else None,
    )

    metrics_path = os.path.join(
        config.project_root, "logs", f"sequifier-{run_name}-metrics.jsonl"
    )
    prune_path = os.path.join(
        config.project_root, "logs", f"sequifier-{run_name}.prune"
    )

    last_read_pos = 0
    best_val_loss = float("inf")

    # 4. Asynchronous Polling & Caching Mitigation
    while process.poll() is None:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                f.seek(last_read_pos)
                for line in f:
                    try:
                        data = json.loads(line)
                        epoch = data.get("epoch")
                        val_loss = data.get("val_loss")
                        if epoch is not None and val_loss is not None:
                            # 5. Cooperative Pruning Evaluation
                            trial.report(val_loss, epoch)
                            best_val_loss = min(best_val_loss, val_loss)

                            if trial.should_prune():
                                open(prune_path, "w").close()
                                try:
                                    process.wait(timeout=60)
                                except subprocess.TimeoutExpired:
                                    process.kill()  # Escalation
                                raise optuna.TrialPruned()

                    except json.JSONDecodeError:
                        pass  # Incomplete line handling (fsync latency)
                last_read_pos = f.tell()
        time.sleep(2)

    # 6. Exit Code Disambiguation
    exit_code = process.returncode
    if exit_code == 143:
        if os.path.exists(prune_path):
            raise optuna.TrialPruned()
        else:
            raise RuntimeError(
                f"Trial pre-empted externally by cluster (SIGTERM). Exit code: {exit_code}"
            )
    elif exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}")

    return best_val_loss


@beartype
def hyperparameter_search(config_path: str, skip_metadata: bool) -> None:
    """Main function for initiating an Optuna-based hyperparameter search process.

    This function loads the configuration, initializes the Optuna study with a
    minimization direction, and kicks off the optimization loop. Once the configured
    number of trials is complete, it prints out the best trial's value and hyperparameters.

    Args:
        config_path (str): Path to the hyperparameter search YAML configuration file.
        skip_metadata (bool): Flag indicating whether to skip loading/processing data metadata.

    Raises:
        ValueError: If `n_trials` is not defined in the configuration.
    """
    config = load_hyperparameter_search_config(config_path, skip_metadata)

    strategy = getattr(config, "search_strategy", "bayesian")
    if strategy in ["sample", "random"]:
        sampler = optuna.samplers.RandomSampler()
    elif strategy == "grid":
        if hasattr(optuna.samplers, "BruteForceSampler"):
            sampler = optuna.samplers.BruteForceSampler()
        else:
            raise RuntimeError(
                "Grid search requires Optuna >= 3.1 for BruteForceSampler."
            )
    else:  # "bayesian"
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name=config.hp_search_name, direction="minimize", sampler=sampler
    )

    n_trials = config.n_trials
    if n_trials is None:
        raise ValueError(
            "n_trials/n_samples must be specified for hyperparameter search."
        )

    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
