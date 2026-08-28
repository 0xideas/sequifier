import csv
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import warnings
from typing import Any, Union

import optuna
import torch._dynamo
import yaml
from loguru import logger
from optuna.trial import TrialState

from sequifier.typechecking import beartype

torch._dynamo.config.suppress_errors = True
from sequifier.config.hyperparameter_search_config import (  # noqa: E402
    load_hyperparameter_search_config,
)
from sequifier.helpers import (  # noqa: E402
    get_best_model_path,
    get_last_training_batch_timedelta,
)
from sequifier.io.yaml import TrainModelDumper  # noqa: E402
from sequifier.logging_paths import (  # noqa: E402
    dataset_artifact_prefix,
    model_log_directory,
)
from sequifier.training.metrics import VALIDATION_FIELDS  # noqa: E402

_DUPLICATE_OF_USER_ATTR = "sequifier_duplicate_of"
_MAX_CONSECUTIVE_DUPLICATE_PROPOSALS = 1000


@beartype
def _monitored_dataset(run_config: Any) -> tuple[str | None, int]:
    dataset_names = tuple(run_config.dataset_training)
    dataset_count = len(dataset_names)
    if dataset_count == 1:
        return None, dataset_count
    evaluation = getattr(run_config, "evaluation", None)
    monitor = getattr(evaluation, "monitor", None)
    if monitor is None:
        raise ValueError(
            "Multi-dataset hyperparameter search requires evaluation.monitor"
        )
    dataset_name = monitor.source.split(".", 1)[0]
    if dataset_name not in run_config.dataset_training:
        raise ValueError(
            f"evaluation.monitor references unknown dataset {dataset_name!r}"
        )
    return dataset_name, dataset_count


@beartype
def create_sampler(config: Any) -> optuna.samplers.BaseSampler:
    strategy = getattr(config, "method", "bayesian")
    global_seed = getattr(config, "global_seed", None)
    if strategy in ["sample"]:
        return optuna.samplers.RandomSampler(seed=global_seed)
    if strategy == "grid":
        if hasattr(optuna.samplers, "BruteForceSampler"):
            return optuna.samplers.BruteForceSampler(seed=global_seed)
        raise RuntimeError("Grid search requires Optuna >= 3.1 for BruteForceSampler.")
    return optuna.samplers.TPESampler(
        seed=global_seed,
        multivariate=True,
    )


@beartype
def set_pdeathsig():
    """Ask Linux to SIGTERM children when this parent dies."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1


@beartype
def objective(
    trial: optuna.Trial, accepted_trials: int, config, run_config: Any = None
) -> Union[float, tuple[float, ...]]:
    """Run one Optuna trial through the CLI trainer and validation metrics."""
    if run_config is None:
        run_config = config.sample_trial(trial, accepted_trials)
    run_name = run_config.model_name
    monitored_dataset, dataset_count = _monitored_dataset(run_config)

    config_path = os.path.join(
        config.project_root, config.model_config_write_path, f"{run_name}.yaml"
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(
            run_config.model_dump(mode="python"),
            f,
            Dumper=TrainModelDumper,
            sort_keys=False,
        )

    log_dir = model_log_directory(config.project_root, run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    validation_path = (
        str(
            dataset_artifact_prefix(
                config.project_root,
                run_name,
                dataset_name=monitored_dataset,
                dataset_count=dataset_count,
            )
        )
        + "-validation.csv"
    )
    prune_path = str(log_dir / f"{run_name}.prune")
    consumed_evaluation_ids: set[str] = set()
    if os.path.exists(validation_path):
        with open(validation_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != VALIDATION_FIELDS:
                raise ValueError(
                    f"Unexpected validation metrics schema in {validation_path}: "
                    f"{reader.fieldnames!r}"
                )
            consumed_evaluation_ids.update(
                row["evaluation_id"] for row in reader if row.get("evaluation_id")
            )

    env = os.environ.copy()
    env["SEQUIFIER_HYPERPARAMETER_SEARCH_RUN"] = "1"
    cmd = ["sequifier", "train", f"--config-path={config_path}"]
    process = subprocess.Popen(
        cmd,
        env=env,
        preexec_fn=set_pdeathsig if sys.platform.startswith("linux") else None,
    )

    best_val_loss = float("inf")
    completed_epochs = 0

    @beartype
    def consume_metrics(
        best_val_loss: float, completed_epochs: int
    ) -> tuple[float, int]:
        """Read complete validation rows; report/prune single-objective trials."""
        if os.path.exists(validation_path):
            with open(validation_path, "r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != VALIDATION_FIELDS:
                    raise ValueError(
                        f"Unexpected validation metrics schema in "
                        f"{validation_path}: {reader.fieldnames!r}"
                    )
                for data in reader:
                    evaluation_id = data.get("evaluation_id")
                    if (
                        not evaluation_id
                        or evaluation_id in consumed_evaluation_ids
                        or data.get("metric") != "loss"
                        or data.get("target") != "__total__"
                        or not data.get("value")
                        or not data.get("global_step")
                        or not data.get("epoch")
                    ):
                        continue

                    val_loss = float(data["value"])
                    global_step = int(data["global_step"])
                    metric_epoch = int(data["epoch"])
                    if data.get("evaluation_kind") == "epoch_end":
                        completed_epochs = max(completed_epochs, metric_epoch)

                    is_multi_objective = (
                        config.evaluation_metrics is not None
                        and len(config.evaluation_metrics) > 1
                    )
                    if not is_multi_objective:
                        trial.report(val_loss, global_step)
                        best_val_loss = min(best_val_loss, val_loss)

                        if config.pruning_warmup_batches is not None:
                            warmup_complete = (
                                global_step >= config.pruning_warmup_batches
                            )
                        else:
                            pruning_warmup_epochs = config.pruning_warmup_epochs or 0
                            warmup_complete = completed_epochs >= pruning_warmup_epochs
                        if (
                            config.prune_trials
                            and warmup_complete
                            and trial.should_prune()
                        ):
                            open(prune_path, "w").close()
                            try:
                                try:
                                    timedelta = get_last_training_batch_timedelta(
                                        run_name, 0, config.project_root
                                    )
                                    timeout_val = (timedelta * 2) + 30
                                except (ValueError, FileNotFoundError):
                                    timeout_val = 60.0

                                process.wait(timeout=timeout_val)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            raise optuna.TrialPruned()
                    consumed_evaluation_ids.add(evaluation_id)
        return best_val_loss, completed_epochs

    while process.poll() is None:
        best_val_loss, completed_epochs = consume_metrics(
            best_val_loss, completed_epochs
        )
        time.sleep(2)

    best_val_loss, _ = consume_metrics(best_val_loss, completed_epochs)

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

    model_type = "onnx" if run_config.export_onnx else "pt"
    model_path, _last_epoch = get_best_model_path(
        config.project_root,
        run_name,
        model_type,
        dataset_name=(monitored_dataset if model_type == "onnx" else None),
        dataset_count=(dataset_count if model_type == "onnx" else 1),
    )
    evaluation_id = os.path.splitext(os.path.basename(model_path))[0]

    if config.evaluation_inference_config:
        evaluation_inference_config = config.evaluation_inference_config
        if not os.path.isabs(evaluation_inference_config) and not os.path.exists(
            evaluation_inference_config
        ):
            evaluation_inference_config = os.path.join(
                config.project_root,
                evaluation_inference_config,
            )
        subprocess.run(
            [
                "sequifier",
                "infer",
                f"--config-path={evaluation_inference_config}",
                f"--model-path={model_path}",
            ],
            check=True,
        )

    if config.evaluation_script and config.evaluation_metrics:
        eval_script_path = config.evaluation_script
        cmd = [sys.executable, eval_script_path, evaluation_id]

        eval_process = subprocess.run(
            cmd, capture_output=True, text=True, cwd=config.project_root
        )

        if eval_process.returncode != 0:
            raise RuntimeError(
                f"Evaluation script failed (exit code {eval_process.returncode}):\n{eval_process.stderr}"
            )

        eval_json_path = os.path.join(
            config.project_root,
            "outputs",
            "evaluations",
            f"{evaluation_id}.json",
        )
        if not os.path.exists(eval_json_path):
            raise FileNotFoundError(
                f"Evaluation JSON not found at expected path: {eval_json_path}"
            )

        with open(eval_json_path, "r") as f:
            eval_results = json.load(f)
            eval_results_keys = set(list(eval_results.keys()))
            evaluation_metrics = set(config.evaluation_metrics)
            missing_metrics = evaluation_metrics.difference(eval_results_keys)
            excess_metrics = eval_results_keys.difference(evaluation_metrics)
            if len(missing_metrics):
                raise ValueError(
                    f"Some of the configured evaluation metrics are not in the script output: {missing_metrics}"
                )
            if len(excess_metrics):
                warnings.warn(
                    f"Some metrics output by the script are not used in hyperparameter optimization: {excess_metrics}"
                )

        metrics = []
        for metric in config.evaluation_metrics:
            if metric not in eval_results:
                raise KeyError(
                    f"Metric '{metric}' missing in {eval_json_path}. Found keys: {list(eval_results.keys())}"
                )
            value = eval_results[metric]
            metrics.append(float("nan") if value is None else float(value))

        if len(metrics) == 1:
            return metrics[0]
        else:
            return tuple(metrics)

    return best_val_loss


@beartype
def _parameter_signature(params: dict[str, Any]) -> str:
    """Return a stable identity for one fully sampled Optuna parameter set."""
    return json.dumps(
        params,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@beartype
def _trained_parameter_signatures(study: optuna.Study) -> dict[str, int]:
    """Map trained parameter sets to the first trial that used each set."""
    signatures: dict[str, int] = {}
    trained_states = (TrialState.COMPLETE, TrialState.PRUNED)
    for trial in study.get_trials(deepcopy=False, states=trained_states):
        signatures.setdefault(_parameter_signature(trial.params), trial.number)
    return signatures


@beartype
def _trained_trial_count(study: optuna.Study) -> int:
    """Count completed and pruned trials that consumed a training run."""
    return len(
        study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED),
        )
    )


@beartype
def _optimize_distinct_trials(study: optuna.Study, config: Any, trials: int) -> None:
    """Train novel configurations until the study contains ``trials`` runs."""
    trained_signatures = _trained_parameter_signatures(study)
    accepted_trials = _trained_trial_count(study)
    consecutive_duplicates = 0

    if accepted_trials >= trials:
        logger.info(
            "Hyperparameter study already contains {} trained trials; "
            "requested total is {}.",
            accepted_trials,
            trials,
        )
        return

    while accepted_trials < trials:
        trial = study.ask()
        try:
            run_config = config.sample_trial(trial, accepted_trials)
        except (Exception, KeyboardInterrupt):
            study.tell(trial, state=TrialState.FAIL)
            raise

        signature = _parameter_signature(trial.params)
        duplicate_of = trained_signatures.get(signature)
        if duplicate_of is not None:
            trial.set_user_attr(_DUPLICATE_OF_USER_ATTR, duplicate_of)
            study.tell(trial, state=TrialState.FAIL)
            consecutive_duplicates += 1
            logger.info(
                "Skipping duplicate hyperparameter trial {} (already trained "
                "as trial {}).",
                trial.number,
                duplicate_of,
            )
            if consecutive_duplicates >= _MAX_CONSECUTIVE_DUPLICATE_PROPOSALS:
                raise RuntimeError(
                    "Unable to sample a novel hyperparameter configuration after "
                    f"{consecutive_duplicates} consecutive duplicate proposals. "
                    "The search space may be exhausted."
                )
            continue

        consecutive_duplicates = 0
        try:
            value = objective(trial, accepted_trials, config, run_config=run_config)
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        except (Exception, KeyboardInterrupt):
            study.tell(trial, state=TrialState.FAIL)
            raise
        else:
            study.tell(trial, value)

        trained_signatures[signature] = trial.number
        accepted_trials += 1


@beartype
def hyperparameter_search(config_path: str, skip_metadata: bool) -> None:
    """Load config, create Optuna study, and optimize trials."""
    config = load_hyperparameter_search_config(config_path, skip_metadata)

    os.makedirs(os.path.join(config.project_root, "state", "optuna"), exist_ok=True)
    sampler = create_sampler(config)

    storage_path = os.path.join(
        config.project_root, "state", "optuna", f"{config.name}.db"
    )

    is_multivariate = (
        config.evaluation_metrics is not None and len(config.evaluation_metrics) > 1
    )

    if is_multivariate:
        study = optuna.create_study(
            study_name=config.name,
            directions=config.evaluation_metric_directions,
            sampler=sampler,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
        )
    else:
        direction = (
            config.evaluation_metric_directions[0]
            if (
                config.evaluation_metric_directions
                and len(config.evaluation_metric_directions) == 1
            )
            else "minimize"
        )
        study = optuna.create_study(
            study_name=config.name,
            direction=direction,
            sampler=sampler,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
        )

    trials = config.trials
    if trials is None and config.method != "grid":
        raise ValueError("trials must be specified for hyperparameter search.")

    if config.method == "grid":
        accepted_trials = _trained_trial_count(study)
        remaining_trials = None if trials is None else max(0, trials - accepted_trials)

        @beartype
        def grid_objective(trial: optuna.Trial):
            nonlocal accepted_trials
            try:
                value = objective(trial, accepted_trials, config)
            except optuna.TrialPruned:
                accepted_trials += 1
                raise
            accepted_trials += 1
            return value

        if remaining_trials is None or remaining_trials > 0:
            study.optimize(grid_objective, n_trials=remaining_trials)
    else:
        assert trials is not None
        _optimize_distinct_trials(study, config, trials)

    if is_multivariate:
        print("\nBest trials (Pareto front):")
        for trial in study.best_trials:
            print(f"  Values: {trial.values}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
    else:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
