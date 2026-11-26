import csv
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
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
from sequifier.helpers import get_last_training_batch_timedelta  # noqa: E402
from sequifier.io.yaml import TrainModelDumper  # noqa: E402
from sequifier.logging_paths import model_log_directory  # noqa: E402
from sequifier.training.metrics import (  # noqa: E402
    VALIDATION_FIELDS,
    validate_metrics_csv_header,
    validate_metrics_csv_row,
)

_DUPLICATE_OF_USER_ATTR = "sequifier_duplicate_of"
_MAX_CONSECUTIVE_DUPLICATE_PROPOSALS = 1000


@dataclass(frozen=True)
class TrialObservation:
    value: float
    step: int
    completed_epochs: int


class TrialEvaluations:
    """Follow the objective phase in its native model or dataset validation table."""

    def __init__(self, run_config, root):
        self.single_phase = len(run_config.ordered_phases) == 1
        self.phase = run_config.ordered_phases[-1]
        self.monitor = self.phase.evaluation.monitor if self.phase.evaluation else None
        self.supervised = self.phase.algorithm == "supervised"
        self.direction = self.monitor.direction if self.monitor else "min"
        name = run_config.run.model_name
        directory = model_log_directory(root, name)
        self.paths = [directory / f"{name}-validation-full.csv"]
        if self.supervised and self.monitor:
            # Coordinated groups write dataset-specific files when multiple
            # datasets share the group. Both layouts record the dataset and part.
            self.paths.append(
                directory / f"{name}-{self.monitor.dataset}-validation-full.csv"
            )
        self.seen: set[str] = set()
        self.completed_epochs = 0
        self.positions: dict[Path, tuple[int, bytes]] = {}

    def _lines(self, stream, path):
        # Only advance past complete lines. Save the last line so recovery can
        # truncate and rewrite a table between polls without losing new rows.
        for line in iter(stream.readline, b""):
            if not line.endswith(b"\n"):
                break
            self.positions[path] = stream.tell(), line
            yield line.decode("utf-8")

    def rows(self):
        for path in self.paths:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                header = stream.readline()
                if not header.endswith(b"\n"):
                    continue
                names = next(csv.reader([header.decode("utf-8")]))
                validate_metrics_csv_header(names, path)
                if names not in (
                    VALIDATION_FIELDS,
                    [key for key in VALIDATION_FIELDS if key != "iterations_total"],
                ):
                    raise ValueError(
                        f"Unexpected validation metrics schema in {path}: {names!r}"
                    )
                if path in self.positions:
                    offset, previous = self.positions[path]
                    stream.seek(offset - len(previous))
                    if stream.read(len(previous)) != previous:
                        stream.seek(len(header))
                for row in csv.DictReader(self._lines(stream, path), fieldnames=names):
                    if None not in row.values():
                        validate_metrics_csv_row(row, path)
                        yield row

    def observations(self):
        if self.monitor is None:
            return
        for evaluation_id, rows in groupby(
            self.rows(), key=lambda row: row["evaluation_id"]
        ):
            if not evaluation_id or evaluation_id in self.seen:
                continue
            selected = None
            phase_clock = False
            for row in rows:
                if row["algorithm"] != self.phase.algorithm:
                    continue
                if row["phase"] != self.phase.name and not (
                    self.supervised and row["phase"] == ""
                ):
                    continue
                if self.supervised and (
                    row["dataset"] != self.monitor.dataset
                    or row["part"] != self.monitor.part
                ):
                    continue
                if row["metric"] == "epochs_completed" and (
                    self.supervised or self.phase.algorithm == "policy_imitation"
                ):
                    phase_clock = True
                    self.completed_epochs = max(
                        self.completed_epochs, int(float(row["value"]))
                    )
                if (
                    row["metric"] == self.monitor.metric
                    and row["target"] == "__total__"
                    and row["is_primary"].lower() == "true"
                ):
                    selected = row
            if selected is None or not selected["value"]:
                continue
            self.seen.add(evaluation_id)
            value = float(selected["value"])
            if not math.isfinite(value):
                continue
            if (
                self.supervised
                and self.single_phase
                and not phase_clock
                and selected["evaluation_kind"] == "epoch_end"
            ):
                self.completed_epochs = max(
                    self.completed_epochs, int(selected["epoch"])
                )
            step = int(
                selected["global_step"] if self.supervised else selected["iteration"]
            )
            yield TrialObservation(value, step, self.completed_epochs)


@beartype
def create_sampler(config: Any) -> optuna.samplers.BaseSampler:
    strategy = config.method
    sampler_seed = config.sampler_seed
    if strategy == "random":
        return optuna.samplers.RandomSampler(seed=sampler_seed)
    if strategy == "grid":
        try:
            from optuna.samplers import BruteForceSampler
        except ImportError as error:
            raise RuntimeError(
                "Grid search requires Optuna >= 3.1 for BruteForceSampler."
            ) from error
        return BruteForceSampler(seed=sampler_seed)
    return optuna.samplers.TPESampler(
        seed=sampler_seed,
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
    run_name = run_config.run.model_name
    evaluations = TrialEvaluations(run_config, config.run.project_root)

    config_path = os.path.join(
        config.run.project_root,
        config.run.outputs.trial_config_directory,
        f"{run_name}.yaml",
    )
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    final_phase = run_config.ordered_phases[-1]
    if not config.evaluation_script_path and evaluations.monitor is None:
        raise ValueError(
            "Search requires a final phase evaluation monitor or a custom evaluation script"
        )
    expected_direction = (
        config.evaluation_metric_directions or [evaluations.direction]
    )[0]
    if (
        not config.evaluation_script_path
        and expected_direction != evaluations.direction
    ):
        raise ValueError(
            "search direction must match the final phase evaluation monitor"
        )
    serialized = run_config.trial_config_values()
    with open(config_path, "w") as f:
        yaml.dump(
            serialized,
            f,
            Dumper=TrainModelDumper,
            sort_keys=False,
        )

    log_dir = model_log_directory(config.run.project_root, run_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    prune_path = (log_dir / f"{run_name}.prune").resolve()
    prune_path.unlink(missing_ok=True)
    single_objective = len(config.evaluation_metrics or ()) <= 1
    # A custom score can use a different direction from the training monitor.
    report_intermediate = (
        single_objective
        and expected_direction == evaluations.direction
        and evaluations.monitor is not None
    )
    if report_intermediate:
        evaluations.seen.update(row["evaluation_id"] for row in evaluations.rows())

    env = os.environ.copy()
    if report_intermediate and config.prune_trials:
        env["SEQUIFIER_HYPERPARAMETER_SEARCH_RUN"] = "1"
        env["SEQUIFIER_SEARCH_PRUNE_PATH"] = str(prune_path)
    else:
        env.pop("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN", None)
    cmd = ["sequifier", "train", f"--config-path={config_path}"]
    process = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=os.name == "posix",
        preexec_fn=set_pdeathsig if sys.platform.startswith("linux") else None,
    )

    def kill_training():
        # Include locally spawned training ranks and environment workers.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()

    def consume_metrics(*, running):
        if not report_intermediate:
            return
        for observation in evaluations.observations():
            trial.report(observation.value, observation.step)
            if not running or not config.prune_trials:
                continue
            if config.pruning_warmup_epochs is not None:
                ready = observation.completed_epochs >= config.pruning_warmup_epochs
            else:
                warmup = (
                    config.pruning_warmup_batches
                    if evaluations.supervised
                    else config.pruning_warmup_iterations
                )
                ready = observation.step >= (warmup or 0)
            if ready and trial.should_prune():
                prune_path.touch()
                # Policy updates can include a full rollout before the next checkpoint.
                timeout = 600.0
                if evaluations.supervised:
                    try:
                        seconds = get_last_training_batch_timedelta(
                            run_name, 0, config.run.project_root
                        )
                        timeout = 2 * seconds + 30
                    except (ValueError, FileNotFoundError):
                        timeout = 60.0
                try:
                    exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    kill_training()
                    raise optuna.TrialPruned() from None
                if exit_code == 143:
                    raise optuna.TrialPruned()
                # Training may already have finished, or failed independently.
                return

    try:
        while process.poll() is None:
            consume_metrics(running=True)
            time.sleep(2)
        consume_metrics(running=False)
        exit_code = process.returncode
        if exit_code == 143:
            if prune_path.exists():
                raise optuna.TrialPruned()
            raise RuntimeError(
                f"Trial pre-empted externally by cluster (SIGTERM). Exit code: {exit_code}"
            )
        if exit_code != 0:
            raise RuntimeError(f"Training failed with exit code {exit_code}")
    finally:
        if process.poll() is None:
            kill_training()
        prune_path.unlink(missing_ok=True)

    report_path = Path(config.run.project_root) / (
        run_config.run.outputs.report_path or f"outputs/training/{run_name}/report.json"
    )
    report = json.loads(report_path.read_text())
    phases = report["phases"]
    if not phases:
        raise ValueError("Training completed without a selected phase")
    final_report = phases[-1]
    selected_metric = final_report.get("selected_metric")
    if not config.evaluation_script_path:
        if final_report["name"] != final_phase.name:
            raise ValueError("Search plan stopped before its objective phase")
        if selected_metric is None or not math.isfinite(selected_metric):
            raise ValueError("Search requires a finite final phase selection metric")
    model_path = report["model_path"]
    evaluation_id = os.path.splitext(os.path.basename(model_path))[0]

    if config.evaluation_inference_config_path:
        inference_model_path = os.path.relpath(model_path, config.run.project_root)
        evaluation_inference_config_path = config.evaluation_inference_config_path
        if not os.path.isabs(evaluation_inference_config_path) and not os.path.exists(
            evaluation_inference_config_path
        ):
            evaluation_inference_config_path = os.path.join(
                config.run.project_root,
                evaluation_inference_config_path,
            )
        subprocess.run(
            [
                "sequifier",
                "infer",
                f"--config-path={evaluation_inference_config_path}",
                f"--model-paths={inference_model_path}",
            ],
            check=True,
        )

    if config.evaluation_script_path and config.evaluation_metrics:
        eval_script_path = config.evaluation_script_path
        cmd = [sys.executable, eval_script_path, evaluation_id]

        eval_process = subprocess.run(
            cmd, capture_output=True, text=True, cwd=config.run.project_root
        )

        if eval_process.returncode != 0:
            raise RuntimeError(
                f"Evaluation script failed (exit code {eval_process.returncode}):\n{eval_process.stderr}"
            )

        eval_json_path = os.path.join(
            config.run.project_root,
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

    return selected_metric


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


def _bind_study_objective(study, identity):
    key = "sequifier_objective"
    stored = study.user_attrs.get(key)
    if stored is None:
        if study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING),
        ):
            raise ValueError(
                "The persisted study has trials without a recorded objective identity. Use a new study name."
            )
        study.set_user_attr(key, identity)
    elif stored != identity:
        raise ValueError(
            "The persisted study uses a different objective metric or evaluation source"
        )


@beartype
def hyperparameter_search(config_path: str, skip_metadata: bool) -> None:
    """Load config, create Optuna study, and optimize trials."""
    config = load_hyperparameter_search_config(config_path, skip_metadata)

    os.makedirs(os.path.join(config.run.project_root, "state", "optuna"), exist_ok=True)

    storage_path = os.path.join(
        config.run.project_root, "state", "optuna", f"{config.run.study_name}.db"
    )

    # The accepted-run index owns the trial YAML, marker and artifact paths.
    # Hold a study-specific lock until all trial results have been committed.
    import fcntl

    with open(f"{storage_path}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_study(config, storage_path)


def _run_study(config, storage_path):
    sampler = create_sampler(config)
    is_multivariate = (
        config.evaluation_metrics is not None and len(config.evaluation_metrics) > 1
    )

    directions = [
        "minimize" if direction == "min" else "maximize"
        for direction in (config.evaluation_metric_directions or ["min"])
    ]
    study = optuna.create_study(
        study_name=config.run.study_name,
        directions=directions,
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    if [direction.name.lower() for direction in study.directions] != directions:
        raise ValueError(
            "The persisted study has different objective directions from this search"
        )

    _bind_study_objective(study, config.objective_identity)

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

    if not study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        logger.info("Study finished without a completed trial.")
        return

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
