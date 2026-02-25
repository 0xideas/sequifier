import argparse
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Import Loguru and your custom logger config
from loguru import logger
from plotly.subplots import make_subplots

from sequifier.helpers import configure_logger

# -------------------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------------------
VAL_PATTERN = re.compile(
    r"\[INFO\] Validation\s+\|\s*Epoch:\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)\s+\|\s*Baseline Loss:\s*([^\s\|]+)"
)
VAR_PATTERN = re.compile(r"\[INFO\]\s+-\s+(.*)")
TRAIN_PATTERN = re.compile(
    r"\[INFO\] Epoch\s*(\d+)\s+\|\s*Batch\s*(\d+)/\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)"
)


# -------------------------------------------------------------------------
# Custom Exceptions & Dataclasses
# -------------------------------------------------------------------------
class LogParsingError(Exception):
    """Raised when a log line does not conform to the expected regex pattern."""

    pass


class DataContinuityError(Exception):
    """Raised when training batches or epochs violate chronological order."""

    pass


@dataclass
class TrainingMetrics:
    """Encapsulates all extracted metrics to avoid returning massive generic tuples."""

    val_losses: Dict[int, float] = field(default_factory=dict)
    baseline_losses: Dict[int, float] = field(default_factory=dict)
    var_losses: Dict[str, Dict[Optional[int], float]] = field(default_factory=dict)
    train_losses: Dict[int, Dict[int, Tuple[int, float]]] = field(default_factory=dict)

    def clear_state(self) -> None:
        """Clears all metrics; used when a sequence run restarts."""
        self.val_losses.clear()
        self.baseline_losses.clear()
        self.var_losses.clear()
        self.train_losses.clear()


# -------------------------------------------------------------------------
# Core Parsing Logic
# -------------------------------------------------------------------------
class LogParser:
    """Handles line-by-line log parsing, encapsulating state to reduce complexity."""

    def __init__(self, model_name: str):
        self.model = model_name
        self.metrics = TrainingMetrics()
        self.current_epoch: Optional[int] = None
        self.current_batch: Optional[int] = None
        self.expected_num_batches: Optional[int] = None
        self.pending_var_loss_epoch: Optional[int] = None

    def parse_file(self, log_file: str) -> TrainingMetrics:
        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    self._process_line(line)
                except Exception as e:
                    raise LogParsingError(f"[{self.model} Line {line_num}]: {e}")

        self._validate_final_metrics()
        return self.metrics

    def _process_line(self, line: str) -> None:
        """Routes the line to the appropriate sub-parser based on strict string matching."""
        if "[INFO] Validation | Epoch:" in line:
            self._process_validation(line)
        elif self.pending_var_loss_epoch is not None and "[INFO]  - " in line:
            self._process_var_loss(line)
        elif "[INFO] Epoch" in line and "| Batch" in line:
            self._process_training(line)
        elif "[INFO] Epoch" in line or "[INFO] Validation" in line:
            self.pending_var_loss_epoch = None

    def _process_validation(self, line: str) -> None:
        match = VAL_PATTERN.search(line)
        if not match:
            raise LogParsingError(f"Malformed Validation log -> '{line.strip()}'")

        epoch = int(match.group(1))
        val_loss = parse_number(match.group(2))
        baseline = parse_number(match.group(3))

        if epoch == 0 or (
            self.current_epoch is not None and epoch < self.current_epoch
        ):
            self.metrics.clear_state()
            self.current_epoch = None
            self.current_batch = None
            self.expected_num_batches = None

        self.metrics.val_losses[epoch] = val_loss
        self.metrics.baseline_losses[epoch] = baseline
        self.pending_var_loss_epoch = epoch

    def _process_var_loss(self, line: str) -> None:
        match = VAR_PATTERN.search(line)
        if not match:
            raise LogParsingError(f"Malformed Variable Loss log -> '{line.strip()}'")

        for part in match.group(1).split(","):
            if ":" not in part:
                raise LogParsingError(
                    f"Missing ':' in variable loss pair -> '{part.strip()}'"
                )

            var_name, v_loss_str = part.split(":", 1)
            var_name = var_name.strip().replace("_loss", "")

            if var_name not in self.metrics.var_losses:
                self.metrics.var_losses[var_name] = {}

            self.metrics.var_losses[var_name][self.pending_var_loss_epoch] = (
                parse_number(v_loss_str)
            )

        self.pending_var_loss_epoch = None

    def _process_training(self, line: str) -> None:
        match = TRAIN_PATTERN.search(line)
        if not match:
            raise LogParsingError(f"Malformed Training Batch log -> '{line.strip()}'")

        epoch, batch, num_batches = map(int, match.groups()[:3])
        loss = parse_number(match.group(4))

        self._validate_chronology(epoch, batch, num_batches)

        self.current_epoch = epoch
        self.current_batch = batch
        self.expected_num_batches = num_batches

        if epoch == 1 and batch == 1 and 0 not in self.metrics.train_losses:
            self._handle_epoch_1_restart()

        if epoch not in self.metrics.train_losses:
            self.metrics.train_losses[epoch] = {}

        if batch in self.metrics.train_losses[epoch]:
            raise DataContinuityError(
                f"Duplicate batch {batch} recorded for Epoch {epoch}."
            )

        self.metrics.train_losses[epoch][batch] = (num_batches, loss)

    def _validate_chronology(self, epoch: int, batch: int, num_batches: int) -> None:
        if self.current_epoch is not None and self.current_batch is not None:
            if epoch == self.current_epoch and batch <= self.current_batch:
                raise DataContinuityError(
                    f"Batch monotonicity violated (was {self.current_batch}, now {batch})."
                )
            if epoch not in (self.current_epoch, self.current_epoch + 1) and not (
                epoch == 1 and batch <= num_batches
            ):
                raise DataContinuityError(
                    f"Epoch transition violated (was {self.current_epoch}, now {epoch})."
                )

        if (
            self.expected_num_batches is not None
            and num_batches != self.expected_num_batches
        ):
            if epoch == self.current_epoch:
                raise DataContinuityError(
                    f"Inconsistent num_batches mid-epoch (was {self.expected_num_batches}, now {num_batches})."
                )

    def _handle_epoch_1_restart(self) -> None:
        """Handles edge cases where a sequence restarts at Epoch 1 skipping Epoch 0."""
        if 0 not in self.metrics.val_losses:
            self.metrics.clear_state()
        else:
            self.metrics.train_losses.clear()
            self.metrics.val_losses = {0: self.metrics.val_losses[0]}
            self.metrics.baseline_losses = {0: self.metrics.baseline_losses[0]}
            for v_name in list(self.metrics.var_losses.keys()):
                if 0 in self.metrics.var_losses[v_name]:
                    self.metrics.var_losses[v_name] = {
                        0: self.metrics.var_losses[v_name][0]
                    }
                else:
                    self.metrics.var_losses[v_name] = {}

    def _validate_final_metrics(self) -> None:
        if not self.metrics.train_losses:
            raise DataContinuityError(
                f"[{self.model}]: No valid training loss data found."
            )
        if not self.metrics.val_losses:
            raise DataContinuityError(
                f"[{self.model}]: No valid validation loss data found."
            )
        if not self.metrics.baseline_losses:
            raise DataContinuityError(f"[{self.model}]: No baseline loss data found.")


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def parse_number(val: str) -> float:
    """Strictly parse numbers, explicitly handling the 'NaN' strings."""
    val = val.strip()
    return np.nan if val == "NaN" else float(val)


def parse_args_to_models(args: argparse.Namespace) -> List[str]:
    """Extracts the list of models from a file or comma-separated string."""
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        return [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]

    return [m.strip() for m in args.models.split(",") if m.strip()]


def get_log_filepath(args: argparse.Namespace, model: str) -> str:
    """Finds the appropriate log file for a given model."""
    log_pattern = os.path.join(
        args.project_root, "logs", f"sequifier-{model}-rank0-3.txt"
    )
    log_files = glob.glob(log_pattern)

    if not log_files:
        log_pattern = os.path.join(
            args.project_root, "logs", f"sequifier-{model}-rank0-2.txt"
        )
        log_files = glob.glob(log_pattern)

    if not log_files:
        raise FileNotFoundError(
            f"No log files found for model '{model}' matching the expected pattern."
        )

    return log_files[0]


def format_plot_data(
    metrics: TrainingMetrics, bucket_batches: Optional[int], model: str
) -> Dict[str, Any]:
    """Formats raw parsed dataclass metrics into chronological arrays for Plotly."""
    val_x = sorted(list(metrics.val_losses.keys()))
    val_y = [metrics.val_losses[e] for e in val_x]
    base_y = [metrics.baseline_losses[e] for e in val_x]

    train_x, train_y = [], []

    for epoch in sorted(list(metrics.train_losses.keys())):
        epoch_dict = metrics.train_losses[epoch]
        if not epoch_dict:
            continue

        epoch_data = [
            (b, epoch_dict[b][0], epoch_dict[b][1]) for b in sorted(epoch_dict.keys())
        ]

        if bucket_batches is not None:
            log_interval = (
                epoch_data[1][0] - epoch_data[0][0]
                if len(epoch_data) > 1
                else epoch_data[0][0]
            )
            log_interval = max(log_interval, 1)

            if bucket_batches % log_interval != 0:
                raise ValueError(
                    f"[{model} Epoch {epoch}]: --bucket-training-batches ({bucket_batches}) "
                    f"MUST be a multiple of the logged batch interval ({log_interval})."
                )

            chunk_size = bucket_batches // log_interval
            for i in range(0, len(epoch_data), chunk_size):
                chunk = epoch_data[i : i + chunk_size]
                avg_loss = sum(c[2] for c in chunk) / len(chunk)
                last_batch, num_batches = chunk[-1][0], chunk[-1][1]
                train_x.append(round(epoch - 1 + last_batch / num_batches, 8))
                train_y.append(avg_loss)
        else:
            for batch, num_batches, loss in epoch_data:
                train_x.append(round(epoch - 1 + batch / num_batches, 8))
                train_y.append(loss)

    if not train_x:
        raise DataContinuityError(
            f"[{model}]: Training arrays ended up empty after formatting."
        )

    return {
        "val_x": val_x,
        "val_y": val_y,
        "base_y": base_y,
        "train_x": train_x,
        "train_y": train_y,
        "var_losses": metrics.var_losses,
    }


# -------------------------------------------------------------------------
# Plotting & Reporting
# -------------------------------------------------------------------------
def _generate_single_model_plot(
    model: str, data: Dict[str, Any], yaxis_type: str, out_path: str
) -> None:
    """Handles subplot logic specifically for a single model."""
    has_var_losses = bool(data.get("var_losses"))
    subplot_titles = (
        ("Global Losses", "Normalized Variable Validation Losses")
        if has_var_losses
        else ("Global Losses", "")
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    fig.add_trace(
        go.Scatter(
            x=data["val_x"], y=data["val_y"], mode="lines", name="Validation Loss"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["train_x"], y=data["train_y"], mode="lines", name="Training Loss"
        ),
        row=1,
        col=1,
    )

    if data["base_y"]:
        fig.add_trace(
            go.Scatter(
                x=data["val_x"],
                y=data["base_y"],
                mode="lines",
                name="Baseline Loss",
                line=dict(dash="dash"),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="Loss", type=yaxis_type, row=1, col=1)

    if has_var_losses:
        for var, epoch_dict in data["var_losses"].items():
            epochs = sorted(list(epoch_dict.keys()))
            if not epochs:
                continue
            base_val = epoch_dict[epochs[0]]
            y_norm = [
                epoch_dict[e] / base_val
                if base_val != 0 and not np.isnan(base_val)
                else epoch_dict[e]
                for e in epochs
            ]
            fig.add_trace(
                go.Scatter(x=epochs, y=y_norm, mode="lines", name=var), row=1, col=2
            )

        fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=2)
        fig.update_yaxes(
            title_text="Loss / Epoch 0 Loss", type=yaxis_type, row=1, col=2
        )
    else:
        logger.warning(
            f"No variable validation losses found for model '{model}'. Second subplot will be empty."
        )

    fig.update_layout(title_text=f"Training Visualization: {model}")
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"Visualization HTML generated and saved successfully to {out_path}")


def _generate_multi_model_plot(
    models: List[str], all_data: Dict[str, Any], yaxis_type: str, out_path: str
) -> None:
    """Handles subplot logic for comparing multiple models side-by-side."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Validation Losses", "Training Losses")
    )
    baseline_val = None

    for model in models:
        data = all_data[model]
        fig.add_trace(
            go.Scatter(
                x=data["val_x"], y=data["val_y"], mode="lines", name=f"{model} Val Loss"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["train_x"],
                y=data["train_y"],
                mode="lines",
                name=f"{model} Train Loss",
            ),
            row=1,
            col=2,
        )

        if data["base_y"]:
            if baseline_val is None:
                baseline_val = data["base_y"][0]
            elif not np.isclose(
                baseline_val, data["base_y"][0], rtol=1e-3, atol=1e-5
            ) and not (np.isnan(baseline_val) and np.isnan(data["base_y"][0])):
                raise DataContinuityError(
                    f"Baseline validation loss is not constant. Expected {baseline_val}, got {data['base_y'][0]} in '{model}'"
                )

    if baseline_val is not None:
        max_x = max(
            [max(all_data[m]["train_x"]) for m in models if all_data[m]["train_x"]]
            + [0]
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_x],
                y=[baseline_val, baseline_val],
                mode="lines",
                name="Baseline Loss",
                line=dict(dash="dash"),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="Loss", type=yaxis_type, row=1, col=1)
    fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="Loss", type=yaxis_type, row=1, col=2)

    fig.update_layout(title_text="Multi-Model Training Visualization")
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"Visualization HTML generated and saved successfully to {out_path}")


def generate_html_report(
    all_data: Dict[str, Any], models: List[str], args: argparse.Namespace
) -> None:
    """Router function to generate the appropriate HTML report based on model count."""
    output_dir = os.path.join(args.project_root, "outputs", "visualization")
    os.makedirs(output_dir, exist_ok=True)

    yaxis_type = "log" if getattr(args, "log_scale", False) else "linear"

    if len(models) == 1:
        model = models[0]
        out_path = os.path.join(output_dir, f"{model}-training-visualization.html")
        _generate_single_model_plot(model, all_data[model], yaxis_type, out_path)
    else:
        out_path = os.path.join(output_dir, "multi-model-training-visualization.html")
        _generate_multi_model_plot(models, all_data, yaxis_type, out_path)


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------
def visualize_training(args: argparse.Namespace) -> None:
    """Main orchestrator function."""
    models = parse_args_to_models(args)
    if not models:
        raise ValueError("No models provided to visualize.")

    bucket_batches = getattr(args, "bucket_training_batches", None)
    all_data = {}

    for model in models:
        # Initialize Loguru for the current model.
        # This will wipe previous handlers and route output to the current model's logs.
        configure_logger(args.project_root, model, rank=0)

        logger.info(f"Parsing log file for model: {model}")
        log_file = get_log_filepath(args, model)

        # Instantiate parser and extract metrics
        parser = LogParser(model)
        metrics = parser.parse_file(log_file)

        formatted_data = format_plot_data(metrics, bucket_batches, model)
        all_data[model] = formatted_data

    # Note: For multi-model setups, the logger context at this stage
    # will belong to the *last* model processed in the loop.
    logger.info("Generating HTML visualizations...")
    generate_html_report(all_data, models, args)
