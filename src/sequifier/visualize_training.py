import argparse
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from beartype import beartype
from loguru import logger
from plotly.subplots import make_subplots

from sequifier.helpers import configure_logger

VAL_PATTERN = re.compile(
    r"\[INFO\] Validation\s+\|\s*Epoch:\s*(\d+)\s+\|\s*Batch:\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)\s+\|\s*Baseline Loss:\s*([^\s\|]+)"
)
VAR_PATTERN = re.compile(r"\[INFO\]\s+-\s+(.*)")
TRAIN_PATTERN = re.compile(
    r"\[INFO\] Epoch\s*(\d+)\s+\|\s*Batch\s*(\d+)/\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)"
)


class LogParsingError(Exception):
    """Malformed training log line."""

    pass


class DataContinuityError(Exception):
    """Non-monotonic training batch or epoch sequence."""

    pass


@dataclass
class TrainingMetrics:
    """Parsed validation, baseline, variable, and training losses."""

    val_losses: dict[float, float] = field(default_factory=dict)
    baseline_losses: dict[float, float] = field(default_factory=dict)
    var_losses: dict[str, dict[Optional[float], float]] = field(default_factory=dict)
    train_losses: dict[float, dict[int, tuple[int, float]]] = field(
        default_factory=dict
    )

    def clear_state(self) -> None:
        """Clear parsed metrics after a detected run restart."""
        self.val_losses.clear()
        self.baseline_losses.clear()
        self.var_losses.clear()
        self.train_losses.clear()


class LogParser:
    """Stateful parser for sequifier training logs."""

    def __init__(self, model_name: str):
        self.model = model_name
        self.metrics = TrainingMetrics()
        self.current_epoch: Optional[int] = None
        self.current_batch: Optional[int] = None
        self.expected_num_batches: Optional[int] = None
        self.pending_var_loss_epoch: Optional[float] = None

    @beartype
    def parse_file(self, log_file: str) -> TrainingMetrics:
        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    self._process_line(line)
                except Exception as e:
                    raise LogParsingError(f"[{self.model} Line {line_num}]: {e}")

        self._validate_final_metrics()
        return self.metrics

    @beartype
    def _process_line(self, line: str) -> None:
        """Dispatch one log line to the matching parser."""
        if "[INFO] Validation | Epoch:" in line:
            self._process_validation(line)
        elif self.pending_var_loss_epoch is not None and "[INFO]  - " in line:
            self._process_var_loss(line)
        elif "[INFO] Epoch" in line and "| Batch" in line:
            self._process_training(line)
        elif "[INFO] Epoch" in line or "[INFO] Validation" in line:
            self.pending_var_loss_epoch = None

    @beartype
    def _process_validation(self, line: str) -> None:
        match = VAL_PATTERN.search(line)
        if not match:
            raise LogParsingError(f"Malformed Validation log -> '{line.strip()}'")

        epoch = int(match.group(1))
        batch = int(match.group(2))
        val_loss = parse_number(match.group(3))
        baseline = parse_number(match.group(4))

        if (epoch == 0 and batch == 0) or (
            self.current_epoch is not None and epoch < self.current_epoch and epoch != 0
        ):
            self.metrics.clear_state()
            self.current_epoch = None
            self.current_batch = None
            self.expected_num_batches = None

        if (
            epoch == 0
            and batch > 0
            and self.current_epoch is not None
            and self.expected_num_batches is not None
        ):
            calc_epoch = self.current_epoch - 1 + (batch / self.expected_num_batches)
        elif self.expected_num_batches is not None and batch > 0:
            calc_epoch = epoch - 1 + (batch / self.expected_num_batches)
        else:
            calc_epoch = float(epoch)

        self.metrics.val_losses[calc_epoch] = val_loss
        self.metrics.baseline_losses[calc_epoch] = baseline
        self.pending_var_loss_epoch = calc_epoch

    @beartype
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

    @beartype
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

    @beartype
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
        """Handle restarts that resume at epoch 1 without epoch 0 logs."""
        if 0.0 not in self.metrics.val_losses:
            self.metrics.clear_state()
        else:
            self.metrics.train_losses.clear()
            self.metrics.val_losses = {0.0: self.metrics.val_losses[0.0]}
            self.metrics.baseline_losses = {0.0: self.metrics.baseline_losses[0.0]}
            for v_name in list(self.metrics.var_losses.keys()):
                if 0.0 in self.metrics.var_losses[v_name]:
                    self.metrics.var_losses[v_name] = {
                        0.0: self.metrics.var_losses[v_name][0.0]
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


@beartype
def parse_number(val: str) -> float:
    """Parse finite floats and literal NaN."""
    val = val.strip()
    return np.nan if val == "NaN" else float(val)


@beartype
def parse_args_to_models(args: argparse.Namespace) -> list[str]:
    """Read model names from a file or comma-separated argument."""
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        return [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]

    return [m.strip() for m in args.models.split(",") if m.strip()]


@beartype
def get_log_filepath(args: argparse.Namespace, model: str) -> str:
    """Return the rank-0 log path for a model."""
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


@beartype
def format_plot_data(
    metrics: TrainingMetrics, bucket_batches: Optional[int], model: str
) -> dict[str, Any]:
    """Convert parsed metrics into Plotly-ready arrays."""
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


@beartype
def _generate_single_model_plot(
    model: str, data: dict[str, Any], yaxis_type: str, out_path: str
) -> None:
    """Write a single-model training report."""
    has_var_losses = bool(data.get("var_losses"))
    subplot_titles = (
        ("Global Losses", "Normalized Variable Validation Losses")
        if has_var_losses
        else ("Global Losses", "")
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

    fig.add_trace(
        go.Scatter(
            x=data["val_x"],
            y=data["val_y"],
            mode="lines",
            name="Validation Loss",
            hovertemplate=f"<b>{model}</b><br>Val Loss: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["train_x"],
            y=data["train_y"],
            mode="lines",
            name="Training Loss",
            hovertemplate=f"<b>{model}</b><br>Train Loss: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
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
                hovertemplate=f"<b>{model}</b><br>Baseline Loss: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
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
                go.Scatter(
                    x=epochs,
                    y=y_norm,
                    mode="lines",
                    name=var,
                    hovertemplate=f"<b>{var}</b>: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
                ),
                row=1,
                col=2,
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


@beartype
def _generate_multi_model_plot(
    models: list[str], all_data: dict[str, Any], yaxis_type: str, out_path: str
) -> None:
    """Write a multi-model training report."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Validation Losses", "Training Losses")
    )
    baseline_val = None
    colors = pc.qualitative.Plotly

    for i, model in enumerate(models):
        data = all_data[model]
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=data["val_x"],
                y=data["val_y"],
                mode="lines",
                name=model,
                legendgroup=model,
                line=dict(color=color),
                showlegend=True,
                hovertemplate=f"<b>{model}</b><br>Val Loss: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["train_x"],
                y=data["train_y"],
                mode="lines",
                name=model,
                legendgroup=model,
                line=dict(color=color),
                showlegend=False,
                hovertemplate=f"<b>{model}</b><br>Train Loss: %{{y}}<br>Epoch: %{{x}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        if data["base_y"]:
            SKIP_BASELINE_CHECK = os.getenv("SKIP_BASELINE_CHECK")
            if baseline_val is None:
                baseline_val = data["base_y"][0]
            elif (
                SKIP_BASELINE_CHECK is None
                and not np.isclose(baseline_val, data["base_y"][0], atol=1e-2)
                and not (np.isnan(baseline_val) and np.isnan(data["base_y"][0]))
                and os.getenv("SEQUIFIER_SKIP_BASELINE_CHECK") is None
            ):
                raise DataContinuityError(
                    f"Baseline validation loss is not constant. Expected {baseline_val}, got {data['base_y'][0]} in '{model}'"
                )

    if baseline_val is not None:
        max_val_x = max(
            [max(all_data[m]["val_x"]) for m in models if all_data[m]["val_x"]] + [0]
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val_x],
                y=[baseline_val, baseline_val],
                mode="lines",
                name="Baseline Loss",
                line=dict(dash="dash", color="black"),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="Loss", type=yaxis_type, row=1, col=1)
    fig.update_xaxes(title_text="Epoch", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="Loss", type=yaxis_type, row=1, col=2)

    fig.update_layout(title_text="Multi-Model Training Visualization")
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"Visualization HTML generated and saved successfully to {out_path}")


@beartype
def generate_html_report(
    all_data: dict[str, Any], models: list[str], args: argparse.Namespace
) -> None:
    """Write the model-count-appropriate HTML report."""
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


@beartype
def visualize_training(args: argparse.Namespace) -> None:
    """Parse logs and write training visualization HTML."""
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
