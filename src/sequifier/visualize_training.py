import argparse
import csv
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
from sequifier.training.metrics import TOTAL_TARGET


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


class StructuredMetricsParser:
    """Read rank-0 training and validation tables for the latest logical run."""

    def __init__(self, model_name: str):
        self.model = model_name

    @beartype
    def parse_files(self, training_file: str, validation_file: str) -> TrainingMetrics:
        training_rows = self._read_rows(training_file)
        validation_rows = self._read_rows(validation_file)
        total_training_rows = [
            row
            for row in training_rows
            if row.get("metric") == "loss" and row.get("target") == TOTAL_TARGET
        ]
        if not total_training_rows:
            raise DataContinuityError(
                f"[{self.model}]: No valid training loss data found."
            )
        run_id = total_training_rows[-1]["run_id"]
        metrics = TrainingMetrics()

        for row in total_training_rows:
            if row.get("run_id") != run_id:
                continue
            epoch = int(row["epoch"])
            batch = int(row["batch"])
            batches_total = int(row["batches_total"])
            metrics.train_losses.setdefault(epoch, {})[batch] = (
                batches_total,
                float(row["value"]),
            )

        for row in validation_rows:
            if row.get("run_id") != run_id:
                continue
            epoch_position = self._epoch_position(row)
            metric = row.get("metric")
            target = row.get("target")
            value = float(row["value"])
            if metric == "loss" and target == TOTAL_TARGET:
                metrics.val_losses[epoch_position] = value
            elif metric == "baseline_loss" and target == TOTAL_TARGET:
                metrics.baseline_losses[epoch_position] = value
            elif metric == "loss" and target:
                metrics.var_losses.setdefault(target, {})[epoch_position] = value

        if not metrics.val_losses:
            raise DataContinuityError(
                f"[{self.model}]: No valid validation loss data found."
            )
        if not metrics.baseline_losses:
            raise DataContinuityError(f"[{self.model}]: No baseline loss data found.")
        return metrics

    @staticmethod
    def _read_rows(path: str) -> list[dict[str, str]]:
        with open(path, "r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    @staticmethod
    def _epoch_position(row: dict[str, str]) -> float:
        epoch = int(row["epoch"])
        batch = int(row["batch"])
        batches_total = int(row["batches_total"])
        if row.get("evaluation_kind") == "initial":
            return 0.0
        if batch > 0 and batches_total > 0:
            return round(epoch - 1 + batch / batches_total, 8)
        return float(epoch)


@beartype
def parse_args_to_models(args: argparse.Namespace) -> list[str]:
    """Read model names from a file or comma-separated argument."""
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        return [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]

    return [m.strip() for m in args.models.split(",") if m.strip()]


@beartype
def get_metrics_filepaths(args: argparse.Namespace, model: str) -> tuple[str, str]:
    """Return the rank-0 training and validation metric paths for a model."""
    log_dir = os.path.join(args.project_root, "logs")
    training_file = os.path.join(log_dir, f"sequifier-{model}-rank0-training.csv")
    validation_file = os.path.join(log_dir, f"sequifier-{model}-rank0-validation.csv")
    missing = [
        path for path in (training_file, validation_file) if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Structured metric files not found for model {model!r}: {missing!r}"
        )
    return training_file, validation_file


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
    """Read structured metrics and write training visualization HTML."""
    models = parse_args_to_models(args)
    if not models:
        raise ValueError("No models provided to visualize.")

    bucket_batches = getattr(args, "bucket_training_batches", None)
    all_data = {}

    for model in models:
        # Route visualization events to the current model's operational logs.
        configure_logger(args.project_root, model, rank=0)

        logger.info(f"Parsing structured metrics for model: {model}")
        training_file, validation_file = get_metrics_filepaths(args, model)

        parser = StructuredMetricsParser(model)
        metrics = parser.parse_files(training_file, validation_file)

        formatted_data = format_plot_data(metrics, bucket_batches, model)
        all_data[model] = formatted_data

    # Note: For multi-model setups, the logger context at this stage
    # will belong to the *last* model processed in the loop.
    logger.info("Generating HTML visualizations...")
    generate_html_report(all_data, models, args)
