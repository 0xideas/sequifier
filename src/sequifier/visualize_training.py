import glob
import os
import re
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------------------
# Pre-compile Regular Expressions for Performance
# -------------------------------------------------------------------------
VAL_PATTERN = re.compile(
    r"\[INFO\] Validation\s+\|\s*Epoch:\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)\s+\|\s*Baseline Loss:\s*([^\s\|]+)"
)
VAR_PATTERN = re.compile(r"\[INFO\]\s+-\s+(.*)")
TRAIN_PATTERN = re.compile(
    r"\[INFO\] Epoch\s*(\d+)\s+\|\s*Batch\s*(\d+)/\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)"
)


def parse_number(val: str) -> float:
    """Strictly parse numbers, explicitly handling the 'NaN' strings from format_number."""
    val = val.strip()
    if val == "NaN":
        return np.nan
    return float(val)


def parse_args_to_models(args) -> list:
    """Extracts the list of models from a file or comma-separated string."""
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        return [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]
    else:
        return [m.strip() for m in args.models.split(",") if m.strip()]


def get_log_filepath(args, model: str) -> str:
    """Finds the appropriate log file for a given model (retaining hardcoded logic)."""
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
            f"CRITICAL: No log files found for model '{model}' matching the expected pattern."
        )

    return log_files[0]


def parse_log_file(log_file: str, model: str) -> tuple:
    """Reads the log file line-by-line and extracts training/validation metrics."""
    val_losses = {}
    baseline_losses = {}
    var_losses = {}
    train_losses = {}

    pending_var_loss_epoch = None
    current_epoch = None
    current_batch = None
    expected_num_batches = None

    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            # STRICT MATCH: Validation
            if "[INFO] Validation | Epoch:" in line:
                val_match = VAL_PATTERN.search(line)
                if not val_match:
                    raise ValueError(
                        f"CRITICAL [Line {line_num}]: Malformed Validation log -> '{line.strip()}'"
                    )

                epoch = int(val_match.group(1))
                val_loss = parse_number(val_match.group(2))
                baseline = parse_number(val_match.group(3))

                # A new sequence run resets at Validation 0. Clear everything prior.
                if epoch == 0 or (current_epoch is not None and epoch < current_epoch):
                    val_losses.clear()
                    baseline_losses.clear()
                    var_losses.clear()
                    train_losses.clear()
                    current_epoch = None
                    current_batch = None
                    expected_num_batches = None

                val_losses[epoch] = val_loss
                baseline_losses[epoch] = baseline
                pending_var_loss_epoch = epoch
                continue

            # STRICT MATCH: Variable Losses
            if pending_var_loss_epoch is not None:
                if "[INFO]  - " in line:
                    var_match = VAR_PATTERN.search(line)
                    if not var_match:
                        raise ValueError(
                            f"CRITICAL [Line {line_num}]: Malformed Variable Loss log -> '{line.strip()}'"
                        )

                    content = var_match.group(1)
                    parts = content.split(",")
                    for p in parts:
                        if ":" not in p:
                            raise ValueError(
                                f"CRITICAL [Line {line_num}]: Missing ':' in variable loss pair -> '{p.strip()}'"
                            )

                        var_name, v_loss = p.split(":", 1)
                        var_name = var_name.strip().replace("_loss", "")
                        v_loss = parse_number(v_loss)
                        var_losses.setdefault(var_name, {})[pending_var_loss_epoch] = (
                            v_loss
                        )

                    pending_var_loss_epoch = None
                    continue
                elif "[INFO] Epoch" in line or "[INFO] Validation" in line:
                    pending_var_loss_epoch = None
                else:
                    pass

            # STRICT MATCH: Training Batches
            if "[INFO] Epoch" in line and "| Batch" in line:
                train_match = TRAIN_PATTERN.search(line)
                if not train_match:
                    raise ValueError(
                        f"CRITICAL [Line {line_num}]: Malformed Training Batch log -> '{line.strip()}'"
                    )

                epoch = int(train_match.group(1))
                batch = int(train_match.group(2))
                num_batches = int(train_match.group(3))
                loss = parse_number(train_match.group(4))

                # Strict Chronological Validation
                if current_epoch is not None and current_batch is not None:
                    if epoch == current_epoch:
                        if batch <= current_batch:
                            raise ValueError(
                                f"CRITICAL [Line {line_num}]: Batch monotonicity violated (was {current_batch}, now {batch})."
                            )
                    elif epoch == current_epoch + 1:
                        pass
                    elif epoch == 1 and batch <= num_batches:
                        pass
                    else:
                        raise ValueError(
                            f"CRITICAL [Line {line_num}]: Epoch transition violated (was {current_epoch}, now {epoch})."
                        )

                if (
                    expected_num_batches is not None
                    and num_batches != expected_num_batches
                ):
                    if epoch == current_epoch:
                        raise ValueError(
                            f"CRITICAL [Line {line_num}]: Inconsistent num_batches mid-epoch (was {expected_num_batches}, now {num_batches})."
                        )

                current_epoch = epoch
                current_batch = batch
                expected_num_batches = num_batches

                # Restart on Epoch 1 if skipped Epoch 0
                if epoch == 1 and batch == 1 and 0 not in train_losses:
                    if 0 not in val_losses:
                        val_losses.clear()
                        baseline_losses.clear()
                        var_losses.clear()
                        train_losses.clear()
                    else:
                        train_losses.clear()
                        val_losses = {0: val_losses[0]} if 0 in val_losses else {}
                        baseline_losses = (
                            {0: baseline_losses[0]} if 0 in baseline_losses else {}
                        )
                        for v_name in var_losses:
                            var_losses[v_name] = (
                                {0: var_losses[v_name][0]}
                                if 0 in var_losses[v_name]
                                else {}
                            )

                if epoch not in train_losses:
                    train_losses[epoch] = {}

                if batch in train_losses[epoch]:
                    raise ValueError(
                        f"CRITICAL [Line {line_num}]: Duplicate batch {batch} recorded for Epoch {epoch}."
                    )

                train_losses[epoch][batch] = (num_batches, loss)

    # Validate extracted data
    if not train_losses:
        raise ValueError(f"CRITICAL [{model}]: No valid training loss data found.")
    if not val_losses:
        raise ValueError(f"CRITICAL [{model}]: No valid validation loss data found.")
    if not baseline_losses:
        raise ValueError(f"CRITICAL [{model}]: No baseline loss data found.")

    return val_losses, baseline_losses, var_losses, train_losses


def format_plot_data(
    val_losses: dict,
    baseline_losses: dict,
    var_losses: dict,
    train_losses: dict,
    bucket_batches: Optional[int],
    model: str,
) -> dict:
    """Formats raw parsed dictionaries into chronological arrays for Plotly."""
    val_x = sorted(list(val_losses.keys()))
    val_y = [val_losses[e] for e in val_x]
    base_y = [baseline_losses[e] for e in val_x]

    train_x = []
    train_y = []

    for epoch in sorted(list(train_losses.keys())):
        epoch_dict = train_losses[epoch]
        if not epoch_dict:
            continue

        epoch_data = [
            (b, epoch_dict[b][0], epoch_dict[b][1]) for b in sorted(epoch_dict.keys())
        ]

        if bucket_batches is not None:
            if len(epoch_data) > 1:
                log_interval = epoch_data[1][0] - epoch_data[0][0]
            else:
                log_interval = epoch_data[0][0]

            if log_interval == 0:
                log_interval = 1

            if bucket_batches % log_interval != 0:
                raise ValueError(
                    f"CRITICAL [{model} Epoch {epoch}]: --bucket-training-batches ({bucket_batches}) MUST be an exact multiple of the logged batch interval ({log_interval})."
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
        raise ValueError(
            f"CRITICAL [{model}]: Training arrays ended up empty after formatting."
        )

    return {
        "val_x": val_x,
        "val_y": val_y,
        "base_y": base_y,
        "train_x": train_x,
        "train_y": train_y,
        "var_losses": var_losses,
    }


def generate_html_report(all_data: dict, models: list, args) -> None:
    """Generates the native Plotly subplots and exports them as a single HTML file."""
    os.makedirs(
        os.path.join(args.project_root, "outputs", "visualization"), exist_ok=True
    )
    log_scale = getattr(args, "log_scale", False)
    yaxis_type = "log" if log_scale else "linear"

    # Single Model Subplots
    if len(models) == 1:
        model = models[0]
        data = all_data[model]

        has_var_losses = "var_losses" in data and data["var_losses"]
        subplot_titles = (
            ("Global Losses", "Normalized Variable Validation Losses")
            if has_var_losses
            else ("Global Losses", "")
        )

        fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles)

        # Plot 1: Global Losses
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

        # Plot 2: Variable Losses
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
            print(
                f"Warning: No variable validation losses found for model '{model}'. Second subplot will be empty."
            )

        fig.update_layout(title_text=f"Training Visualization: {model}")
        out_path = os.path.join(
            args.project_root,
            "outputs",
            "visualization",
            f"{model}-training-visualization.html",
        )

    # Multi-Model Subplots
    else:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Validation Losses", "Training Losses")
        )
        baseline_val = None

        for model in models:
            data = all_data[model]

            # Plot 1: Multi Validation
            fig.add_trace(
                go.Scatter(
                    x=data["val_x"],
                    y=data["val_y"],
                    mode="lines",
                    name=f"{model} Val Loss",
                ),
                row=1,
                col=1,
            )
            # Plot 2: Multi Training
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

            # Baseline Consistency Check
            if data["base_y"]:
                if baseline_val is None:
                    baseline_val = data["base_y"][0]
                else:
                    if not np.isclose(
                        baseline_val, data["base_y"][0], rtol=1e-3, atol=1e-5
                    ) and not (np.isnan(baseline_val) and np.isnan(data["base_y"][0])):
                        raise ValueError(
                            f"CRITICAL: Baseline validation loss is not constant across models. Expected {baseline_val}, got {data['base_y'][0]} in model '{model}'"
                        )

        # Draw Global Baseline on the training graph if available
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
        out_path = os.path.join(
            args.project_root,
            "outputs",
            "visualization",
            "multi-model-training-visualization.html",
        )

    # Native Plotly Export (removes custom HTML layout dependency)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Visualization HTML generated and saved successfully to {out_path}")


def visualize_training(args):
    """Main orchestrator function."""
    models = parse_args_to_models(args)
    if not models:
        raise ValueError("CRITICAL: No models provided to visualize.")

    bucket_batches = getattr(args, "bucket_training_batches", None)
    all_data = {}

    for model in models:
        log_file = get_log_filepath(args, model)
        val_losses, baseline_losses, var_losses, train_losses = parse_log_file(
            log_file, model
        )
        formatted_data = format_plot_data(
            val_losses, baseline_losses, var_losses, train_losses, bucket_batches, model
        )

        all_data[model] = formatted_data

    generate_html_report(all_data, models, args)
