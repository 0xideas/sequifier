import glob
import os
import re

import numpy as np
import plotly.graph_objects as go


def parse_number(val: str) -> float:
    """Strictly parse numbers, explicitly handling the 'NaN' strings from format_number."""
    val = val.strip()
    if val == "NaN":
        return np.nan
    return float(val)


def visualize_training(args):
    # 1. Parse Input Argument strictly
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        models = [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not models:
        raise ValueError("CRITICAL: No models provided to visualize.")

    # 2. Extract logs per model
    all_data = {}
    for model in models:
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

        # Use the first matched log file
        log_file = log_files[0]

        val_losses = {}
        baseline_losses = {}
        var_losses = {}
        train_losses = {}

        expect_var_losses = None
        current_epoch = None
        current_batch = None
        expected_num_batches = None

        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                # STRICT MATCH: Validation
                if "[INFO] Validation | Epoch:" in line:
                    val_match = re.search(
                        r"\[INFO\] Validation\s+\|\s*Epoch:\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)\s+\|\s*Baseline Loss:\s*([^\s\|]+)",
                        line,
                    )
                    if not val_match:
                        raise ValueError(
                            f"CRITICAL [Line {line_num}]: Malformed Validation log -> '{line.strip()}'"
                        )

                    epoch = int(val_match.group(1))
                    val_loss = parse_number(val_match.group(2))
                    baseline = parse_number(val_match.group(3))

                    # A new sequence run resets at Validation 0. Clear everything prior.
                    if epoch == 0 or (
                        current_epoch is not None and epoch < current_epoch
                    ):
                        val_losses.clear()
                        baseline_losses.clear()
                        var_losses.clear()
                        train_losses.clear()
                        current_epoch = None
                        current_batch = None
                        expected_num_batches = None

                    val_losses[epoch] = val_loss
                    baseline_losses[epoch] = baseline
                    expect_var_losses = epoch
                    continue

                # STRICT MATCH: Variable Losses (Expected immediately after validation if targets > 1)
                if expect_var_losses is not None:
                    if "[INFO]  - " in line:
                        var_match = re.search(r"\[INFO\]\s+-\s+(.*)", line)
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
                            var_losses.setdefault(var_name, {})[expect_var_losses] = (
                                v_loss
                            )

                        expect_var_losses = None
                        continue
                    elif "[INFO] Epoch" in line or "[INFO] Validation" in line:
                        # Found structural logs meaning no variable losses existed for this validation step
                        expect_var_losses = None
                    else:
                        # Non-structural logs (e.g., class shares) can pass, keep expecting var losses if relevant
                        pass

                # STRICT MATCH: Training Batches
                if "[INFO] Epoch" in line and "| Batch" in line:
                    train_match = re.search(
                        r"\[INFO\] Epoch\s*(\d+)\s+\|\s*Batch\s*(\d+)/\s*(\d+)\s+\|\s*Loss:\s*([^\s\|]+)",
                        line,
                    )
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
                            pass  # Valid transition
                        elif epoch == 1 and batch <= num_batches:
                            pass  # Valid restart
                        else:
                            raise ValueError(
                                f"CRITICAL [Line {line_num}]: Epoch transition violated (was {current_epoch}, now {epoch})."
                            )

                    if (
                        expected_num_batches is not None
                        and num_batches != expected_num_batches
                    ):
                        # The very last batch could theoretically change if datasets change dynamically, but standard sequifier shouldn't
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
                            # Keep only Epoch 0 data
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

        # 3. Validate extracted data (Fail immediately if missing)
        if not train_losses:
            raise ValueError(f"CRITICAL [{model}]: No valid training loss data found.")
        if not val_losses:
            raise ValueError(
                f"CRITICAL [{model}]: No valid validation loss data found."
            )
        if not baseline_losses:
            raise ValueError(f"CRITICAL [{model}]: No baseline loss data found.")

        # Format points
        val_x = sorted(list(val_losses.keys()))
        val_y = [val_losses[e] for e in val_x]
        base_y = [baseline_losses[e] for e in val_x]

        train_x = []
        train_y = []
        bucket_batches = getattr(args, "bucket_training_batches", None)

        for epoch in sorted(list(train_losses.keys())):
            epoch_dict = train_losses[epoch]
            if not epoch_dict:
                continue

            # Extract perfectly chronological lists
            epoch_data = [
                (b, epoch_dict[b][0], epoch_dict[b][1])
                for b in sorted(epoch_dict.keys())
            ]

            if bucket_batches is not None:
                if len(epoch_data) > 1:
                    log_interval = epoch_data[1][0] - epoch_data[0][0]
                else:
                    log_interval = epoch_data[0][0]

                if log_interval == 0:
                    log_interval = 1

                # Strict Bucketing Check
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

        all_data[model] = {
            "val_x": val_x,
            "val_y": val_y,
            "base_y": base_y,
            "train_x": train_x,
            "train_y": train_y,
            "var_losses": var_losses,
        }

    os.makedirs("outputs", exist_ok=True)
    fig1 = go.Figure()
    fig2 = go.Figure()

    os.makedirs(
        os.path.join(args.project_root, "outputs", "visualization"), exist_ok=True
    )

    # 4. Create Plots based on input cardinality
    if len(models) == 1:
        model = models[0]
        data = all_data[model]

        fig1.add_trace(
            go.Scatter(
                x=data["val_x"], y=data["val_y"], mode="lines", name="Validation Loss"
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=data["train_x"], y=data["train_y"], mode="lines", name="Training Loss"
            )
        )
        if data["base_y"]:
            fig1.add_trace(
                go.Scatter(
                    x=data["val_x"],
                    y=data["base_y"],
                    mode="lines",
                    name="Baseline Loss",
                    line=dict(dash="dash"),
                )
            )
        fig1.update_layout(
            title=f"Global Losses: {model}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            xaxis=dict(dtick=1),
        )

        if getattr(args, "log_scale", False):
            fig1.update_yaxes(type="log")

        if "var_losses" in data and data["var_losses"]:
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
                fig2.add_trace(go.Scatter(x=epochs, y=y_norm, mode="lines", name=var))
            fig2.update_layout(
                title="Normalized Variable Validation Losses",
                xaxis_title="Epoch",
                yaxis_title="Loss / Epoch 0 Loss",
                xaxis=dict(dtick=1),
            )

            if getattr(args, "log_scale", False):
                fig2.update_yaxes(type="log")

        out_path = os.path.join(
            args.project_root,
            "outputs",
            "visualization",
            f"{model}-training-visualization.html",
        )

    else:
        baseline_val = None
        for model in models:
            data = all_data[model]
            fig1.add_trace(
                go.Scatter(
                    x=data["val_x"],
                    y=data["val_y"],
                    mode="lines",
                    name=f"{model} Val Loss",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=data["train_x"],
                    y=data["train_y"],
                    mode="lines",
                    name=f"{model} Train Loss",
                )
            )

            if data["base_y"]:
                if baseline_val is None:
                    baseline_val = data["base_y"][0]
                else:
                    # Strict validation across multiple models
                    if not np.isclose(
                        baseline_val, data["base_y"][0], rtol=1e-3, atol=1e-5
                    ) and not (np.isnan(baseline_val) and np.isnan(data["base_y"][0])):
                        raise ValueError(
                            f"CRITICAL: Baseline validation loss is not constant across models. Expected {baseline_val}, got {data['base_y'][0]} in model '{model}'"
                        )

        if baseline_val is not None:
            max_x = max(
                [max(all_data[m]["train_x"]) for m in models if all_data[m]["train_x"]]
                + [0]
            )
            fig2.add_trace(
                go.Scatter(
                    x=[0, max_x],
                    y=[baseline_val, baseline_val],
                    mode="lines",
                    name="Baseline Loss",
                    line=dict(dash="dash"),
                )
            )

        fig1.update_layout(
            title="Validation Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            xaxis=dict(dtick=1),
        )
        fig2.update_layout(
            title="Training Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            xaxis=dict(dtick=1),
        )

        if getattr(args, "log_scale", False):
            fig1.update_yaxes(type="log")
            fig2.update_yaxes(type="log")

        out_path = os.path.join(
            args.project_root,
            "outputs",
            "visualization",
            "multi-model-training-visualization.html",
        )

    # 5. Generate Responsive HTML using inline CSS flexbox constraints
    html1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")
    html2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Visualization</title>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0; overflow-x: hidden; }}
        .container {{ display: flex; flex-wrap: wrap; width: 100%; }}
        .plot {{ flex: 1 1 50%; min-width: 400px; max-width: 50%; box-sizing: border-box; padding: 10px; }}
        @media (max-width: 800px) {{ .plot {{ flex: 1 1 100%; max-width: 100%; }} }}
    </style>
</head>
<body>
    <div class="container"><div class="plot">{html1}</div><div class="plot">{html2}</div></div>

    <script>
        window.addEventListener('load', function() {{
            window.dispatchEvent(new Event('resize'));
        }});
    </script>
</body>
</html>"""
    with open(out_path, "w") as f:
        f.write(html_template)
    print(f"Visualization HTML generated and saved successfully to {out_path}")
