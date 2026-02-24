import glob
import os
import re

import numpy as np
import plotly.graph_objects as go


def visualize_training(args):
    # 1. Parse Input Argument
    if os.path.isfile(args.models) and args.models.endswith(".txt"):
        with open(args.models, "r") as f:
            content = f.read()
        models = [m.strip() for m in re.split(r"[\n,]", content) if m.strip()]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not models:
        print("No models provided.")
        return

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
                f"No log files found for model '{model}' matching the expected pattern."
            )

        # Use the first matched log file
        log_file = log_files[0]

        val_losses = {}
        baseline_losses = {}
        var_losses = {}
        train_losses = {}

        with open(log_file, "r") as f:
            lines = f.readlines()

        expect_var_losses = None

        for line in lines:
            val_match = re.search(
                r"Validation\s+\|\s*Epoch:\s*(\d+)\s+\|\s*Loss:\s*([\d\.e+-]+)\s+\|\s*Baseline Loss:\s*([\d\.e+-]+)",
                line,
            )
            if val_match:
                epoch = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                baseline = float(val_match.group(3))

                # A new sequence run resets at Validation 0. Clear everything prior.
                if epoch == 0:
                    val_losses.clear()
                    baseline_losses.clear()
                    var_losses.clear()
                    train_losses.clear()

                val_losses[epoch] = val_loss
                baseline_losses[epoch] = baseline
                expect_var_losses = epoch
                continue

            if expect_var_losses is not None:
                var_match = re.search(r"\[INFO\]\s+-\s+(.*)", line)
                if var_match:
                    content = var_match.group(1)
                    parts = content.split(",")
                    for p in parts:
                        if ":" in p:
                            var_name, v_loss = p.split(":")
                            var_name = var_name.strip().replace("_loss", "")
                            v_loss = float(v_loss.strip())
                            var_losses.setdefault(var_name, {})[expect_var_losses] = (
                                v_loss
                            )
                expect_var_losses = None
                continue

            train_match = re.search(
                r"Epoch\s*(\d+)\s+\|\s*Batch\s*(\d+)/\s*(\d+)\s+\|\s*Loss:\s*([\d\.e+-]+)",
                line,
            )
            if train_match:
                epoch = int(train_match.group(1))
                batch = int(train_match.group(2))
                num_batches = int(train_match.group(3))
                loss = float(train_match.group(4))

                # print(f"{epoch = }, {batch = }, {num_batches = }, {loss = }")

                # Restart on Epoch 1 if skipped Epoch 0
                if epoch == 1 and batch == 1:
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

                # Dictionary overwrites previous attempts if a mid-epoch crash occurred
                if epoch not in train_losses:
                    train_losses[epoch] = {}
                train_losses[epoch][batch] = (num_batches, loss)

        # 3. Validate extracted data
        if not train_losses:
            raise ValueError(
                f"Data Error in '{model}': No valid training loss data found after the last restart."
            )
        if not val_losses:
            raise ValueError(
                f"Data Error in '{model}': No valid validation loss data found after the last restart."
            )
        if not baseline_losses:
            raise ValueError(
                f"Data Error in '{model}': No baseline loss data found in the logs."
            )

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

            # Extract perfectly chronological lists from our batch dictionary
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
                if bucket_batches % log_interval != 0:
                    raise ValueError(
                        f"Model {model} Epoch {epoch}: --bucket-training-batches ({bucket_batches}) must be a multiple of the logged batch interval ({log_interval})."
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
                f"Data Error in '{model}': Training arrays are empty after formatting."
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
        else:
            print(
                f"Warning: No variable validation losses found for model '{model}'. Second plot will be empty."
            )

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
                    if not np.isclose(
                        baseline_val, data["base_y"][0], rtol=1e-3, atol=1e-5
                    ) and not (np.isnan(baseline_val) and np.isnan(data["base_y"][0])):
                        raise ValueError(
                            f"Baseline validation loss is not constant across models. Expected {baseline_val}, got {data['base_y'][0]} in model '{model}'"
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
        /* Added max-width to strictly constrain Plotly's initial canvas calculation */
        .plot {{ flex: 1 1 50%; min-width: 400px; max-width: 50%; box-sizing: border-box; padding: 10px; }}
        /* Reset max-width for mobile stacking */
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
