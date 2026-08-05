"""Pivot Sequifier's tidy validation metric table into one row per evaluation."""

import sys
from pathlib import Path

import pandas as pd


def process_metrics(file_path: str) -> Path:
    """Write a wide loss-only table next to a structured validation CSV."""
    source = Path(file_path)
    metrics = pd.read_csv(source)
    losses = metrics[metrics["metric"].isin(["loss", "baseline_loss"])].copy()
    losses["measurement"] = losses["metric"] + "." + losses["target"]

    context_columns = [
        "schema_version",
        "run_id",
        "session_id",
        "timestamp_utc",
        "model",
        "rank",
        "evaluation_id",
        "evaluation_kind",
        "epoch",
        "batch",
        "batches_total",
        "global_step",
        "learning_rate",
        "elapsed_seconds",
    ]
    table = losses.pivot(
        index=context_columns,
        columns="measurement",
        values="value",
    ).reset_index()
    table.columns.name = None

    output_path = source.with_name(f"{source.stem}-wide.csv")
    table.to_csv(output_path, index=False)
    print(f"Extracted {len(table)} validation evaluations to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_validation_losses_to_table.py <validation.csv>")
        raise SystemExit(2)
    process_metrics(sys.argv[1])
