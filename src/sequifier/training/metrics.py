"""Append-only structured training and validation metric files."""

from __future__ import annotations

import csv
import os
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sequifier.logging_paths import dataset_artifact_prefix
from sequifier.typechecking import beartype

METRICS_SCHEMA_VERSION = 1
TOTAL_TARGET = "__total__"

COMMON_FIELDS = [
    "schema_version",
    "run_id",
    "session_id",
    "timestamp_utc",
    "model",
    "dataset",
    "part",
    "rank",
    "epoch",
    "batch",
    "batches_total",
    "global_step",
]

TRAINING_FIELDS = COMMON_FIELDS + [
    "window_batches",
    "metric",
    "target",
    "value",
    "learning_rate",
    "seconds_per_batch",
]

VALIDATION_FIELDS = COMMON_FIELDS + [
    "evaluation_id",
    "evaluation_kind",
    "metric",
    "target",
    "value",
    "learning_rate",
    "elapsed_seconds",
]

CLASS_SHARE_FIELDS = COMMON_FIELDS + [
    "evaluation_id",
    "evaluation_kind",
    "target",
    "class_id",
    "class_label",
    "count",
    "total_count",
    "share",
    "status",
]


@beartype
def _timestamp_utc() -> str:
    """Return an RFC 3339 UTC timestamp suitable for tabular interchange."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@beartype
def _metric_value(value: Any) -> float:
    """Convert scalar tensor/NumPy/Python values without display rounding."""
    item = getattr(value, "item", None)
    scalar: Any = item() if callable(item) else value
    return float(scalar)


class _CsvAppender:
    """Validate and append complete row groups to one CSV file."""

    @beartype
    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    @beartype
    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as file:
                existing = next(csv.reader(file), None)
            if existing != self.fields:
                raise ValueError(
                    f"Structured metric schema mismatch in {self.path}: "
                    f"expected {self.fields!r}, found {existing!r}."
                )
            return

        with self.path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=self.fields).writeheader()
            file.flush()

    @beartype
    def append(self, rows: Iterable[Mapping[str, Any]], *, durable: bool) -> None:
        row_group = list(rows)
        if not row_group:
            return
        with self.path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields, extrasaction="raise")
            writer.writerows(row_group)
            file.flush()
            if durable:
                os.fsync(file.fileno())


class StructuredMetricWriters:
    """Own the three rank-0 structured metric tables for one model."""

    @beartype
    def __init__(
        self,
        project_root: str,
        model_name: str,
        rank: int,
        *,
        class_share_columns: Iterable[str] = (),
        dataset_name: str | None = None,
        dataset_count: int = 1,
        validation_enabled: bool = True,
    ) -> None:
        prefix = dataset_artifact_prefix(
            project_root,
            model_name,
            dataset_name=dataset_name,
            dataset_count=dataset_count,
        )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rank = rank
        self.class_share_columns = tuple(class_share_columns)
        self.training_path = Path(f"{prefix}-training.csv")
        self.validation_path = Path(f"{prefix}-validation.csv")
        self.class_share_path = Path(f"{prefix}-validation-class-shares.csv")
        self._training: _CsvAppender | None = _CsvAppender(
            self.training_path, TRAINING_FIELDS
        )
        self._validation: _CsvAppender | None = (
            _CsvAppender(self.validation_path, VALIDATION_FIELDS)
            if validation_enabled
            else None
        )
        self.validation_enabled = validation_enabled
        self._class_shares: _CsvAppender | None = None

    @beartype
    def _common(
        self,
        *,
        run_id: str,
        session_id: str,
        timestamp: str,
        epoch: int,
        batch: int,
        batches_total: int,
        global_step: int,
        dataset: str | None = None,
        part: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": run_id,
            "session_id": session_id,
            "timestamp_utc": timestamp,
            "model": self.model_name,
            "dataset": dataset or self.dataset_name or "",
            "part": part or "",
            "rank": self.rank,
            "epoch": epoch,
            "batch": batch,
            "batches_total": batches_total,
            "global_step": global_step,
        }

    @beartype
    def write_training(
        self,
        *,
        run_id: str,
        session_id: str,
        epoch: int,
        batch: int,
        batches_total: int,
        global_step: int,
        window_batches: int,
        total_loss: Any,
        target_losses: Mapping[str, Any],
        learning_rate: float,
        seconds_per_batch: float,
        dataset: str | None = None,
        part: str | None = None,
    ) -> None:
        timestamp = _timestamp_utc()
        common = self._common(
            run_id=run_id,
            session_id=session_id,
            timestamp=timestamp,
            epoch=epoch,
            batch=batch,
            batches_total=batches_total,
            global_step=global_step,
            dataset=dataset,
            part=part,
        )
        measurements = [(TOTAL_TARGET, total_loss), *target_losses.items()]
        rows = [
            {
                **common,
                "window_batches": window_batches,
                "metric": "loss",
                "target": target,
                "value": _metric_value(value),
                "learning_rate": float(learning_rate),
                "seconds_per_batch": float(seconds_per_batch),
            }
            for target, value in measurements
        ]
        if self._training is None:
            self._training = _CsvAppender(self.training_path, TRAINING_FIELDS)
        self._training.append(rows, durable=False)

    @beartype
    def write_validation(
        self,
        *,
        run_id: str,
        session_id: str,
        evaluation_kind: str,
        epoch: int,
        batch: int,
        batches_total: int,
        global_step: int,
        total_loss: Any,
        target_losses: Mapping[str, Any],
        baseline_loss: Any,
        baseline_target_losses: Mapping[str, Any],
        class_distributions: Mapping[str, Iterable[Mapping[str, Any]]],
        learning_rate: float,
        elapsed_seconds: float,
        dataset: str | None = None,
        part: str | None = None,
    ) -> str:
        timestamp = _timestamp_utc()
        evaluation_id = uuid.uuid4().hex
        common = self._common(
            run_id=run_id,
            session_id=session_id,
            timestamp=timestamp,
            epoch=epoch,
            batch=batch,
            batches_total=batches_total,
            global_step=global_step,
            dataset=dataset,
            part=part,
        )
        validation_context = {
            **common,
            "evaluation_id": evaluation_id,
            "evaluation_kind": evaluation_kind,
            "learning_rate": float(learning_rate),
            "elapsed_seconds": float(elapsed_seconds),
        }
        measurements = [
            ("loss", TOTAL_TARGET, total_loss),
            ("baseline_loss", TOTAL_TARGET, baseline_loss),
            *(("loss", target, value) for target, value in target_losses.items()),
            *(
                ("baseline_loss", target, value)
                for target, value in baseline_target_losses.items()
            ),
        ]
        if not self.validation_enabled:
            raise ValueError("Validation metrics are disabled for this dataset")
        if self._validation is None:
            self._validation = _CsvAppender(self.validation_path, VALIDATION_FIELDS)
        self._validation.append(
            [
                {
                    **validation_context,
                    "metric": metric,
                    "target": target,
                    "value": _metric_value(value),
                }
                for metric, target, value in measurements
            ],
            durable=True,
        )

        class_rows: list[dict[str, Any]] = []
        class_context = {
            **common,
            "evaluation_id": evaluation_id,
            "evaluation_kind": evaluation_kind,
        }
        for target, distribution in class_distributions.items():
            target_rows = list(distribution)
            if not target_rows:
                class_rows.append(
                    {
                        **class_context,
                        "target": target,
                        "class_id": "",
                        "class_label": "",
                        "count": 0,
                        "total_count": 0,
                        "share": "",
                        "status": "no_valid_predictions",
                    }
                )
                continue
            for class_row in target_rows:
                class_rows.append(
                    {
                        **class_context,
                        "target": target,
                        "class_id": class_row["class_id"],
                        "class_label": class_row["class_label"],
                        "count": class_row["count"],
                        "total_count": class_row["total_count"],
                        "share": class_row["share"],
                        "status": "ok",
                    }
                )
        if class_rows and not self.class_share_columns:
            raise ValueError(
                "Class-share rows were provided without configured class-share "
                "columns."
            )
        if class_rows and self._class_shares is None:
            self._class_shares = _CsvAppender(self.class_share_path, CLASS_SHARE_FIELDS)
        if self._class_shares is not None:
            self._class_shares.append(class_rows, durable=True)
        return evaluation_id
