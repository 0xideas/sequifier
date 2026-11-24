"""Export one row per evaluation from Sequifier's full validation metric CSV."""

import csv
import sys
from pathlib import Path


def process_metrics(file_path: str) -> Path:
    source = Path(file_path)
    evaluations = {}
    measurements = {}
    primary_fields = ["primary_metric", "primary_target", "primary_value"]
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"evaluation_id", "metric", "target", "value"}
        if missing := required.difference(fields):
            raise ValueError(
                f"Expected a full validation CSV. Missing {sorted(missing)}"
            )
        context = [
            key
            for key in fields
            if key not in {"metric", "target", "value", "is_primary"}
        ]
        for row in reader:
            identity = (row.get("run_id"), row["evaluation_id"])
            record = evaluations.setdefault(
                identity, {key: row[key] for key in context}
            )
            metric, target = row["metric"], row["target"]
            # Escape separators so a dotted metric or target is still unambiguous.
            column = ".".join(
                part.replace("\\", "\\\\").replace(".", "\\.")
                for part in (metric, target)
            )
            if column in record:
                raise ValueError(
                    f"Duplicate measurement {column!r} in evaluation {identity}"
                )
            record[column] = row["value"]
            measurements[column] = None
            primary = row.get("is_primary")
            is_primary = (
                (metric, target) == ("loss", "__total__")
                if primary is None
                else primary.lower() == "true"
            )
            if is_primary:
                record.update(
                    primary_metric=metric,
                    primary_target=target,
                    primary_value=row["value"],
                )

    output_path = source.with_name(f"{source.stem}-wide.csv")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[*context, *primary_fields, *measurements]
        )
        writer.writeheader()
        writer.writerows(evaluations.values())
    print(f"Extracted {len(evaluations)} validation evaluations to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python extract_validation_losses_to_table.py <model-validation-full.csv>"
        )
    process_metrics(sys.argv[1])
