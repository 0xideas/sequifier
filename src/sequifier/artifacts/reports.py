"""Common, versioned discovery records for every training runner."""

from pathlib import Path

from sequifier.io.atomic import write_json_atomic


def phase_report(*, index, name, algorithm, model_path, checkpoint_path, **details):
    return {
        "artifact_type": "sequifier_phase_report",
        "format_version": 1,
        "index": index,
        "name": name,
        "algorithm": algorithm,
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "monitor": None,
        "direction": None,
        "handoff": "last",
        "selected_metric": None,
        "selected_transaction": None,
        "selected_weight_hash": None,
        "clocks": {},
        "metrics": [],
        "provenance": {},
        "stop_scope": None,
        "completion_reason": "completed",
        **details,
    }


def run_report(*, model_path, checkpoint_path, phases, **details):
    return {
        "artifact_type": "sequifier_run_manifest",
        "format_version": 3,
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "phases": list(phases),
        "run_id": None,
        "session_id": None,
        "status": "complete",
        "completion_reason": "completed",
        "exports": [],
        "source_provenance": {},
        "initialization": {},
        "resume": {},
        **details,
    }


def publish_run_report(report, path):
    """Keep the supervised manifest discovery location on every route."""
    destination = Path(path).resolve()
    manifest = Path(report["checkpoint_path"]).resolve().parent / "manifest.json"
    write_json_atomic(destination, report)
    if manifest != destination:
        write_json_atomic(manifest, report)
