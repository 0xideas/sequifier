"""Common, versioned discovery records for every training runner."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sequifier.contract_versions import ContractVersions
from sequifier.io.atomic import write_json_atomic


class PhaseResult(BaseModel):
    """Selected phase outcome shared by runners, checkpoints, and reports."""

    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    algorithm: str
    handoff: str
    monitor: str | None
    direction: str | None
    selected_metric: float | None
    selected_transaction: int | None
    selected_weight_hash: str
    clocks: dict
    provenance: dict
    metrics: tuple[dict, ...] = ()
    stop_scope: str | None
    completion_reason: str

    @classmethod
    def from_phase(
        cls, index, phase, monitor, *, stop_scope, completion_reason=None, **selection
    ):
        return cls(
            index=index,
            name=phase.name,
            algorithm=phase.algorithm,
            handoff=phase.handoff,
            monitor=monitor.metric if monitor else None,
            direction=monitor.direction if monitor else None,
            stop_scope=stop_scope,
            completion_reason=completion_reason
            or ("early_stopping" if stop_scope else "completed"),
            **selection,
        )

    def report(self, model_path, checkpoint_path):
        return {
            **self.model_dump(mode="json"),
            "artifact_type": "sequifier_phase_report",
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
            "format_version": ContractVersions.PHASE_REPORT,
        }


def run_report(*, model_path, checkpoint_path, phases, **details):
    return {
        "artifact_type": "sequifier_run_manifest",
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
        "format_version": ContractVersions.RUN_MANIFEST,
    }


def publish_run_report(report, path):
    """Keep the supervised manifest discovery location on every route."""
    destination = Path(path).resolve()
    manifest = Path(report["checkpoint_path"]).resolve().parent / "manifest.json"
    write_json_atomic(destination, report)
    if manifest != destination:
        write_json_atomic(manifest, report)
