"""Common, versioned discovery records for every training runner."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from sequifier.artifacts.state_dict import weight_fingerprint
from sequifier.contract_versions import ContractVersions
from sequifier.io.atomic import write_json_atomic


@dataclass(frozen=True)
class HandoffCandidate:
    metric: float | None = None
    transaction: int | None = None
    weights: dict[str, Tensor] | None = None

    def require_weights(self) -> dict[str, Tensor]:
        if self.weights is None:
            raise ValueError("Selected handoff has no model weights")
        return self.weights

    def select(
        self, phase, last: Callable[[], "HandoffCandidate"]
    ) -> "HandoffCandidate":
        """Select the best snapshot or capture the final weights on demand."""
        if phase.handoff == "last":
            return last()
        if self.metric is None or self.transaction is None:
            raise ValueError(f"{phase.name}: best handoff requires a finite evaluation")
        return self


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
        cls,
        index,
        phase,
        monitor,
        *,
        selection: HandoffCandidate,
        clocks,
        provenance,
        stop_scope,
        completion_reason=None,
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
            selected_metric=selection.metric,
            selected_transaction=selection.transaction,
            selected_weight_hash=weight_fingerprint(selection.require_weights()),
            clocks=clocks,
            provenance=provenance,
        )

    def report(self, model_path, checkpoint_path):
        return {
            **self.model_dump(mode="json"),
            "artifact_type": "sequifier_phase_report",
            "model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
            "format_version": ContractVersions.PHASE_REPORT,
        }


def run_report(
    outputs,
    checkpoint,
    identity,
    commits,
    *,
    resume,
    completion_reason,
    source_provenance,
    additional_exports=(),
    status="complete",
    **details,
):
    return {
        "artifact_type": "sequifier_run_manifest",
        "model_path": str(outputs.model_path),
        "checkpoint_path": str(checkpoint),
        "phases": [commit.report_result() for commit in commits],
        "run_id": identity.run_id,
        "session_id": identity.session_id,
        "status": status,
        "completion_reason": completion_reason,
        "exports": [
            *(str(path) for path in additional_exports),
            str(outputs.model_path),
        ],
        "source_provenance": source_provenance,
        "initialization": {},
        "resume": resume.model_dump(mode="json") if resume is not None else {},
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
