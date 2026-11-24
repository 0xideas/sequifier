"""Common, versioned discovery records for every training runner."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch import Tensor

from sequifier.artifacts.state_dict import weight_fingerprint
from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import file_checksum
from sequifier.io.atomic import write_json_atomic
from sequifier.io.output_paths import run_manifest_path
from sequifier.runtime.plan import NORMAL_COMPLETION


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
            or ("early_stopping" if stop_scope else NORMAL_COMPLETION),
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


def completed_run_report(config, checkpoint, run_id, commits):
    """Reuse a completed publication only for its validated recovery boundary."""
    outputs = config.outputs
    path = Path(outputs.report_path)
    manifest = run_manifest_path(config)
    try:
        source = path if path.is_file() else manifest
        if not source.is_file():
            source = Path(checkpoint).resolve().parent / "manifest.json"
        content = source.read_bytes()
        report = json.loads(content)
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or (
        report.get("artifact_type") != "sequifier_run_manifest"
        or report.get("format_version") != ContractVersions.RUN_MANIFEST
        or report.get("status") != "complete"
        or report.get("run_id") != run_id
        or report.get("model_path") != str(outputs.model_path)
        or report.get("checkpoint_path") != str(checkpoint)
        or report.get("phases") != [commit.report_result() for commit in commits]
    ):
        return None
    checksums = report.get("export_checksums", {})
    exports = report.get("exports")
    if (
        not isinstance(exports, list)
        or not all(isinstance(item, str) for item in exports)
        or str(outputs.model_path) not in exports
        or not isinstance(checksums, dict)
        or (checksums and set(checksums) != set(exports))
    ):
        return None
    for export in exports:
        if not Path(export).is_file() or (
            export in checksums and file_checksum(export) != checksums[export]
        ):
            return None
    # A crash can separate the two atomic manifest publications. Repair the
    # discovery copy without issuing a new session or publishing the model.
    for destination in {path, manifest}:
        if destination != source and (
            not destination.is_file() or destination.read_bytes() != content
        ):
            write_json_atomic(destination, report)
    for commit in commits:
        commit.publish()
    return report


def publish_run_report(report, config):
    """Publish the configured report and the model-scoped discovery copy."""
    destination = Path(config.outputs.report_path).resolve()
    manifest = run_manifest_path(config)
    report = {
        **report,
        "export_checksums": {path: file_checksum(path) for path in report["exports"]},
    }
    write_json_atomic(destination, report)
    if manifest != destination:
        write_json_atomic(manifest, report)
