"""Shared execution envelope around runner-owned exact recovery state."""

from collections.abc import Mapping

from sequifier.artifacts.model_artifact import ModelArtifact
from sequifier.contract_versions import ContractVersions, require_contract_version


def phase_boundary(run_id, *, selected=None, active=None):
    """References made authoritative by the same write as the runner state."""
    return {
        "format_version": ContractVersions.PHASE_BOUNDARY,
        "run_id": run_id,
        "selected": selected,
        "active": active,
    }


def selected_model_identity(completed):
    if not completed:
        return None
    latest = completed[-1]
    return {"path": latest["model_path"], "sha256": latest["model_sha256"]}


def checkpoint_envelope(recovery_type, recovery, model, *, boundary):
    return {
        "artifact_type": "sequifier_run_checkpoint",
        "format_version": ContractVersions.RUN_CHECKPOINT,
        "model": model.state_dict() if isinstance(model, ModelArtifact) else model,
        "recovery_type": recovery_type,
        "recovery": recovery,
        "phase_boundary": boundary,
    }


def validate_checkpoint_envelope(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a Sequifier run checkpoint envelope mapping")
    require_contract_version(
        "run checkpoint envelope",
        payload.get("format_version"),
        ContractVersions.RUN_CHECKPOINT,
    )
    if payload.get("artifact_type") != "sequifier_run_checkpoint":
        raise ValueError("Expected a Sequifier run checkpoint envelope")
    if payload.get("recovery_type") not in ("supervised", "policy", "learning"):
        raise ValueError(
            f"Unsupported checkpoint recovery kind {payload.get('recovery_type')!r}"
        )
    boundary = payload.get("phase_boundary")
    if boundary is not None:
        if not isinstance(boundary, dict):
            raise ValueError("Run checkpoint requires a phase boundary mapping")
        require_contract_version(
            "phase boundary",
            boundary.get("format_version"),
            ContractVersions.PHASE_BOUNDARY,
        )
    if not isinstance(payload.get("recovery"), dict):
        raise ValueError("Run checkpoint requires a recovery mapping")


def recovery_payload(payload, recovery_type):
    """Validate the shared envelope before restoring runner-owned state."""
    validate_checkpoint_envelope(payload)
    if payload["recovery_type"] != recovery_type:
        raise ValueError(
            f"Checkpoint recovery kind {payload['recovery_type']!r}; expected {recovery_type!r}"
        )
    return payload["recovery"]


def require_phase_boundary(payload, expected):
    if payload.get("phase_boundary") != expected:
        raise ValueError("Phase boundary disagrees with checkpoint recovery")
