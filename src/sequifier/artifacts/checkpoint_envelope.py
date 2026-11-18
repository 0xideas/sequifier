"""Shared execution envelope around runner-owned exact recovery state."""

from collections.abc import Mapping

from sequifier.artifacts.model_artifact import ModelArtifact
from sequifier.contract_versions import ContractVersions, require_contract_version


def checkpoint_envelope(recovery_type, recovery, model):
    return {
        "artifact_type": "sequifier_run_checkpoint",
        "format_version": ContractVersions.RUN_CHECKPOINT,
        "model": model.state_dict() if isinstance(model, ModelArtifact) else model,
        "recovery_type": recovery_type,
        "recovery": recovery,
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
