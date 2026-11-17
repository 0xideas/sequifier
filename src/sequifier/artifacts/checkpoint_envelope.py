"""Shared execution envelope around runner-owned exact recovery state."""

from sequifier.artifacts.model_artifact import ModelArtifact

CHECKPOINT_FORMAT_VERSION = 4


def checkpoint_envelope(recovery_type, recovery, model):
    return {
        "artifact_type": "sequifier_run_checkpoint",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict() if isinstance(model, ModelArtifact) else model,
        "recovery_type": recovery_type,
        "recovery": recovery,
    }


def recovery_payload(payload, recovery_type):
    """Validate the shared envelope before restoring runner-owned state."""
    legacy = {
        "supervised": ("sequifier_run_checkpoint", 3),
        "policy": ("sequifier_policy_run", 7),
        "learning": ("sequifier_learning_run", 2),
    }
    if isinstance(payload, dict) and (
        payload.get("artifact_type"),
        payload.get("format_version"),
    ) == legacy.get(recovery_type):
        return payload
    if (
        isinstance(payload, dict)
        and payload.get("artifact_type") == "sequifier_run_checkpoint"
    ):
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported run checkpoint envelope version")
        if payload.get("recovery_type", "supervised") != recovery_type:
            raise ValueError(
                f"Checkpoint does not contain {recovery_type} recovery state"
            )
        if not isinstance(payload.get("recovery"), dict):
            raise ValueError("Run checkpoint requires a recovery mapping")
        return payload["recovery"]
    raise ValueError("Expected a Sequifier run checkpoint envelope")
