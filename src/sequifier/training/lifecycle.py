from pathlib import Path
from typing import Any

from sequifier.artifacts.backbone_repository import publish_revision
from sequifier.artifacts.manifests import write_manifest
from sequifier.artifacts.run_checkpoint import checkpoint_path


def publish_final_backbone(model: Any, source_epoch: int) -> dict[str, Any]:
    config = model.hparams.model_spec.backbone
    if config.repository is None:
        return {"success": False, "reason": "repository_not_configured"}
    if not config.repository.publish:
        return {"success": False, "reason": "publication_disabled"}
    return publish_revision(
        model.backbone,
        config,
        model.project_root,
        parent_revision_id=model._backbone_parent_revision_id,
        source_run_id=model.model_name,
        source_epoch=source_epoch,
    )


def terminal_manifest_path(training_config: Any) -> Path:
    return checkpoint_path(training_config).parent / "manifest.json"


def write_terminal_manifest(
    model: Any,
    *,
    status: str,
    completion_reason: str,
    source_epoch: int,
    exports_succeeded: bool,
    publication: dict[str, Any],
) -> None:
    write_manifest(
        terminal_manifest_path(model.hparams),
        {
            "artifact_type": "sequifier_run",
            "format_version": 1,
            "run_id": model.model_name,
            "status": status,
            "completion_reason": completion_reason,
            "source_epoch": source_epoch,
            "exports_succeeded": exports_succeeded,
            "backbone_parent_revision_id": model._backbone_parent_revision_id,
            "backbone_publication": publication,
        },
    )
