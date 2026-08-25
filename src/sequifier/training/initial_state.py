from typing import Any

import torch

from sequifier.artifacts.backbone_repository import load_revision, select_revision
from sequifier.artifacts.run_checkpoint import select_run_checkpoint
from sequifier.typechecking import beartype


@beartype
def select_initial_state(training_config: Any) -> dict[str, Any]:
    """Choose exactly one source, with complete-run resume taking precedence."""
    run_checkpoint = select_run_checkpoint(training_config)
    if run_checkpoint is not None:
        return {"kind": "run_checkpoint", **run_checkpoint}

    revision = select_revision(
        training_config.model_spec.backbone, training_config.project_root
    )
    if revision is not None:
        return {"kind": "backbone_revision", **revision}
    return {"kind": "fresh"}


@beartype
def load_model_initial_state(model: Any, source: dict[str, Any]) -> dict | None:
    """Load model tensors before wrapping and optimizer construction."""
    if source["kind"] == "fresh":
        model._backbone_parent_revision_id = None
        return None
    if source["kind"] == "backbone_revision":
        load_revision(model.backbone, source)
        model._backbone_parent_revision_id = source["revision_id"]
        return None
    if source["kind"] != "run_checkpoint":
        raise ValueError(f"Unknown initial state source {source['kind']!r}.")

    checkpoint = torch.load(source["path"], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Run checkpoint payload must be a dictionary.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "backbone_parent_revision_id" not in checkpoint:
        raise ValueError("Run checkpoint is missing backbone_parent_revision_id.")
    model._backbone_parent_revision_id = checkpoint["backbone_parent_revision_id"]
    return checkpoint
