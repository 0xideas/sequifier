from typing import Any


def pt_bundle(model: Any, training_config: Any, export_with_dropout: bool) -> dict:
    """Return a self-contained PyTorch export payload."""
    return {
        "artifact_type": "sequifier_model",
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "training_config": training_config.model_dump(mode="python"),
        "export_with_dropout": export_with_dropout,
    }
