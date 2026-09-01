from __future__ import annotations

from pathlib import Path
from typing import Any

from sequifier.artifacts.model_artifact import build_model_artifact, save_model_artifact


class PyTorchModelExporter:
    def export(
        self, network: Any, config: Any, destination: Path, options: Any
    ) -> Path:
        artifact = build_model_artifact(
            network,
            config,
            provenance={"export": options.suffix, "epoch": options.epoch},
        )
        return save_model_artifact(artifact, destination)
