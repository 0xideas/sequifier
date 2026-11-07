"""Atomic JSON storage, with validation at both read and write boundaries."""

import json
import os
import tempfile
from pathlib import Path

from sequifier.trajectory.schema import Trajectory, validate_trajectory


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    validated = validate_trajectory(trajectory)
    # Reject non-JSON/non-finite metadata as well as invalid numeric records.
    payload = json.dumps(validated.model_dump(mode="json"), indent=2, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_trajectory(path: str | Path) -> Trajectory:
    return validate_trajectory(json.loads(Path(path).read_text(encoding="utf-8")))
