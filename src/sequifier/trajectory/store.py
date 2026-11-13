"""Validated JSON trajectory storage with shared atomic publication."""

import json
from pathlib import Path

from sequifier.io.atomic import write_json_atomic
from sequifier.trajectory.schema import Trajectory, validate_trajectory


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    write_json_atomic(path, validate_trajectory(trajectory).model_dump(mode="json"))


def load_trajectory(path: str | Path) -> Trajectory:
    return validate_trajectory(json.loads(Path(path).read_text(encoding="utf-8")))
