"""Validated JSON trajectory storage with shared atomic publication."""

import json
from pathlib import Path

from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.io.atomic import write_json_atomic
from sequifier.trajectory.schema import Trajectory, validate_trajectory

_TRAJECTORY_VERSIONS = {
    "sequifier.numeric-forecast-trajectory": ContractVersions.FORECAST_TRAJECTORY,
    "sequifier.scalar-policy-trajectory": ContractVersions.SCALAR_POLICY_TRAJECTORY,
    "sequifier.policy-trajectory": ContractVersions.JOINT_POLICY_TRAJECTORY,
}


def save_trajectory(trajectory: Trajectory, path: str | Path) -> None:
    values = validate_trajectory(trajectory).model_dump(mode="json")
    values["version"] = _TRAJECTORY_VERSIONS[values["format"]]
    write_json_atomic(path, values)


def load_trajectory(path: str | Path) -> Trajectory:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = values.get("format") if isinstance(values, dict) else None
    if not isinstance(kind, str) or kind not in _TRAJECTORY_VERSIONS:
        raise ValueError("Unsupported trajectory format")
    version = values.pop("version", None)
    expected = _TRAJECTORY_VERSIONS[kind]
    if (
        kind == "sequifier.policy-trajectory"
        and version == ContractVersions.SINGLE_SEED_JOINT_POLICY_TRAJECTORY
    ):
        expected = ContractVersions.SINGLE_SEED_JOINT_POLICY_TRAJECTORY
    require_contract_version(kind, version, expected)
    return validate_trajectory(values)
