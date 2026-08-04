import contextlib
import fcntl
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ARTIFACT_TYPE = "sequifier_backbone"
FORMAT_VERSION = 1
SUPPORTED_DTYPES = {
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
SUPPORTED_DTYPES.update(
    dtype
    for name in ("float8_e4m3fn", "float8_e5m2")
    if (dtype := getattr(torch, name, None)) is not None
)


def canonical_architecture(architecture: Any) -> dict[str, Any]:
    if hasattr(architecture, "model_dump"):
        return architecture.model_dump(mode="json")
    if not isinstance(architecture, dict):
        raise TypeError("Backbone architecture must be a mapping or Pydantic model.")
    return architecture


def architecture_fingerprint(architecture: Any) -> str:
    payload = json.dumps(
        canonical_architecture(architecture),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_repository_path(path: str, project_root: str) -> Path:
    repository_path = Path(path)
    if not repository_path.is_absolute():
        repository_path = Path(project_root) / repository_path
    return repository_path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read backbone repository metadata {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Backbone repository metadata {path} must be an object.")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w") as file:
            json.dump(value, file, sort_keys=True, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        with contextlib.suppress(OSError):
            temporary_path.unlink()


def latest_pointer(repository_path: Path) -> dict[str, Any] | None:
    path = repository_path / "latest.json"
    if not path.exists():
        return None
    pointer = _read_json(path)
    if not isinstance(pointer.get("revision_id"), str):
        raise ValueError(f"Backbone latest pointer {path} has no revision_id.")
    revision_id = pointer["revision_id"]
    if not revision_id or Path(revision_id).name != revision_id:
        raise ValueError(f"Backbone latest pointer {path} has an unsafe revision_id.")
    parent = pointer.get("parent_revision_id")
    if parent is not None and not isinstance(parent, str):
        raise ValueError(
            f"Backbone latest pointer {path} has an invalid parent_revision_id."
        )
    return pointer


def select_revision(backbone_config: Any, project_root: str) -> dict[str, Any] | None:
    repository_path = resolve_repository_path(
        backbone_config.repository.path, project_root
    )
    pointer = latest_pointer(repository_path)
    if pointer is None:
        if backbone_config.repository.load_policy == "required":
            raise FileNotFoundError(
                f"Required backbone revision does not exist in {repository_path}."
            )
        return None

    revision_id = pointer["revision_id"]
    revision_path = repository_path / "revisions" / revision_id
    manifest_path = revision_path / "manifest.json"
    weights_path = revision_path / "weights.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"Backbone revision {revision_id!r} is incomplete in {revision_path}."
        )
    manifest = _read_json(manifest_path)
    expected_fingerprint = architecture_fingerprint(backbone_config.architecture)
    manifest_architecture = manifest.get("architecture")
    try:
        manifest_fingerprint = architecture_fingerprint(manifest_architecture)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Backbone revision {revision_id!r} has an invalid architecture."
        ) from exc
    checks = {
        "artifact_type": (manifest.get("artifact_type"), ARTIFACT_TYPE),
        "format_version": (manifest.get("format_version"), FORMAT_VERSION),
        "backbone_id": (manifest.get("backbone_id"), backbone_config.id),
        "revision_id": (manifest.get("revision_id"), revision_id),
        "architecture_fingerprint": (
            manifest.get("architecture_fingerprint"),
            expected_fingerprint,
        ),
        "manifest_architecture_fingerprint": (
            manifest_fingerprint,
            expected_fingerprint,
        ),
        "parent_revision_id": (
            manifest.get("parent_revision_id"),
            pointer.get("parent_revision_id"),
        ),
    }
    mismatches = [
        f"{name}: found {actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            f"Incompatible backbone revision {revision_id!r}: " + "; ".join(mismatches)
        )
    return {
        "repository_path": str(repository_path),
        "revision_id": revision_id,
        "parent_revision_id": manifest.get("parent_revision_id"),
        "manifest": manifest,
        "weights_path": str(weights_path),
    }


def _validate_state_dict(backbone: nn.Module, state_dict: Any) -> None:
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError(
            "Backbone weights must be a string-to-tensor state dictionary."
        )

    expected = backbone.state_dict()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        raise ValueError(
            "Backbone state-dict keys do not match exactly: "
            f"missing={missing}, unexpected={unexpected}."
        )
    invalid_dtypes = {
        key: str(value.dtype)
        for key, value in state_dict.items()
        if value.dtype not in SUPPORTED_DTYPES
    }
    if invalid_dtypes:
        raise ValueError(
            f"Backbone state dict contains unsupported dtypes: {invalid_dtypes}"
        )
    shape_mismatches = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in state_dict.items()
        if value.shape != expected[key].shape
    }
    if shape_mismatches:
        raise ValueError(
            f"Backbone state dict contains incompatible shapes: {shape_mismatches}"
        )


def load_revision(backbone: nn.Module, selected_revision: dict[str, Any]) -> None:
    state_dict = torch.load(
        selected_revision["weights_path"], map_location="cpu", weights_only=True
    )
    _validate_state_dict(backbone, state_dict)
    backbone.load_state_dict(state_dict, strict=True)


def _new_revision_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def publish_revision(
    backbone: nn.Module,
    backbone_config: Any,
    project_root: str,
    *,
    parent_revision_id: str | None,
    source_run_id: str,
    source_epoch: int,
) -> dict[str, Any]:
    repository_path = resolve_repository_path(
        backbone_config.repository.path, project_root
    )
    revisions_path = repository_path / "revisions"
    revisions_path.mkdir(parents=True, exist_ok=True)
    lock_path = repository_path / "lock"

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        pointer = latest_pointer(repository_path)
        current_revision_id = pointer["revision_id"] if pointer else None
        if current_revision_id != parent_revision_id:
            return {
                "success": False,
                "reason": "compare_and_swap_conflict",
                "expected_parent_revision_id": parent_revision_id,
                "current_revision_id": current_revision_id,
            }

        revision_id = _new_revision_id()
        revision_path = revisions_path / revision_id
        revision_path.mkdir(exist_ok=False)
        weights_path = revision_path / "weights.pt"
        temporary_weights_path = revision_path / f".weights.{uuid.uuid4().hex}.tmp"
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in backbone.state_dict().items()
        }
        _validate_state_dict(backbone, state_dict)
        try:
            torch.save(state_dict, temporary_weights_path)
            os.replace(temporary_weights_path, weights_path)
        finally:
            with contextlib.suppress(OSError):
                temporary_weights_path.unlink()

        architecture = canonical_architecture(backbone_config.architecture)
        manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "format_version": FORMAT_VERSION,
            "backbone_id": backbone_config.id,
            "revision_id": revision_id,
            "parent_revision_id": parent_revision_id,
            "architecture_fingerprint": architecture_fingerprint(architecture),
            "architecture": architecture,
            "source_run_id": source_run_id,
            "source_epoch": source_epoch,
        }
        _write_json_atomic(revision_path / "manifest.json", manifest)
        _write_json_atomic(
            repository_path / "latest.json",
            {
                "revision_id": revision_id,
                "parent_revision_id": parent_revision_id,
            },
        )
        return {
            "success": True,
            "revision_id": revision_id,
            "parent_revision_id": parent_revision_id,
            "manifest_path": str(revision_path / "manifest.json"),
            "weights_path": str(weights_path),
        }
