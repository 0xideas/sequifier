from pathlib import Path
from typing import Any

from sequifier.io.atomic import write_json_atomic
from sequifier.typechecking import beartype


@beartype
def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    write_json_atomic(path, manifest, sort_keys=True, default=str, allow_nan=True)
