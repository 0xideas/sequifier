"""Path-scoped checkpoints for preprocessing commands that publish a directory."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import write_json_atomic

WORKSPACE_DIR = ".sequifier-preprocessing"


@contextmanager
def preprocessing_workspace(destination, *, indexed, resume):
    """Yield a checkpoint directory, or None when publication already completed."""
    import fcntl

    destination = Path(destination).absolute()
    parent = destination.parent / WORKSPACE_DIR
    parent.mkdir(parents=True, exist_ok=True)
    manifest_name = "dataset.json" if indexed else "manifest.json"
    expected_format = (
        "sequifier.indexed-sequences" if indexed else "sequifier.sequence-data"
    )
    work = parent / f"{destination.name}.{manifest_name}.partial"
    lock_path = parent / f"{destination.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            manifest = destination / manifest_name
            if manifest.is_file():
                values = json.loads(manifest.read_text())
                if values.get("format") != expected_format:
                    raise FileExistsError(destination)
                if not indexed:
                    require_contract_version(
                        expected_format,
                        values.get("version"),
                        ContractVersions.SEQUENCE_DATA_MANIFEST,
                    )
                if work.is_dir():
                    shutil.rmtree(work)
                logger.info(f"Found published dataset at {destination}")
                yield None
                return
            # Interrupted work lives beside the destination. Never recursively
            # remove a populated directory that this command did not create.
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(destination)
            destination.rmdir()
        if work.exists():
            if resume:
                logger.info(f"Resuming preprocessing from {work}")
            else:
                logger.info(f"Discarding partial preprocessing at {work}")
                shutil.rmtree(work)
        work.mkdir(exist_ok=True)
        publication = work / "publication"
        if publication.exists():
            shutil.rmtree(publication)
        yield work
        shutil.rmtree(work)


def indexed_preparation(config, destination, work, files, **ordered_fields):
    """Match both partial items and published views to the inputs that produced them."""
    if work is None:
        manifest = json.loads((destination / "dataset.json").read_text())
        previous = manifest.get("provenance", {}).get("preparation")
    else:
        path = work / "preparation.json"
        previous = json.loads(path.read_text()) if path.exists() else None
    if previous and config.run.reproducibility._requested_random:
        config = config.model_copy(deep=True)
        config.run.reproducibility.seed = previous["seed"]
    signature = fingerprint(
        {
            "settings": config.model_dump(
                mode="json", exclude={"reprocess", "indexed_output_path"}
            ),
            "split_order": list(config.splits),
            **ordered_fields,
            "files": files,
        }
    )
    identity = {"signature": signature, "seed": config.run.reproducibility.seed}
    if work is None and identity != previous:
        raise ValueError(
            "Published dataset belongs to different inputs or settings; choose a new indexed_output_path"
        )
    if work is not None:
        if previous is not None and previous != identity:
            raise ValueError(
                "Preprocessing inputs or settings changed; set reprocess: true to rebuild partial work"
            )
        if previous is None and any(work.iterdir()):
            raise ValueError(
                "Unidentified partial work; set reprocess: true to rebuild"
            )
        write_json_atomic(work / "preparation.json", identity)
    return config, identity
