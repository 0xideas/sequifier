"""Path-scoped checkpoints for preprocessing commands that publish a directory."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

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
                if work.is_dir():
                    shutil.rmtree(work)
                logger.info(f"Reusing published dataset at {destination}")
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
