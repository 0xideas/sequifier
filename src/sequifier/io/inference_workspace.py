"""Destination-scoped execution and publication for the inference command."""

import json
import shutil
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger

from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.helpers import normalize_path
from sequifier.io.atomic import atomic_path, write_json_atomic
from sequifier.io.output_paths import (
    require_file_destinations,
    validate_inference_destination,
)

_DIRECTORY = ".sequifier-inference"


def _entry(path):
    path = Path(path)
    path = path.parent.resolve() / path.name
    if _DIRECTORY in path.parts or path.name in {"", ".", ".."}:
        raise ValueError(f"Reserved inference destination: {path}")
    require_file_destinations([path])
    return path


def _control(path):
    return path.parent / _DIRECTORY / "files" / path.name


@contextmanager
def _lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _primary(request):
    root = request.run.project_root
    rollout = getattr(request, "rollout", None)
    if rollout:
        return normalize_path(
            rollout.trajectory_path or rollout.output.manifest_path, root
        )
    from sequifier.infer import _inference_destination

    output = request.output
    path = output.path if output else None
    if path and request.task == "policy":
        return normalize_path(path, root)
    destination, _ = _inference_destination(
        root,
        output.format if output else "csv",
        "embeddings" if request.task == "embedding" else "predictions",
        Path(request.artifact.paths[0]).stem,
        0,
        output_path=path,
    )
    return destination


class InferenceWorkspace:
    def __init__(self, directory, record, previous=None):
        self.directory = directory
        self.record = record
        self.outputs = {}
        self.previous = previous

    @property
    def work(self):
        return self.directory / UUID(self.record["id"]).hex

    def save(self):
        write_json_atomic(self.directory / "run.json", self.record)

    def begin(self, inputs):
        self.work.mkdir(exist_ok=True)
        self.record.update(
            state="running", inputs=[str(Path(path).absolute()) for path in inputs]
        )
        self.save()
        if self.previous is not None and self.previous.is_dir():
            shutil.rmtree(self.previous)

    def publish(self):
        """Serialize overlapping output sets, including another run's sidecars."""
        outputs = self.record["outputs"]
        paths = [Path(name) for name in outputs]
        with ExitStack() as locks:
            for parent in sorted({path.parent for path in paths}):
                locks.enter_context(_lock(parent / _DIRECTORY / "publish.lock"))
            for index, path in enumerate(paths):
                owner_path = _control(path) / "owner.json"
                if path.is_file() and _read(owner_path) == self.record["id"]:
                    continue
                staged = self.work / f"output-{index}"
                if not staged.is_file():
                    return False
                validate_inference_destination(
                    path, self.record["inputs"], label="output"
                )
                with atomic_path(path, durable=True) as temporary:
                    shutil.copyfile(staged, temporary)
                    # Invalidate the previous publication before replacing the file.
                    write_json_atomic(owner_path, None)
                write_json_atomic(owner_path, self.record["id"])
            if self.record["state"] != "complete":
                self.record["state"] = "complete"
                self.save()
            if "rollout" not in self.record["request"]:
                for index in range(len(outputs)):
                    (self.work / f"output-{index}").unlink(missing_ok=True)
        return True


_active: ContextVar[InferenceWorkspace | None] = ContextVar(
    "inference_workspace", default=None
)


@contextmanager
def inference_workspace(request, *, random_seed, skip_metadata):
    primary = _entry(_primary(request))
    # Check declared sources before creating the destination's lock directory.
    declared = [
        *request.artifact.paths,
        request.input.path,
        request.input.static_data_path,
        request.input.metadata_config_path,
    ]
    validate_inference_destination(
        primary,
        [normalize_path(path, request.run.project_root) for path in declared if path],
        label="inference output",
    )
    directory = _control(primary)
    identity = request.model_dump(mode="json", exclude_none=True)
    identity["run"]["project_root"] = str(Path(request.run.project_root).resolve())
    if not getattr(request, "rollout", None) and request.output is None:
        from sequifier.config.inference_request import (
            ForecastOutput,
            PolicyOutput,
            TableOutput,
        )

        default = {
            "forecast": ForecastOutput,
            "policy": PolicyOutput,
            "embedding": TableOutput,
        }
        identity["output"] = default[request.task]().model_dump(exclude_none=True)
    if rollout := getattr(request, "rollout", None):
        if rollout.output:
            identity["sequence_data_version"] = ContractVersions.SEQUENCE_DATA_MANIFEST
    if random_seed:
        identity["run"]["reproducibility"]["seed"] = "random"
    if (
        request.task == "policy"
        and request.rollout
        and request.rollout.seed_history == "current"
    ):
        # A legacy single-observation rollout can resume its original adapter journal.
        identity["rollout"].pop("seed_history", None)
    if request.task == "policy" and not request.rollout:
        # Invalidate cached tables when policy prediction semantics change.
        identity["prediction_table_version"] = ContractVersions.POLICY_PREDICTION_TABLE
    identity["skip_metadata"] = skip_metadata
    with _lock(directory / "run.lock"):
        record = _read(directory / "run.json")
        if record is not None:
            require_contract_version(
                "inference workspace",
                record.get("version"),
                ContractVersions.INFERENCE_WORKSPACE,
            )
        previous = None
        if record is not None and json.dumps(
            record["request"], sort_keys=True
        ) != json.dumps(identity, sort_keys=True):
            previous = InferenceWorkspace(directory, record).work
            if record["state"] != "complete" and (previous / "reset.json").exists():
                raise RuntimeError(
                    f"Finish the interrupted rollout at {primary} before replacing it"
                )
            record = None
        if record is None:
            record = {
                "version": ContractVersions.INFERENCE_WORKSPACE,
                "id": uuid4().hex,
                "state": "running",
                "request": identity,
                "seed": request.run.reproducibility.seed,
            }
        work = InferenceWorkspace(directory, record, previous)
        if record["state"] in {"complete", "ready"} and work.publish():
            logger.info(f"Reusing inference at '{primary}'")
            yield None
            return
        if record["state"] == "complete" and "rollout" not in identity:
            work.previous = work.work
            record["id"] = uuid4().hex
        reset = _read(work.work / "reset.json")
        if reset and reset["recovery"] == "manual":
            raise RuntimeError(
                f"Interrupted inference episode {record['id']} at {primary} "
                "requires manual adapter recovery"
            )
        token = _active.set(work)
        try:
            yield work
            record.update(state="ready", outputs=list(work.outputs))
            work.save()
            if not work.publish():
                raise RuntimeError("Missing staged inference output")
        finally:
            _active.reset(token)


@contextmanager
def inference_output(path):
    work = _active.get()
    if work is None:
        yield path
        return
    path = str(_entry(path))
    name = f"output-{len(work.outputs)}"
    if path in work.outputs:
        raise ValueError(f"Multiple inference outputs target {path}")
    with atomic_path(work.work / name, durable=True) as temporary:
        yield str(temporary)
    work.outputs[path] = None


def rollout_journal():
    work = _active.get()
    if work is not None:
        from sequifier.rollout.inference_journal import InferenceJournal

        return InferenceJournal(work.work, work.record["id"])
    return None


def bind_inference_inputs(config):
    work = _active.get()
    if work is not None:
        work.begin(config.input_paths())


def write_inference_data(data, path, write_format):
    from sequifier.helpers import write_data

    with inference_output(path) as destination:
        write_data(data, destination, write_format)


def write_inference_trajectory(trajectory, path):
    from sequifier.trajectory.store import save_trajectory

    with inference_output(path) as destination:
        save_trajectory(trajectory, destination)
