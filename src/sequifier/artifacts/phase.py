"""Selected models and sidecar identities at committed phase boundaries."""

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from sequifier.artifacts.reports import PhaseResult
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.atomic import atomic_path, write_json_atomic


@dataclass(frozen=True)
class PhaseArtifacts:
    model: Path

    @classmethod
    def for_phase(cls, directory, index, name, filename="model.pt"):
        return cls(Path(directory) / f"{index:03d}-{name}" / filename)

    @property
    def recovery(self):
        return self.model.with_name("run.pt")

    @property
    def commits(self):
        return self.model.parent / "commits"

    @property
    def snapshots(self):
        return self.model.parent / "checkpoints"

    @property
    def report(self):
        return self.model.with_name("report.json")

    @property
    def telemetry(self):
        return self.model.parent / "telemetry" / "records.sqlite3"

    def describe(self):
        return {"actor_path": str(self.model), "report_path": str(self.report)}


@dataclass(frozen=True)
class PhaseCommit:
    """An immutable model and report payload published by a run checkpoint."""

    model_path: str
    model_sha256: str
    result: dict
    result_fingerprint: str
    sidecars: dict
    format_version: int = ContractVersions.PHASE_COMMIT

    @classmethod
    def stage(
        cls, artifacts, result: PhaseResult, model, *, checkpoint_path, telemetry=None
    ):
        from sequifier.artifacts.model_artifact import save_model_artifact

        path = artifacts.commits / f"{uuid4().hex}.pt"
        save_model_artifact(model, path)
        report = result.report(artifacts.model, checkpoint_path)
        sidecars = {"telemetry": telemetry} if telemetry is not None else {}
        return cls(
            str(path),
            file_checksum(path),
            report,
            fingerprint({"result": report, "sidecars": sidecars}),
            sidecars,
        )

    def with_result(self, result):
        return replace(
            self,
            result=result,
            result_fingerprint=fingerprint(
                {"result": result, "sidecars": self.sidecars}
            ),
        )

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, values, *, checkpoint_path=None):
        require_contract_version(
            "phase commit", values.get("format_version"), ContractVersions.PHASE_COMMIT
        )
        commit = cls(**values)
        if fingerprint({"result": commit.result, "sidecars": commit.sidecars}) != (
            commit.result_fingerprint
        ):
            raise ValueError("Committed phase result changed")
        commit.verify_model()
        if checkpoint_path is not None and commit.result["checkpoint_path"] != str(
            checkpoint_path
        ):
            return commit.with_result(
                {**commit.result, "checkpoint_path": str(checkpoint_path)}
            )
        return commit

    def verify_model(self):
        verify_artifact(
            self.model_path, self.model_sha256, "Committed phase model changed"
        )

    @property
    def model_identity(self):
        return {"path": self.model_path, "sha256": self.model_sha256}

    def report_result(self):
        result = dict(self.result)
        if result["algorithm"] == "supervised":
            return result
        from sequifier.rl.telemetry import TelemetryJournal

        sidecar = self.sidecars.get("telemetry")
        history = (
            TelemetryJournal.read(Path(sidecar["path"]), sidecar["position"])
            if sidecar
            else {}
        )
        result["metrics"] = history.get("metrics", [])
        if result["algorithm"] in {"ppo", "grpo"}:
            result["provenance"] = {
                **result["provenance"],
                **{
                    key: history.get(key, [])
                    for key in ("evaluations", "continuations")
                },
            }
        return result

    def publish(self):
        report = self.report_result()
        artifacts = PhaseArtifacts(Path(report["model_path"]))
        path = artifacts.model
        if not path.is_file() or file_checksum(path) != self.model_sha256:
            with atomic_path(path, durable=True) as temporary:
                copyfile(self.model_path, temporary)
        try:
            if json.loads(artifacts.report.read_text(encoding="utf-8")) == report:
                return
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        write_json_atomic(artifacts.report, report)


class CommittedPhases(Sequence[PhaseCommit]):
    """The ordered prefix of a resolved plan made durable by a checkpoint."""

    def __init__(self, phases, offset=0):
        self._phases = phases
        self._offset = offset
        self._commits: list[PhaseCommit] = []

    def __len__(self):
        return len(self._commits)

    def __getitem__(self, index):
        return self._commits[index]

    def append(self, commit: PhaseCommit):
        position = len(self)
        if position >= len(self._phases):
            raise ValueError("Completed phases exceed the training plan")
        phase = self._phases[position]
        expected = (self._offset + position, phase.name, phase.algorithm, phase.handoff)
        actual = tuple(
            commit.result[key] for key in ("index", "name", "algorithm", "handoff")
        )
        if actual != expected:
            raise ValueError(
                f"Phase commit {actual!r} disagrees with training plan {expected!r}"
            )
        self._commits.append(commit)

    @classmethod
    def load(cls, phases, records, *, offset=0, checkpoint_path=None):
        completed = cls(phases, offset)
        for record in records:
            completed.append(PhaseCommit.load(record, checkpoint_path=checkpoint_path))
        return completed

    def records(self):
        return [commit.as_dict() for commit in self]

    @property
    def selected_model(self):
        return self[-1].model_identity if self else None


def publish_phase_commits(commits: Iterable[PhaseCommit]) -> None:
    for commit in commits:
        commit.publish()


@dataclass(frozen=True)
class PhasePublication:
    """Publish once the checkpoint that owns a phase boundary is durable."""

    save_parent: Callable[[Path, PhaseCommit | None], None] | None = None

    @property
    def standalone(self) -> bool:
        return self.save_parent is None

    def after_checkpoint(self, path: Path, commit: PhaseCommit | None):
        if self.save_parent is not None:
            self.save_parent(path, commit)
        if commit is not None:
            commit.publish()

    def replay_native(self, commits: Iterable[PhaseCommit]):
        if self.standalone:
            publish_phase_commits(commits)


def verify_artifact(path, checksum, message):
    if path and file_checksum(path) != checksum:
        raise ValueError(message)
