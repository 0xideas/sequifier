"""Selected models and sidecar identities at committed phase boundaries."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_path, write_json_atomic
from sequifier.trajectory.sources import file_checksum


@dataclass(frozen=True)
class PhaseArtifacts:
    model: Path

    @classmethod
    def for_phase(cls, directory, index, name, filename="model.pt"):
        return cls(Path(directory) / f"{index:03d}-{name}" / filename)

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
    def stage(cls, artifacts, result, model, *, telemetry=None):
        from sequifier.artifacts.model_artifact import save_model_artifact

        path = artifacts.model.parent / "commits" / f"{uuid4().hex}.pt"
        save_model_artifact(model, path)
        sidecars = {"telemetry": telemetry} if telemetry is not None else {}
        return cls(
            str(path),
            file_checksum(path),
            result,
            fingerprint({"result": result, "sidecars": sidecars}),
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
    def load(cls, values):
        commit = cls(**values)
        require_contract_version(
            "phase commit", commit.format_version, ContractVersions.PHASE_COMMIT
        )
        if fingerprint({"result": commit.result, "sidecars": commit.sidecars}) != (
            commit.result_fingerprint
        ):
            raise ValueError("Committed phase result changed")
        verify_artifact(
            commit.model_path, commit.model_sha256, "Committed phase model changed"
        )
        return commit

    @property
    def model_identity(self):
        return {"path": self.model_path, "sha256": self.model_sha256}

    def report_result(self):
        from sequifier.rl.telemetry import TelemetryJournal

        result = dict(self.result)
        if result["algorithm"] == "supervised":
            return result
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

    def publish(self, artifacts, report):
        path = artifacts.model
        if not path.is_file() or file_checksum(path) != self.model_sha256:
            with atomic_path(path, durable=True) as temporary:
                copyfile(self.model_path, temporary)
        write_json_atomic(artifacts.report, report)


def verify_artifact(path, checksum, message):
    if path and file_checksum(path) != checksum:
        raise ValueError(message)
