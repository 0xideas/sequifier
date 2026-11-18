"""Paths, identity and publication records at committed phase boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import write_json_atomic
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
    def receipt(self):
        return self.model.with_name("completed.json")

    @property
    def telemetry(self):
        return self.model.parent / "telemetry" / "records.sqlite3"

    def describe(self):
        return {"actor_path": str(self.model), "report_path": str(self.report)}

    def publish_model(self, artifact):
        from sequifier.artifacts.model_artifact import save_model_artifact

        save_model_artifact(artifact, self.model)

    def publish_report(self, report):
        write_json_atomic(self.report, report)

    def record(self, report):
        return {
            "sha256": file_checksum(self.model),
            "result_fingerprint": fingerprint(report),
        }

    def read_receipt(self, identity, *, run_id=None):
        if not self.receipt.is_file():
            return None
        receipt = json.loads(self.receipt.read_text())
        if run_id is not None and receipt.get("run_id") != run_id:
            return None
        if receipt["identity"] != identity or receipt["sha256"] != file_checksum(
            self.model
        ):
            raise ValueError("completed phase artifact or receipt changed")
        require_contract_version(
            "phase report",
            receipt["report"].get("format_version"),
            ContractVersions.PHASE_REPORT,
        )
        return receipt

    def write_receipt(self, identity, report, *, run_id=None):
        write_json_atomic(
            self.receipt,
            {
                "identity": identity,
                "run_id": run_id,
                "sha256": file_checksum(self.model),
                "report": report,
            },
        )


def verify_artifact(path, checksum, message):
    if path and file_checksum(path) != checksum:
        raise ValueError(message)
