"""Paths, identity and publication records at committed phase boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import write_json_atomic
from sequifier.trajectory.sources import file_checksum


@dataclass(frozen=True)
class PhaseArtifacts:
    model: Path

    @classmethod
    def for_phase(cls, directory, index, name, filename="actor.pt"):
        return cls(Path(directory) / f"{index:03d}-{name}" / filename)

    @property
    def report(self):
        return self.model.with_name("report.json")

    @property
    def receipt(self):
        return self.model.with_name("completed.json")

    def describe(self):
        return {"actor_path": str(self.model), "report_path": str(self.report)}

    def record(self, report):
        return {
            "sha256": file_checksum(self.model),
            "result_fingerprint": fingerprint(report),
        }

    def read_receipt(self, identity):
        if not self.receipt.is_file():
            return None
        receipt = json.loads(self.receipt.read_text())
        if receipt["identity"] != identity or receipt["sha256"] != file_checksum(
            self.model
        ):
            raise ValueError("completed phase artifact or receipt changed")
        return receipt

    def write_receipt(self, identity, report):
        write_json_atomic(
            self.receipt,
            {
                "identity": identity,
                "sha256": file_checksum(self.model),
                "report": report,
            },
        )


def verify_artifact(path, checksum, message):
    if path and file_checksum(path) != checksum:
        raise ValueError(message)
