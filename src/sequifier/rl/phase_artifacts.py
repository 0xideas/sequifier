"""Commit completed phase exports before their run checkpoint handoff."""

import json
from pathlib import Path

from sequifier.fingerprints import fingerprint
from sequifier.rl.checkpoint import save_policy
from sequifier.rl.run_state import (
    require_equal,
    sync_file,
    weight_fingerprint,
    write_json_atomic,
)
from sequifier.trajectory.sources import file_checksum


class PhaseArtifacts:
    def __init__(self, directory, representation):
        self.directory = Path(directory)
        self.representation = representation

    def paths(self, result):
        directory = self.directory / f"{result.index:03d}-{result.name}"
        return (
            directory / "actor.pt",
            directory / "report.json",
            directory / "manifest.json",
        )

    def _manifest(self, result):
        actor, report, _ = self.paths(result)
        return {
            "format": "sequifier.policy-phase",
            "version": 1,
            "representation_fingerprint": self.representation.identity,
            "result_fingerprint": fingerprint(result.as_dict()),
            "selected_weight_hash": result.selected_weight_hash,
            "actor": {"path": actor.name, "sha256": file_checksum(actor)},
            "report": {"path": report.name, "sha256": file_checksum(report)},
        }

    def validate(self, result, record):
        """Check a committed export without loading earlier models into memory."""
        _, _, manifest = self.paths(result)
        require_equal(str(manifest), record["path"], "resume.phase_manifest.path")
        require_equal(
            record["sha256"], file_checksum(manifest), "resume.phase_manifest.sha256"
        )
        require_equal(
            self._manifest(result),
            json.loads(manifest.read_text(encoding="utf-8")),
            "resume.phase_artifacts",
        )

    def commit(self, actor, result):
        """Publish actor, report, then manifest; the run checkpoint commits last.

        Files not referenced by a committed run checkpoint are staged exports.
        A retry can reuse a complete staged export or replace an incomplete one.
        """
        require_equal(
            result.selected_weight_hash,
            weight_fingerprint(actor.network.state_dict()),
            "handoff.selected_actor",
        )
        actor_path, report, manifest = self.paths(result)
        if manifest.is_file():
            record = {"path": str(manifest), "sha256": file_checksum(manifest)}
            try:
                self.validate(result, record)
                return record
            except (OSError, ValueError, KeyError, TypeError):
                pass
        save_policy(actor, actor_path, metadata=result.as_dict())
        sync_file(actor_path)
        write_json_atomic(report, result.as_dict())
        write_json_atomic(manifest, self._manifest(result))
        return {"path": str(manifest), "sha256": file_checksum(manifest)}
