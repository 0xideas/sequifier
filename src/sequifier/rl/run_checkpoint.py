"""Policy recovery with artifact-backed, authoritative committed phase boundaries."""

import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch

from sequifier.artifacts.checkpoint_envelope import (
    checkpoint_envelope,
    recovery_payload,
)
from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    ModelArtifactMetadata,
    ModelExecutionConfig,
)
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.fingerprints import fingerprint
from sequifier.io.atomic import atomic_save
from sequifier.rl.checkpoint import phase_actor_path, policy_model_artifact
from sequifier.rl.references import validate_reference_state
from sequifier.rl.run_state import (
    PHASE_STATE_VERSION,
    ActorRepresentation,
    PolicyPhaseResult,
    clone_weights,
    require_equal,
    weight_fingerprint,
)
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import new_run_id
from sequifier.runtime.random_state import capture_rng_state
from sequifier.trajectory.sources import file_checksum

POLICY_RECOVERY_VERSION = 8
ONLINE_TELEMETRY = ("evaluations", "continuations")


@dataclass(frozen=True)
class CompletedPhase:
    result: PolicyPhaseResult
    sha256: str
    result_fingerprint: str
    telemetry_position: str | None = None

    @classmethod
    def from_dict(cls, values, *, legacy):
        result = values["result"]
        return cls(
            PolicyPhaseResult(**{**result, "metrics": tuple(result["metrics"])}),
            values["sha256"],
            values["result_fingerprint"],
            None if legacy else values["telemetry_position"],
        )

    def as_dict(self):
        return asdict(self)

    def report_result(self):
        artifacts = PhaseArtifacts(Path(self.result.provenance["actor_path"]))
        history = TelemetryJournal.read(artifacts.telemetry, self.telemetry_position)
        return replace(
            self.result,
            metrics=tuple(history.get("metrics", ())),
            provenance={
                **self.result.provenance,
                **{
                    key: history.get(key, [])
                    for key in ONLINE_TELEMETRY
                    if self.result.algorithm in {"ppo", "grpo"}
                },
            },
        )


@dataclass
class RunCheckpoint:
    plan: dict
    representation: ActorRepresentation
    data_identity: dict
    data_contract: list
    adapter_contract: dict
    initialization: dict
    incoming_actor: dict
    random: dict
    reference_contract: dict = field(default_factory=dict)
    completed: list[CompletedPhase] = field(default_factory=list)
    phase_state: dict | None = None
    continuation: dict | None = None
    stopped: bool = False
    run_id: str = field(default_factory=new_run_id)

    @property
    def phase_index(self):
        return len(self.completed)

    @classmethod
    def load(cls, path, config, plan, data_identity, *, run_id=None):
        path = Path(path)
        if config.resume.policy == "never":
            return None
        if not path.is_file():
            if config.resume.policy == "required":
                raise FileNotFoundError(path)
            return None
        payload = recovery_payload(
            torch.load(path, map_location="cpu", weights_only=False), "policy"
        )
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != "sequifier_policy_run"
            or payload.get("format_version") not in {7, POLICY_RECOVERY_VERSION}
        ):
            raise ValueError(
                "policy resume requires a version-7 or version-8 policy run checkpoint; initialize from its model artifact to start a new run"
            )
        if run_id is not None and payload.get("run_id") != run_id:
            return None
        try:
            legacy = payload["format_version"] == 7
            saved_plan = payload["plan"]
            require_equal(plan, saved_plan, "resume.plan")
            require_equal(data_identity, payload["data_identity"], "resume.data")
            completed = [
                CompletedPhase.from_dict(record, legacy=legacy)
                for record in payload["completed"]
            ]
            checkpoint = cls(
                **{
                    f.name: payload[f.name]
                    for f in fields(cls)
                    if f.name not in {"representation", "completed"}
                    and f.name in payload
                },
                representation=ActorRepresentation(payload["representation"]),
                completed=completed,
            )
            require_equal(
                checkpoint.representation.identity,
                payload["representation_fingerprint"],
                "resume.representation_fingerprint",
            )
            checkpoint.validate(config, legacy=legacy)
            if legacy:
                checkpoint.migrate_telemetry(config)
            return checkpoint
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError(f"malformed policy run checkpoint: {exc}") from exc

    def validate(self, config, *, legacy=False):
        if self.phase_index > len(config.phases):
            raise ValueError("checkpoint completed handoffs disagree with the plan")
        validate_finite_state(self.incoming_actor, "incoming_actor")
        # Parsing the owned representation also validates the persisted policy schema.
        self.representation.policy
        for index, record in enumerate(self.completed):
            result = record.result
            phase = config.phases[index]
            require_equal(
                (index, phase.name, phase.algorithm, phase.handoff),
                (result.index, result.name, result.algorithm, result.handoff),
                "resume.completed_phase",
            )
            path = phase_actor_path(config.phase_directory, index, phase.name)
            require_equal(
                str(path), result.provenance["actor_path"], "resume.actor_path"
            )
            require_equal(
                str(path.with_name("report.json")),
                result.provenance["report_path"],
                "resume.report_path",
            )
            require_equal(
                record.sha256, file_checksum(path), "resume.phase_actor.sha256"
            )
            result_data = result.as_dict()
            if not legacy:
                result_data["telemetry_position"] = record.telemetry_position
            require_equal(
                record.result_fingerprint,
                fingerprint(result_data),
                "resume.phase_result",
            )
            validate_finite_state(result_data, "completed_phase")
        incoming_identity = weight_fingerprint(self.incoming_actor)
        require_equal(
            self.completed[-1].result.selected_weight_hash
            if self.completed
            else self.initialization["weight_hash"],
            incoming_identity,
            "resume.incoming_actor",
        )
        state = self.phase_state
        if state is None:
            return
        if self.phase_index == len(config.phases):
            raise ValueError("completed run cannot have a live phase state")
        if state["version"] != (3 if legacy else PHASE_STATE_VERSION):
            raise ValueError("unsupported phase checkpoint state version")
        if not legacy and not isinstance(state["telemetry_position"], str):
            raise ValueError("phase checkpoint requires a committed telemetry position")
        transaction = state["transaction"]
        if (
            not isinstance(transaction, int) or isinstance(transaction, bool)
        ) or transaction < 1:
            raise ValueError(
                "phase checkpoint must follow a completed optimizer transaction"
            )
        selection = state["selection"]
        best = selection["best_transaction"]
        if best is not None and (
            (not isinstance(best, int) or isinstance(best, bool))
            or not 0 <= best <= transaction
        ):
            raise ValueError(
                "checkpoint selected transaction is outside committed progress"
            )
        if (selection["best_metric"] is None) != (
            selection["best_weights"] is None
        ) or (best is None) != (selection["best_weights"] is None):
            raise ValueError(
                "checkpoint best actor, metric and transaction must be recorded together"
            )
        validate_reference_state(config.phases[self.phase_index], self)
        validate_finite_state(state, "phase")

    def migrate_telemetry(self, config):
        for index, record in enumerate(self.completed):
            result = record.result
            artifacts = PhaseArtifacts(Path(result.provenance["actor_path"]))
            provenance = dict(result.provenance)
            histories = {key: provenance.pop(key, ()) for key in ONLINE_TELEMETRY}
            position = import_telemetry(
                artifacts.telemetry, {"metrics": result.metrics, **histories}
            )
            compact = replace(result, metrics=(), provenance=provenance)
            self.completed[index] = replace(
                record,
                result=compact,
                result_fingerprint=fingerprint(
                    {**compact.as_dict(), "telemetry_position": position}
                ),
                telemetry_position=position,
            )
        if self.phase_state is not None:
            state = self.phase_state
            phase = config.phases[self.phase_index]
            artifacts = PhaseArtifacts.for_phase(
                config.phase_directory, self.phase_index, phase.name
            )
            state["telemetry_position"] = import_telemetry(
                artifacts.telemetry,
                {key: state.pop(key, ()) for key in ("metrics", *ONLINE_TELEMETRY)},
            )
            state["version"] = PHASE_STATE_VERSION

    def validate_actor(self, actor, units):
        self.representation.validate_actor(actor, units, "resume.representation")
        expected = actor.network.state_dict()
        weights = [self.incoming_actor]
        if self.phase_state is not None:
            state = self.phase_state
            weights += [
                state["networks"]["actor"],
                state["reference"],
                state["selection"]["best_weights"],
            ]
        for snapshot in weights:
            if snapshot is None:
                continue
            if snapshot.keys() != expected.keys() or any(
                snapshot[k].shape != v.shape or snapshot[k].dtype != v.dtype
                for k, v in expected.items()
            ):
                raise ValueError(
                    "checkpoint actor tensors disagree with the resolved model"
                )
        actor.network.load_state_dict(self.incoming_actor, strict=True)

    def save(self, path, device, state=None):
        """Atomically commit the run and its algorithm-owned recovery state."""
        self.random = capture_rng_state(device)
        self.phase_state = state
        model = ModelArtifact(
            3,
            ModelExecutionConfig(self.representation.values["execution"]),
            state["networks"]["actor"] if state else self.incoming_actor,
            ModelArtifactMetadata(),
        )
        atomic_save(
            path,
            checkpoint_envelope(
                "policy",
                {
                    "artifact_type": "sequifier_policy_run",
                    "format_version": POLICY_RECOVERY_VERSION,
                    **{
                        f.name: getattr(self, f.name)
                        for f in fields(self)
                        if f.name not in {"representation", "completed"}
                    },
                    "representation": self.representation.canonical,
                    "representation_fingerprint": self.representation.identity,
                    "completed": [record.as_dict() for record in self.completed],
                },
                model,
            ),
        )

    def finish_phase(self, actor, result, telemetry_position):
        require_equal(self.phase_index, result.index, "handoff.phase_index")
        require_equal(
            result.selected_weight_hash,
            weight_fingerprint(actor.network.state_dict()),
            "handoff.selected_actor",
        )
        artifacts = PhaseArtifacts(Path(result.provenance["actor_path"]))
        result_data = result.as_dict()
        artifacts.publish_model(policy_model_artifact(actor, metadata=result_data))
        weights = clone_weights(actor.network)
        self.completed.append(
            CompletedPhase(
                result,
                **artifacts.record(
                    {**result_data, "telemetry_position": telemetry_position}
                ),
                telemetry_position=telemetry_position,
            )
        )
        self.stopped = result.stop_scope == "end_plan"
        self.incoming_actor = weights
        self.phase_state = None


def import_telemetry(path, histories):
    with TelemetryJournal.open(path) as journal:
        for kind, records in histories.items():
            for record in records:
                journal.append(kind, record)
        return journal.commit()


def validate_finite_state(value, field):
    """Check algorithm-owned optimizer/network trees before restoring them."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite checkpoint tensor: {field}")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_finite_state(item, f"{field}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_finite_state(item, f"{field}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite checkpoint value: {field}")
