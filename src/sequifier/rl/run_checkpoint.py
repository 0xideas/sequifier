"""Policy recovery with artifact-backed, authoritative committed phase boundaries."""

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from sequifier.artifacts.checkpoint_envelope import (
    checkpoint_envelope,
    phase_boundary,
    recovery_payload,
    require_phase_boundary,
)
from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    ModelArtifactMetadata,
    ModelExecutionConfig,
)
from sequifier.artifacts.phase import PhaseArtifacts, PhaseCommit
from sequifier.contract_versions import ContractVersions, require_contract_version
from sequifier.io.atomic import atomic_save
from sequifier.rl.checkpoint import phase_actor_path, policy_model_artifact
from sequifier.rl.references import validate_reference_state
from sequifier.rl.run_state import (
    ActorRepresentation,
    OptimizerContinuation,
    PhaseExecution,
    PolicyPhaseResult,
    require_equal,
    weight_fingerprint,
)
from sequifier.runtime.plan import new_run_id
from sequifier.runtime.random_state import capture_rng_state


@dataclass(frozen=True)
class CompletedPhase:
    commit: PhaseCommit

    @property
    def result(self):
        return PolicyPhaseResult.from_report(self.commit.result)

    @classmethod
    def from_dict(cls, values, checkpoint):
        commit = PhaseCommit.load(values)
        if commit.result["checkpoint_path"] != str(checkpoint):
            commit = commit.with_result(
                {**commit.result, "checkpoint_path": str(checkpoint)}
            )
        return cls(commit)

    def as_dict(self):
        return self.commit.as_dict()


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
    continuation: OptimizerContinuation | None = None
    stopped: bool = False
    run_id: str = field(default_factory=new_run_id)

    @property
    def phase_index(self):
        return len(self.completed)

    @property
    def next_phase_index(self):
        return self.plan.get("phase_offset", 0) + self.phase_index

    @classmethod
    def load(cls, path, config, plan, data_identity=None, *, run_id=None):
        path = Path(path)
        if config.resume.policy == "never":
            return None
        if not path.is_file():
            if config.resume.policy == "required":
                raise FileNotFoundError(path)
            return None
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        payload = recovery_payload(envelope, "policy")
        require_contract_version(
            "policy recovery",
            payload.get("format_version"),
            ContractVersions.POLICY_RECOVERY,
        )
        if payload.get("artifact_type") != "sequifier_policy_run":
            raise ValueError("Expected a policy checkpoint recovery payload")
        if run_id is not None and payload.get("run_id") != run_id:
            if config.resume.policy == "required":
                raise ValueError("Policy recovery belongs to another run")
            return None
        try:
            saved_plan = payload["plan"]
            require_equal(plan, saved_plan, "resume.plan")
            completed = [
                CompletedPhase.from_dict(record, path)
                for record in payload["completed"]
            ]
            checkpoint = cls(
                **{
                    f.name: payload[f.name]
                    for f in fields(cls)
                    if f.name not in {"representation", "completed"}
                },
                representation=ActorRepresentation(payload["representation"]),
                completed=completed,
            )
            require_equal(
                checkpoint.representation.identity,
                payload["representation_fingerprint"],
                "resume.representation_fingerprint",
            )
            if data_identity is not None:
                checkpoint.validate_dependencies(config, data_identity)
            checkpoint.validate(config)
            require_phase_boundary(envelope, checkpoint.phase_boundary())
            return checkpoint
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError(f"malformed policy run checkpoint: {exc}") from exc

    def validate_dependencies(self, config, data_identity):
        if config.actor_initialization == "history_only":
            path = str((Path(config.project_root) / config.initialize_from).resolve())
            require_equal(
                self.initialization["transfer"]["source_sha256"],
                data_identity[path],
                "resume.initialize_from.sha256",
            )
        require_equal(data_identity, self.data_identity, "resume.data")

    def validate(self, config):
        if self.phase_index > len(config.phases):
            raise ValueError("checkpoint completed handoffs disagree with the plan")
        validate_finite_state(self.incoming_actor, "incoming_actor")
        # Parsing the owned representation also validates the persisted policy schema.
        self.representation.policy
        for position, record in enumerate(self.completed):
            result = record.result
            phase = config.phases[position]
            index = config.phase_offset + position
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
                str(path), record.commit.result["model_path"], "resume.model_path"
            )
            if sidecar := record.commit.sidecars.get("telemetry"):
                require_equal(
                    str(PhaseArtifacts(path).telemetry),
                    sidecar["path"],
                    "resume.telemetry.path",
                )
            validate_finite_state(record.commit.result, "completed_phase")
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
        require_contract_version(
            "policy phase state",
            state.get("version"),
            ContractVersions.POLICY_PHASE_STATE,
        )
        artifacts = PhaseArtifacts.for_phase(
            config.phase_directory,
            self.next_phase_index,
            config.phases[self.phase_index].name,
        )
        require_equal(
            str(artifacts.telemetry), state["telemetry_path"], "resume.telemetry.path"
        )
        if not isinstance(state["telemetry_position"], str):
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
        validate_reference_state(
            config.phases[self.phase_index],
            self,
            self.next_phase_index,
        )
        validate_finite_state(state, "phase")

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

    def phase_boundary(self):
        state = self.phase_state
        active = None
        if state is not None:
            active = {
                "index": self.next_phase_index,
                "transaction": state["transaction"],
                "sidecars": {
                    "telemetry": {
                        "path": state["telemetry_path"],
                        "position": state["telemetry_position"],
                    }
                },
            }
        return phase_boundary(
            self.run_id,
            selected=self.completed[-1].commit.model_identity
            if self.completed
            else None,
            active=active,
        )

    def save(self, path, device, state=None):
        """Atomically commit the run and its algorithm-owned recovery state."""
        self.random = capture_rng_state(device)
        self.phase_state = state
        model = ModelArtifact(
            ContractVersions.MODEL_ARTIFACT,
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
                    "format_version": ContractVersions.POLICY_RECOVERY,
                    **{
                        f.name: vars(self)[f.name]
                        for f in fields(self)
                        if f.name not in {"representation", "completed"}
                    },
                    "representation": self.representation.canonical,
                    "representation_fingerprint": self.representation.identity,
                    "completed": [record.as_dict() for record in self.completed],
                },
                model,
                boundary=self.phase_boundary(),
            ),
        )

    def finish_phase(self, execution: PhaseExecution, telemetry_position, checkpoint):
        actor, result = execution.actor, execution.result
        require_equal(self.next_phase_index, result.index, "handoff.phase_index")
        require_equal(
            result.selected_weight_hash,
            weight_fingerprint(execution.weights),
            "handoff.selected_actor",
        )
        artifacts = PhaseArtifacts(Path(result.provenance["actor_path"]))
        result_data = result.as_dict()
        commit = PhaseCommit.stage(
            artifacts,
            result_data,
            checkpoint_path=checkpoint,
            model=policy_model_artifact(
                actor, metadata=result_data, weights=execution.weights
            ),
            telemetry={"path": str(artifacts.telemetry), "position": telemetry_position}
            if telemetry_position is not None
            else None,
        )
        self.completed.append(CompletedPhase(commit))
        self.stopped = result.stop_scope == "end_plan"
        self.incoming_actor = execution.weights
        self.continuation = execution.continuation
        self.phase_state = None


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
