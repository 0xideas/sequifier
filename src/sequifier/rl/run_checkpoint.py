"""Self-contained policy recovery with one authoritative committed phase boundary."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sequifier.rl.registry import BUILTIN_ALGORITHMS
from sequifier.rl.run_state import (
    ActorRepresentation,
    PolicyPhaseResult,
    atomic_save,
    capture_rng_state,
    clone_weights,
    require_equal,
    weight_fingerprint,
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
    results: list[PolicyPhaseResult] = field(default_factory=list)
    selected_actors: list[dict] = field(default_factory=list)
    phase_state: dict | None = None

    @property
    def phase_index(self):
        return len(self.results)

    @property
    def status(self):
        if self.phase_index == len(self.plan["phases"]):
            return "complete"
        if self.phase_state is not None:
            return "running"
        return "handoff" if self.results else "ready"

    @classmethod
    def load(cls, path, config, plan, data_identity):
        path = Path(path)
        if config.resume.policy == "never":
            return None
        if not path.is_file():
            if config.resume.policy == "required":
                raise FileNotFoundError(path)
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("artifact_type") != "sequifier_policy_run"
            or payload.get("format_version") != 3
        ):
            raise ValueError("policy resume requires a version-3 policy run checkpoint")
        try:
            require_equal(plan, payload["plan"], "resume.plan")
            require_equal(data_identity, payload["data_identity"], "resume.data")
            results = [
                PolicyPhaseResult(**{**result, "metrics": tuple(result["metrics"])})
                for result in payload["completed_results"]
            ]
            checkpoint = cls(
                payload["plan"],
                ActorRepresentation(payload["representation"]),
                payload["data_identity"],
                payload["data_contract"],
                payload["adapter_contract"],
                payload["initialization"],
                payload["incoming_actor"],
                payload["random"],
                payload["reference_contract"],
                results,
                payload["selected_actors"],
                payload["phase_state"],
            )
            require_equal(
                checkpoint.representation.identity,
                payload["representation_fingerprint"],
                "resume.representation_fingerprint",
            )
            checkpoint.validate(config)
            return checkpoint
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError(f"malformed policy run checkpoint: {exc}") from exc

    def validate(self, config):
        if len(self.results) > len(config.phases) or len(self.selected_actors) != len(
            self.results
        ):
            raise ValueError("checkpoint completed handoffs disagree with the plan")
        validate_finite_state(self.incoming_actor, "incoming_actor")
        validate_finite_state(self.selected_actors, "selected_actors")
        # Parsing the owned representation also validates the persisted policy schema.
        self.representation.policy
        for index, (result, weights) in enumerate(
            zip(self.results, self.selected_actors)
        ):
            phase = config.phases[index]
            require_equal(
                (index, phase.name, phase.algorithm, phase.handoff),
                (result.index, result.name, result.algorithm, result.handoff),
                "resume.completed_phase",
            )
            require_equal(
                result.selected_weight_hash,
                weight_fingerprint(weights),
                "resume.selected_actor.weights",
            )
            validate_finite_state(result.as_dict(), "completed_phase")
        incoming_identity = weight_fingerprint(self.incoming_actor)
        require_equal(
            self.results[-1].selected_weight_hash
            if self.results
            else self.initialization["weight_hash"],
            incoming_identity,
            "resume.incoming_actor",
        )
        state = self.phase_state
        if state is None:
            return
        if self.phase_index == len(config.phases):
            raise ValueError("completed run cannot have a live phase state")
        phase = config.phases[self.phase_index]
        if state["version"] != BUILTIN_ALGORITHMS[phase.algorithm].state_version:
            raise ValueError("unsupported phase checkpoint state version")
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
            or not 1 <= best <= transaction
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
        reference = getattr(phase, "reference", None)
        if (reference is None) != (state["reference"] is None):
            raise ValueError("checkpoint reference disagrees with the phase")
        if reference is not None:
            identity = weight_fingerprint(state["reference"])
            require_equal(
                identity, state["reference_identity"], "resume.embedded_reference"
            )
            expected = (
                incoming_identity
                if reference.source == "phase_start"
                else self.reference_contract[str(self.phase_index)]
            )
            require_equal(expected, identity, "resume.reference.weights")
        validate_finite_state(state, "phase")

    def validate_actor(self, actor, units):
        self.representation.validate_actor(actor, units, "resume.representation")
        expected = actor.network.state_dict()
        weights = [self.incoming_actor, *self.selected_actors]
        if self.phase_state is not None:
            state = self.phase_state
            weights += [
                state["actor"],
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
        atomic_save(
            path,
            {
                "artifact_type": "sequifier_policy_run",
                "format_version": 3,
                "plan": self.plan,
                "representation": self.representation.canonical,
                "representation_fingerprint": self.representation.identity,
                "data_identity": self.data_identity,
                "data_contract": self.data_contract,
                "adapter_contract": self.adapter_contract,
                "reference_contract": self.reference_contract,
                "initialization": self.initialization,
                "random": self.random,
                "incoming_actor": self.incoming_actor,
                "selected_actors": self.selected_actors,
                "completed_results": [result.as_dict() for result in self.results],
                "phase_state": self.phase_state,
            },
        )

    def finish_phase(self, actor, result):
        require_equal(self.phase_index, result.index, "handoff.phase_index")
        weights = clone_weights(actor.network)
        require_equal(
            result.selected_weight_hash,
            weight_fingerprint(weights),
            "handoff.selected_actor",
        )
        self.results.append(result)
        self.selected_actors.append(weights)
        self.incoming_actor = weights
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
