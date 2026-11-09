"""Shared policy transactions, representation checks and phase result contracts."""

import hashlib
import json
import math
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.config.policy_optimization import OptimizationPlan
from sequifier.config.policy_plan import PhaseConfig
from sequifier.rl.policy import CategoricalPolicy
from sequifier.runtime.random_state import RandomState, RandomStateManager

if TYPE_CHECKING:
    from sequifier.config.train_config import PolicyRunConfig

PhaseType = TypeVar("PhaseType", bound=PhaseConfig)
DataType = TypeVar("DataType")


def clone_weights(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def weight_fingerprint(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
        sync_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        sync_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def sync_file(path):
    """Flush an atomically published file before committing a reference to it."""
    path = Path(path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def capture_rng_state(device):
    return {
        "local": RandomStateManager(device).capture_local().state_dict(),
        "cuda_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(device, state):
    RandomStateManager(device).restore(RandomState.from_state_dict(state["local"]))
    if state["cuda_all"] is not None:
        if (
            not torch.cuda.is_available()
            or len(state["cuda_all"]) != torch.cuda.device_count()
        ):
            raise ValueError("policy checkpoint CUDA RNG topology differs")
        torch.cuda.set_rng_state_all(state["cuda_all"])
    if state["mps"] is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("policy checkpoint requires the saved MPS RNG")
        torch.mps.set_rng_state(state["mps"])


def require_equal(expected, actual, field):
    """Locate the first mismatching contract field rather than a generic failure."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(
                f"{field}: expected fields {sorted(expected)}, received {sorted(actual)}"
            )
        for key in expected:
            require_equal(expected[key], actual[key], f"{field}.{key}")
    elif expected != actual:
        raise ValueError(f"{field}: expected {expected!r}, received {actual!r}")


@dataclass(frozen=True)
class ActorRepresentation:
    # Canonical JSON provides an immutable owned value, even for nested schemas.
    canonical: str

    @classmethod
    def from_actor(cls, actor, units):
        return cls(
            json.dumps(
                {
                    "policy": actor.config.model_dump(mode="json"),
                    "model": actor.execution_config.model.model_dump(mode="json"),
                    "statistics": actor.interface.selected_columns_statistics,
                    "units": dict(getattr(actor, "units", units)),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )

    @property
    def values(self):
        return json.loads(self.canonical)

    @property
    def policy(self):
        return CategoricalPolicyConfig.model_validate(self.values["policy"])

    @property
    def identity(self):
        return hashlib.sha256(self.canonical.encode()).hexdigest()

    def validate_actor(self, actor, units, field="actor"):
        require_equal(self.values, type(self).from_actor(actor, units).values, field)


@dataclass(frozen=True)
class PolicyPhaseResult:
    index: int
    name: str
    algorithm: str
    monitor: str
    direction: str
    handoff: str
    selected_metric: float | None
    selected_transaction: int
    selected_weight_hash: str
    clocks: dict
    metrics: tuple[dict, ...]
    provenance: dict

    def as_dict(self):
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class PolicyRunResult:
    policy: CategoricalPolicy
    phases: tuple[PolicyPhaseResult, ...]
    representation: ActorRepresentation
    model_path: str
    report_path: str
    checkpoint_path: str


@dataclass(frozen=True)
class PhaseContext(Generic[PhaseType, DataType]):
    run: "PolicyRunConfig"
    index: int
    phase: PhaseType
    actor: CategoricalPolicy
    representation: ActorRepresentation
    data: DataType
    reference: CategoricalPolicy | None
    reference_identity: str | None
    resume_state: dict | None
    commit: Callable[[dict], None]
    optimization: dict[str, OptimizationPlan]


class Selection:
    """Strict comparisons retain the earliest finite best on ties."""

    def __init__(self, phase, monitor, direction, state=None):
        self.phase, self.monitor, self.direction = phase, monitor, direction
        self.best_metric, self.best_weights, self.best_transaction = None, None, None
        self.last_metric = None
        if state:
            self.best_metric = state["best_metric"]
            self.best_weights = state["best_weights"]
            self.best_transaction = state["best_transaction"]
            self.last_metric = state["last_metric"]

    def observe(self, actor, metric, transaction):
        if metric is not None and not math.isfinite(metric):
            raise ValueError(f"non-finite selection metric {self.monitor}")
        self.last_metric = metric
        if metric is None:
            return
        if self.best_metric is None or (
            metric < self.best_metric
            if self.direction == "min"
            else metric > self.best_metric
        ):
            self.best_metric = float(metric)
            self.best_weights = clone_weights(actor.network)
            self.best_transaction = transaction

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "best_metric",
                "best_weights",
                "best_transaction",
                "last_metric",
            )
        }

    def finish(self, context, transaction, clocks, metrics, provenance):
        actor = deepcopy(context.actor).eval()
        if self.phase.handoff == "best":
            if self.best_weights is None:
                raise ValueError(
                    f"phases.{self.phase.name}: handoff best requires a finite {self.monitor}"
                )
            actor.network.load_state_dict(self.best_weights, strict=True)
            if self.best_transaction is None:
                raise ValueError("best actor has no selected transaction")
            selected, metric = self.best_transaction, self.best_metric
        else:
            selected, metric = transaction, self.last_metric
        result = PolicyPhaseResult(
            context.index,
            self.phase.name,
            self.phase.algorithm,
            self.monitor,
            self.direction,
            self.phase.handoff,
            metric,
            selected,
            weight_fingerprint(actor.network.state_dict()),
            clocks,
            tuple(metrics),
            provenance,
        )
        return actor, result


def apply_gradients(module, optimizer, scheduler, clip):
    torch.nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in module.parameters()):
        raise ValueError(
            "policy optimizer produced non-finite parameters; transaction was not committed"
        )
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def phase_events(phase, transaction, *, final):
    """Evaluation and checkpointing use the committed transaction clock."""
    return (
        final or transaction % phase.evaluation_interval == 0,
        final or transaction % phase.checkpoint_interval == 0,
    )


def actor_state(
    context, optimizer, scheduler, selection, transaction, metrics, **extra
):
    return {
        "version": 1,
        "actor": clone_weights(context.actor.network),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "selection": selection.state_dict(),
        "transaction": transaction,
        "metrics": metrics,
        "reference": clone_weights(context.reference.network)
        if context.reference is not None
        else None,
        "reference_identity": context.reference_identity,
        "random": capture_rng_state(context.actor.device),
        **extra,
    }


def restore_actor_state(context, optimizer, scheduler):
    state = context.resume_state
    if state is None:
        return
    context.actor.network.load_state_dict(state["actor"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(context.actor.device, state["random"])
