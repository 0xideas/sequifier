"""Shared policy transactions, representation checks and phase result contracts."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.config.policy_optimization import OptimizationPlan
from sequifier.config.policy_plan import PhaseConfig
from sequifier.rl.policy import CategoricalPolicy
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state

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


class PhaseSession:
    """Own common progress and optimization; algorithms own their update loops."""

    def __init__(self, context, *, modules=None, frozen=None):
        self.context = context
        self.modules = {"actor": context.actor.network, **(modules or {})}
        self.networks = {**self.modules, **(frozen or {})}
        if self.modules.keys() != context.optimization.keys():
            raise ValueError("phase optimization roles disagree with trainable modules")
        self.optimization = {
            role: context.optimization[role].build(module)
            for role, module in self.modules.items()
        }
        self.steps = dict.fromkeys(self.modules, 0)
        self.sampler = torch.Generator().manual_seed(context.run.seed + context.index)
        self.state = context.resume_state or {}
        self.transaction = self.state.get("transaction", 0)
        self.metrics = list(self.state.get("metrics", []))
        selection = getattr(context.phase, "evaluation", context.phase)
        monitor = getattr(selection, "metric", getattr(selection, "monitor", None))
        self.selection = Selection(
            context.phase, monitor, selection.direction, self.state.get("selection")
        )
        if self.state:
            require_equal(
                sorted(self.networks), sorted(self.state["networks"]), "phase.networks"
            )
            require_equal(
                sorted(self.optimization),
                sorted(self.state["optimization"]),
                "phase.optimization",
            )
            for name, module in self.networks.items():
                module.load_state_dict(self.state["networks"][name], strict=True)
            for role, (optimizer, scheduler) in self.optimization.items():
                saved = self.state["optimization"][role]
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                steps = saved["steps"]
                budget = context.optimization[role].step_budget
                if (
                    (not isinstance(steps, int))
                    or steps < self.transaction
                    or (budget is not None and steps > budget)
                ):
                    raise ValueError(
                        f"{role} optimizer clock exceeds committed progress/budget"
                    )
                self.steps[role] = steps
            self.sampler.set_state(self.state["sampler"])
            self.restore_random()

    def restore_random(self):
        if self.state:
            restore_rng_state(self.context.actor.device, self.state["random"])

    def zero_grad(self):
        for optimizer, _ in self.optimization.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        # Check every gradient before mutating any participating network.
        norms = {
            role: float(
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(),
                    self.context.optimization[role].gradient_clip,
                    error_if_nonfinite=True,
                )
            )
            for role, module in self.modules.items()
        }
        for role, (optimizer, scheduler) in self.optimization.items():
            optimizer.step()
            if any(
                not torch.isfinite(p).all() for p in self.modules[role].parameters()
            ):
                raise ValueError(
                    f"{role} optimizer produced non-finite parameters; transaction was not committed"
                )
            scheduler.step()
            self.steps[role] += 1
        self.zero_grad()
        return norms

    def should_evaluate(self, final):
        return final or self.transaction % self.context.phase.evaluation_interval == 0

    def observe(self, metric):
        self.selection.observe(self.context.actor, metric, self.transaction)

    def should_checkpoint(self, final):
        return final or self.transaction % self.context.phase.checkpoint_interval == 0

    def checkpoint(self, final, **extra):
        if not self.should_checkpoint(final):
            return
        self.context.commit(
            {
                **extra,
                "version": 2,
                "networks": {
                    name: clone_weights(module)
                    for name, module in self.networks.items()
                },
                "optimization": {
                    role: {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": self.steps[role],
                    }
                    for role, (optimizer, scheduler) in self.optimization.items()
                },
                "sampler": self.sampler.get_state(),
                "selection": self.selection.state_dict(),
                "transaction": self.transaction,
                "metrics": self.metrics,
                "reference": clone_weights(self.context.reference.network)
                if self.context.reference is not None
                else None,
                "reference_identity": self.context.reference_identity,
                "random": capture_rng_state(self.context.actor.device),
            }
        )

    def finish(self, clocks, provenance):
        return self.selection.finish(
            self.context, self.transaction, clocks, self.metrics, provenance
        )
