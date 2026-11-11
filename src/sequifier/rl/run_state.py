"""Shared policy transactions, representation checks and phase result contracts."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import torch

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_plan import PhaseConfig
from sequifier.rl.policy import Policy
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
    from sequifier.rl.run_config import ResolvedPolicyRun

PhaseType = TypeVar("PhaseType", bound=PhaseConfig)
DataType = TypeVar("DataType")
PHASE_STATE_VERSION = 2


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
        from pydantic_core import to_jsonable_python

        from sequifier.artifacts.model_export import model_execution_config

        return cls(
            json.dumps(
                {
                    "execution": to_jsonable_python(
                        model_execution_config(actor.execution_config)
                    ),
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
        return ResolvedPolicyBinding.model_validate(self.values["policy"])

    @property
    def identity(self):
        return hashlib.sha256(self.canonical.encode()).hexdigest()

    def validate_actor(self, actor, units, field="actor"):
        actual = type(self).from_actor(actor, units).values
        if "execution" not in self.values:
            actual.pop("execution")
        require_equal(self.values, actual, field)


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
    policy: Policy
    phases: tuple[PolicyPhaseResult, ...]
    representation: ActorRepresentation
    model_path: str
    report_path: str
    checkpoint_path: str


@dataclass(frozen=True)
class PhaseContext(Generic[PhaseType, DataType]):
    run: "ResolvedPolicyRun"
    index: int
    phase: PhaseType
    actor: Policy
    data: DataType
    reference: Policy | None
    reference_identity: str | None
    resume_state: dict | None
    commit: Callable[[dict], None]
    step_budget: int | None


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
        actor = context.actor.eval()
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
        actor = context.actor
        if not hasattr(actor, "_phase_trainability"):
            actor._phase_trainability = {
                name: p.requires_grad for name, p in actor.network.named_parameters()
            }
        for name, parameter in actor.network.named_parameters():
            parameter.requires_grad_(actor._phase_trainability[name])
        for path in context.phase.freeze_modules:
            actor.network.get_submodule(path).requires_grad_(False)
        self.modules = {"actor": context.actor.network, **(modules or {})}
        self.networks = {**self.modules, **(frozen or {})}
        self.configs = {
            role: config.resolve_budget(context.step_budget)
            for role, config in context.phase.optimization_configs().items()
        }
        if self.modules.keys() != self.configs.keys():
            raise ValueError("phase optimization roles disagree with trainable modules")
        self.optimization = {
            role: self.configs[role].build(module, step_budget=context.step_budget)
            for role, module in self.modules.items()
        }
        continuation = getattr(actor, "_phase_continuation", None)
        if (
            context.phase.optimizer_state == "continue"
            and not context.resume_state
            and continuation is None
        ):
            raise ValueError("continued phase requires the preceding optimizer state")
        if (
            context.phase.optimizer_state == "continue"
            and not context.resume_state
            and continuation
        ):
            require_equal(
                context.phase.algorithm,
                continuation["algorithm"],
                "optimization.algorithm",
            )
            require_equal(
                sorted(self.networks),
                sorted(continuation["networks"]),
                "optimization.roles",
            )
            for name, module in self.networks.items():
                module.load_state_dict(continuation["networks"][name], strict=True)
            for role, (optimizer, scheduler) in self.optimization.items():
                optimizer.load_state_dict(
                    continuation["optimization"][role]["optimizer"]
                )
                scheduler.load_state_dict(
                    continuation["optimization"][role]["scheduler"]
                )
        self.steps = dict.fromkeys(self.modules, 0)
        self.sampler = torch.Generator().manual_seed(context.run.seed + context.index)
        self.state = context.resume_state or {}
        self.transaction = self.state.get("transaction", 0)
        self.metrics = list(self.state.get("metrics", []))
        selection = context.phase.evaluation.monitor
        monitor = selection.metric
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
                budget = context.step_budget
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
                    self.configs[role].gradient_clip_norm or float("inf"),
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
        return (
            final
            or self.transaction % self.context.phase.evaluation_interval_iterations == 0
        )

    def observe(self, metric):
        self.selection.observe(self.context.actor, metric, self.transaction)

    def record_evaluation(self, metric, **record):
        self.observe(metric)
        self.metrics.append({"transaction": self.transaction, **record})

    def optimization_state(self, *, include_steps=False):
        return {
            role: {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                **({"steps": self.steps[role]} if include_steps else {}),
            }
            for role, (optimizer, scheduler) in self.optimization.items()
        }

    def should_checkpoint(self, final):
        return (
            final
            or self.transaction % self.context.phase.checkpoint_interval_iterations == 0
        )

    def checkpoint(self, final, **extra):
        if not self.should_checkpoint(final):
            return
        self.context.commit(
            {
                **extra,
                "version": PHASE_STATE_VERSION,
                "networks": {
                    name: clone_weights(module)
                    for name, module in self.networks.items()
                },
                "optimization": self.optimization_state(include_steps=True),
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
        next_index = self.context.index + 1
        preserve = (
            next_index < len(self.context.run.phases)
            and self.context.run.phases[next_index].optimizer_state == "continue"
        )
        if preserve:
            self.context.actor._phase_continuation = {
                "algorithm": self.context.phase.algorithm,
                "networks": {
                    name: clone_weights(module)
                    for name, module in self.networks.items()
                },
                "optimization": self.optimization_state(),
            }
        elif hasattr(self.context.actor, "_phase_continuation"):
            del self.context.actor._phase_continuation
        return self.selection.finish(
            self.context, self.transaction, clocks, self.metrics, provenance
        )
