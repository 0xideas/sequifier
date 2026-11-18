"""Shared policy transactions, representation checks and phase result contracts."""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import torch

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.config.policy_plan import PhaseConfig
from sequifier.rl.models import HistoryCritic
from sequifier.rl.policy import Policy
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import metric_improved, select_handoff, stop_scope
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state
from sequifier.runtime.stopping import StoppingState

if TYPE_CHECKING:
    from sequifier.rl.run_config import ResolvedPolicyRun

PhaseType = TypeVar("PhaseType", bound=PhaseConfig)
DataType = TypeVar("DataType")
PHASE_STATE_VERSION = 4


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


def _legacy_critic_prefixes(module, interface_names):
    prefixes = []
    for path, critic in module.named_modules():
        if isinstance(critic, HistoryCritic):
            root = f"{path + '.' if path else ''}network.interfaces."
            active = root + critic.interface_name
            prefixes.extend((f"{active}.decoder.", f"{active}.prediction_scales."))
            prefixes.extend(
                f"{root}{name}."
                for name in interface_names
                if name != critic.interface_name
            )
    return tuple(prefixes)


def _prune_legacy_optimizer(state, optimizer, module, weights, saved_mask, mask):
    """Restore positional optimizer slots after dropping unused critic parameters."""
    groups = state["param_groups"]
    if len(groups) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError(
            "legacy critic optimizer recovery cannot remove parameters from a multi-group plan"
        )
    names = set(saved_mask)
    saved_order = [name for name in weights if name in names]
    group = groups[0]
    slots = group["params"]
    require_equal(len(names), len(saved_order), "legacy critic optimizer names")
    require_equal(len(saved_order), len(slots), "legacy critic optimizer slots")
    current_names = {id(p): name for name, p in module.named_parameters()}
    keep = set(mask)
    require_equal(
        [current_names[id(p)] for p in optimizer.param_groups[0]["params"]],
        [name for name in saved_order if name in keep],
        "legacy critic optimizer order",
    )
    dropped = {slot for name, slot in zip(saved_order, slots) if name not in keep}
    return {
        **state,
        "param_groups": [
            {**group, "params": [slot for slot in slots if slot not in dropped]}
        ],
        "state": {
            slot: value for slot, value in state["state"].items() if slot not in dropped
        },
    }


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
    stop_scope: str | None = None

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
    telemetry: TelemetryJournal


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
        if metric_improved(metric, self.best_metric, self.direction):
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

    def finish(self, context, transaction: int, clocks, provenance):
        actor = context.actor.eval()
        if self.phase.handoff == "best" and self.best_weights is None:
            raise ValueError(
                f"phases.{self.phase.name}: handoff best requires a finite {self.monitor}"
            )
        selected, metric = select_handoff(
            self.phase.handoff,
            (transaction, self.last_metric),
            (self.best_transaction, self.best_metric),
        )
        if selected is None:
            raise ValueError(f"{self.phase.name}: no selected transaction")
        if self.phase.handoff == "best":
            actor.network.load_state_dict(self.best_weights, strict=True)
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
            (),
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
        self.modules = {"actor": context.actor.network, **(modules or {})}
        self.mask_ids = {}
        for role, module in self.modules.items():
            eligible = {id(p) for p in module.parameters() if p.requires_grad}
            mask = context.phase.trainability.policy(role).resolve(module)
            if not eligible - mask:
                raise ValueError(
                    f"{context.phase.name}: {role} leaves no trainable parameters"
                )
            for parameter in module.parameters():
                if id(parameter) in mask:
                    parameter.requires_grad_(False)
            self.mask_ids[role] = sorted(
                name for name, p in module.named_parameters() if p.requires_grad
            )
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
            self._restore_networks_and_optimizers(continuation, "optimization")
        self.steps = dict.fromkeys(self.modules, 0)
        self.sampler = torch.Generator().manual_seed(context.run.seed + context.index)
        self.state = context.resume_state or {}
        self.transaction = self.state.get("transaction", 0)
        phase = context.phase
        self.iteration_budget = (
            phase.iterations
            if hasattr(phase, "iterations")
            else (phase.total_env_steps + phase.rollout_env_steps - 1)
            // phase.rollout_env_steps
            if hasattr(phase, "total_env_steps")
            else (phase.total_groups + phase.groups_per_iteration - 1)
            // phase.groups_per_iteration
            if hasattr(phase, "total_groups")
            else context.step_budget or 1
        )
        self.stopping = StoppingState(**self.state.get("stopping", {}))
        self.initial_evaluated = self.state.get("initial_evaluated", False)
        self._count_stopping = True
        self.last_checkpoint_time = time.monotonic()
        self.last_snapshot_time = self.last_checkpoint_time
        self.telemetry = context.telemetry
        selection = context.phase.evaluation.monitor
        monitor = selection.metric
        self.selection = Selection(
            context.phase, monitor, selection.direction, self.state.get("selection")
        )
        if self.state:
            self._restore_networks_and_optimizers(self.state, "phase")
            for role, saved in self.state["optimization"].items():
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

    def _restore_networks_and_optimizers(self, state, field):
        require_equal(
            sorted(self.networks), sorted(state["networks"]), f"{field}.networks"
        )
        require_equal(
            sorted(self.mask_ids),
            sorted(state["trainable_parameters"]),
            f"{field}.trainability roles",
        )
        require_equal(
            sorted(self.optimization),
            sorted(state["optimization"]),
            f"{field}.optimization",
        )
        interface_names = self.context.actor.execution_config.interfaces
        for role, module in self.networks.items():
            weights = state["networks"][role]
            prefixes = (
                _legacy_critic_prefixes(module, interface_names)
                if role != "actor"
                else ()
            )
            obsolete = (
                {name for name in weights if name.startswith(prefixes)}
                if prefixes
                else set()
            )
            module.load_state_dict(
                {name: value for name, value in weights.items() if name not in obsolete}
                if obsolete
                else weights,
                strict=True,
            )
            if role not in self.optimization:
                continue
            mask = self.mask_ids[role]
            saved_mask = state["trainable_parameters"][role]
            retained = [name for name in saved_mask if name not in obsolete]
            require_equal(mask, retained, f"{field}.trainability.{role}")
            optimizer, scheduler = self.optimization[role]
            saved = state["optimization"][role]
            optimizer_state = saved["optimizer"]
            if retained != saved_mask:
                optimizer_state = _prune_legacy_optimizer(
                    optimizer_state, optimizer, module, weights, saved_mask, mask
                )
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(saved["scheduler"])

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

    def snapshot_due(self):
        minutes = self.context.phase.checkpoints.snapshot_interval_minutes
        return (
            minutes is not None
            and time.monotonic() - self.last_snapshot_time >= minutes * 60
        )

    def should_evaluate_initial(self):
        due = self.context.phase.evaluation.before_phase and not self.initial_evaluated
        if due:
            self.initial_evaluated = True
            self._count_stopping = False
        return due

    def should_evaluate(self, final):
        scheduled = (
            final
            or self.transaction % self.context.phase.evaluation_interval_iterations == 0
        )
        self._count_stopping = scheduled
        return scheduled or (
            self.snapshot_due() and self.context.phase.evaluation.on_snapshot
        )

    def observe(self, metric, **record):
        if self.context.run._metrics_service is not None:
            values = {**record, self.selection.monitor: metric}
            self.context.run._metrics_service.record(
                self,
                values,
                validation=True,
                primary=self.selection.monitor,
                initial=self.transaction == 0,
            )
        self.selection.observe(self.context.actor, metric, self.transaction)
        if self._count_stopping:
            self.stopping.observe(
                self.context.phase.stopping,
                metric,
                self.selection.direction,
                self.transaction,
            )

    def record_evaluation(self, metric, **record):
        self.observe(metric, **record)
        self.telemetry.append("metrics", {"transaction": self.transaction, **record})

    def record_training(self, *, final=False, **record):
        self.telemetry.append("metrics", {"transaction": self.transaction, **record})
        if self.context.run._metrics_service is not None:
            self.context.run._metrics_service.record(self, record, final=final)

    def evaluate_validation(self, evaluator, **record):
        values = evaluator()
        key = self.selection.monitor.removeprefix("validation_")
        self.record_evaluation(
            values[key],
            **record,
            **{f"validation_{name}": value for name, value in values.items()},
        )

    def complete_update(
        self,
        *,
        final,
        training,
        evaluate,
        checkpoint=None,
        evaluation_record: dict | Callable[[], dict] | None = None,
    ):
        """Commit an offline update after all participating optimizers have stepped."""
        self.transaction += 1
        self.record_training(final=final, **training)
        if self.should_evaluate(final):
            extra = (
                evaluation_record()
                if callable(evaluation_record)
                else evaluation_record
            )
            evaluate(**training, **(extra or {}))
        self.checkpoint(final, **(checkpoint or {}))

    def optimizer_clocks(self):
        return {f"{role}_optimizer_steps": steps for role, steps in self.steps.items()}

    def record_continuation(self, **record):
        self.telemetry.append(
            "continuations", {"transaction": self.transaction, **record}
        )

    def evaluate_online(self, adapter, *, initial=False, **progress):
        from sequifier.rl.ppo_evaluation import evaluate_policy

        phase = self.context.phase
        result = evaluate_policy(
            self.context.actor, phase.evaluation, forbidden_adapter=adapter
        )
        if phase.algorithm == "grpo":
            score = float(
                torch.tensor(
                    result["score_mean"] * phase.reward_scale, dtype=torch.float32
                )
            )
            if not math.isfinite(score):
                raise ValueError("GRPO evaluation score exceeds finite FP32 range")
            self.observe(
                score, **{k: v for k, v in result.items() if k != "score_mean"}
            )
            result["score_mean"] = score
        else:
            self.observe(result["selection_value"], **result)
            if initial:
                progress["initial"] = True
        self.telemetry.append(
            "evaluations", {**result, "transaction": self.transaction, **progress}
        )

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
            or self.stopping.triggered
            or self.snapshot_due()
            or (
                self.context.run.latest_checkpoint_interval_minutes is not None
                and time.monotonic() - self.last_checkpoint_time
                >= self.context.run.latest_checkpoint_interval_minutes * 60
            )
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
                "stopping": asdict(self.stopping),
                "initial_evaluated": self.initial_evaluated,
                "trainable_parameters": self.mask_ids,
                "transaction": self.transaction,
                "telemetry_position": self.telemetry.commit(),
                "reference": clone_weights(self.context.reference.network)
                if self.context.reference is not None
                else None,
                "reference_identity": self.context.reference_identity,
                "random": capture_rng_state(self.context.actor.device),
            }
        )

        retained = (
            final
            or self.stopping.triggered
            or self.snapshot_due()
            or self.transaction % self.context.phase.checkpoint_interval_iterations == 0
        )
        if retained:
            from pathlib import Path
            from shutil import copyfile

            from sequifier.io.atomic import atomic_path

            destination = (
                Path(self.context.run.phase_directory)
                / f"{self.context.index:03d}-{self.context.phase.name}"
                / "checkpoints"
                / f"transaction-{self.transaction:08d}.pt"
            )
            with atomic_path(destination, durable=True) as temporary:
                copyfile(self.context.run.checkpoint_path, temporary)
        self.last_checkpoint_time = time.monotonic()
        if self.snapshot_due():
            self.last_snapshot_time = self.last_checkpoint_time

    def finish(self, clocks, provenance):
        if self.context.run._metrics_service is not None:
            self.context.run._metrics_service.flush(self)
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
                "trainable_parameters": self.mask_ids,
            }
        elif hasattr(self.context.actor, "_phase_continuation"):
            del self.context.actor._phase_continuation
        actor, result = self.selection.finish(
            self.context,
            self.transaction,
            clocks,
            {
                **provenance,
                "trainable_parameters": self.mask_ids,
                "resolved_seeds": {
                    "run": self.context.run.seed,
                    "phase_sampling": self.context.run.seed + self.context.index,
                },
            },
        )

        from dataclasses import replace

        return actor, replace(
            result,
            stop_scope=stop_scope(self.context.phase.stopping, self.stopping.triggered),
        )
