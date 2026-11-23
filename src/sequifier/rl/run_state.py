"""Shared policy transactions, representation checks and phase result contracts."""

import math
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypedDict, TypeVar

import torch

from sequifier.artifacts.model_artifact import ModelExecutionConfig
from sequifier.artifacts.reports import HandoffCandidate, PhaseResult
from sequifier.config.policy_plan import PhaseConfig
from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import fingerprint
from sequifier.rl.phase_optimization import PhaseOptimization, PhaseRoles
from sequifier.rl.policy import Policy
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import metric_improved, stop_scope
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state
from sequifier.runtime.stopping import StoppingState

if TYPE_CHECKING:
    from sequifier.rl.metrics import PolicyMetrics
    from sequifier.rl.run_config import ResolvedPolicyRun

PhaseType = TypeVar("PhaseType", bound=PhaseConfig)
DataType = TypeVar("DataType")


def clone_weights(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


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
    execution: ModelExecutionConfig
    interface: str

    @classmethod
    def from_actor(cls, actor):
        return cls(
            ModelExecutionConfig.from_training_config(actor.execution_config),
            actor.interface_name,
        )

    @property
    def identity(self):
        return fingerprint((self.interface, self.execution.identity))

    def validate_actor(self, actor, field="actor"):
        actual = type(self).from_actor(actor)
        require_equal(self.identity, actual.identity, field)


@dataclass(frozen=True)
class PolicyRunResult:
    policy: Policy
    phases: tuple[PhaseResult, ...]
    representation: ActorRepresentation
    model_path: str
    report_path: str
    checkpoint_path: str


class OptimizerContinuation(TypedDict):
    algorithm: str
    networks: dict[str, dict[str, torch.Tensor]]
    optimization: dict[str, dict]
    trainable_parameters: dict[str, list[str]]


@dataclass(frozen=True)
class PhaseExecution:
    actor: Policy
    weights: dict[str, torch.Tensor]
    result: PhaseResult
    continuation: OptimizerContinuation | None


@dataclass(frozen=True)
class PolicyPhaseIdentity:
    index: int
    run_seed: int
    sampling_seed: int

    @classmethod
    def resolve(cls, index: int, run_seed: int):
        return cls(index, run_seed, run_seed + index)

    def grpo_seed(self, restart):
        from sequifier.rl.collection import stream_seed

        return stream_seed(
            self.run_seed, f"grpo-phase-{self.index}-restart-{restart}", 0
        )


@dataclass(frozen=True)
class PhaseContext(Generic[PhaseType, DataType]):
    run: "ResolvedPolicyRun"
    identity: PolicyPhaseIdentity
    phase: PhaseType
    actor: Policy
    data: DataType
    reference: Policy | None
    reference_identity: str | None
    resume_state: dict | None
    commit: Callable[[dict], None]
    optimization: PhaseOptimization
    telemetry: TelemetryJournal
    adapter_contract: dict
    metrics: "PolicyMetrics | None" = None
    continuation: OptimizerContinuation | None = None
    preserve_optimizer: bool = False

    @property
    def index(self):
        return self.identity.index


class Selection:
    """Strict comparisons retain the earliest finite best on ties."""

    def __init__(self, phase, monitor, direction, state=None):
        self.phase, self.monitor, self.direction = phase, monitor, direction
        self.best = HandoffCandidate(**state["best"]) if state else HandoffCandidate()
        self.last_metric = state["last_metric"] if state else None

    def observe(self, actor, metric, transaction):
        if metric is not None and not math.isfinite(metric):
            raise ValueError(f"non-finite selection metric {self.monitor}")
        self.last_metric = metric
        if metric is None:
            return
        if metric_improved(metric, self.best.metric, self.direction):
            self.best = HandoffCandidate(
                float(metric), transaction, clone_weights(actor.network)
            )

    def state_dict(self):
        return {"best": vars(self.best), "last_metric": self.last_metric}

    def finish(self, context, transaction: int, clocks, provenance, scope):
        actor = context.actor.eval()
        selected = self.best.select(
            self.phase,
            lambda: HandoffCandidate(
                self.last_metric, transaction, clone_weights(actor.network)
            ),
        )
        weights = selected.require_weights()
        if self.phase.handoff == "best":
            actor.network.load_state_dict(weights, strict=True)
        result = PhaseResult.from_phase(
            context.index,
            self.phase,
            self.phase.evaluation.monitor,
            stop_scope=scope,
            selection=selected,
            clocks=clocks,
            provenance=provenance,
        )
        return actor, weights, result


@dataclass(frozen=True)
class TransactionSchedule:
    timestamp: float
    evaluate: bool
    checkpoint: bool
    retain_snapshot: bool
    timed_snapshot: bool


class PhaseSession:
    """Own common progress and optimization; algorithms own their update loops."""

    def __init__(self, context):
        self.context = context
        roles = PhaseRoles.build(
            context.actor, context.phase, restoring=context.resume_state is not None
        )
        self.modules = roles.trainable
        self.networks = {**self.modules, **roles.frozen}
        self.mask_ids = context.optimization.masks
        self.configs = context.optimization.configs
        self.optimization = context.optimization.build(roles)
        if context.phase.optimizer_state == "continue" and context.resume_state is None:
            continuation = context.continuation
            if continuation is None:
                raise ValueError(
                    "continued phase requires the preceding optimizer state"
                )
            require_equal(
                context.phase.algorithm,
                continuation["algorithm"],
                "optimization.algorithm",
            )
            self._restore_networks_and_optimizers(continuation, "optimization")
        self.steps = dict.fromkeys(self.modules, 0)
        self.sampler = torch.Generator().manual_seed(context.identity.sampling_seed)
        self.state = context.resume_state or {}
        self.transaction = self.state.get("transaction", 0)
        self.iteration_budget = context.phase.iteration_budget(
            context.optimization.step_budget
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
                budget = context.optimization.step_budget
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
        for role, module in self.networks.items():
            module.load_state_dict(state["networks"][role], strict=True)
            if role in self.optimization:
                require_equal(
                    self.mask_ids[role],
                    state["trainable_parameters"][role],
                    f"{field}.trainability.{role}",
                )
                optimizer, scheduler = self.optimization[role]
                saved = state["optimization"][role]
                optimizer.load_state_dict(saved["optimizer"])
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

    def evaluate_initial(self, evaluate, *args, **kwargs):
        if self.context.phase.evaluation.before_phase and not self.initial_evaluated:
            self._count_stopping = False
            evaluate(*args, initial=True, **kwargs)
            self.initial_evaluated = True
        self.iteration_started = time.monotonic()

    def transaction_schedule(self, final):
        phase = self.context.phase
        now = time.monotonic()
        snapshot_minutes = phase.checkpoints.snapshot_interval_minutes
        latest_minutes = (
            self.context.run.parent.run.recovery.latest_checkpoint_interval_minutes
        )
        timed = (
            snapshot_minutes is not None
            and now - self.last_snapshot_time >= snapshot_minutes * 60
        )
        latest = (
            latest_minutes is not None
            and now - self.last_checkpoint_time >= latest_minutes * 60
        )
        retain = (
            final
            or timed
            or self.transaction % phase.checkpoint_interval_iterations == 0
        )
        scheduled_evaluation = (
            final or self.transaction % phase.evaluation_interval_iterations == 0
        )
        self._count_stopping = scheduled_evaluation
        return TransactionSchedule(
            timestamp=now,
            evaluate=scheduled_evaluation or (retain and phase.evaluation.on_snapshot),
            checkpoint=retain or latest,
            retain_snapshot=retain,
            timed_snapshot=timed,
        )

    def observe(self, metric, *, elapsed_seconds, **record):
        if self.context.metrics is not None:
            values = {**record, self.selection.monitor: metric}
            self.context.metrics.record(
                self,
                values,
                elapsed_seconds=elapsed_seconds,
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

    def record_evaluation(self, metric, *, elapsed_seconds, **record):
        self.observe(metric, elapsed_seconds=elapsed_seconds, **record)
        self.telemetry.append("metrics", {"transaction": self.transaction, **record})

    def record_training(self, *, elapsed_seconds, final=False, **record):
        self.telemetry.append("metrics", {"transaction": self.transaction, **record})
        if self.context.metrics is not None:
            self.context.metrics.record(
                self, record, elapsed_seconds=elapsed_seconds, final=final
            )

    def evaluate_validation(self, evaluator, **record):
        started = time.monotonic()
        values = evaluator()
        key = self.selection.monitor.removeprefix("validation_")
        self.record_evaluation(
            values[key],
            elapsed_seconds=time.monotonic() - started,
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
        self.record_training(
            final=final,
            elapsed_seconds=time.monotonic() - self.iteration_started,
            **training,
        )
        schedule = self.transaction_schedule(final)
        if schedule.evaluate:
            extra = (
                evaluation_record()
                if callable(evaluation_record)
                else evaluation_record
            )
            evaluate(**training, **(extra or {}))
        self.checkpoint(schedule, checkpoint)
        self.iteration_started = time.monotonic()

    def complete_online_update(
        self, *, final, training: dict | Callable[[], dict], evaluate, checkpoint
    ):
        """Evaluate the updated policy before recording collection metrics and recovery."""
        self.transaction += 1
        elapsed_seconds = time.monotonic() - self.iteration_started
        schedule = self.transaction_schedule(final)
        if schedule.evaluate:
            evaluate()
        metrics = training() if callable(training) else training
        self.record_training(
            final=final,
            elapsed_seconds=elapsed_seconds,
            transaction=self.transaction,
            **metrics,
        )
        self.checkpoint(schedule, checkpoint)
        self.iteration_started = time.monotonic()

    def optimizer_clocks(self):
        return {f"{role}_optimizer_steps": steps for role, steps in self.steps.items()}

    def record_continuation(self, **record):
        self.telemetry.append(
            "continuations", {"transaction": self.transaction, **record}
        )

    def evaluate_online(self, adapter, *, adjust=None, **progress):
        from sequifier.rl.ppo_evaluation import evaluate_policy

        phase = self.context.phase
        result = evaluate_policy(
            self.context.actor,
            phase.evaluation,
            forbidden_adapter=adapter,
            expected_identity=self.context.adapter_contract[
                f"{self.context.index}.evaluation"
            ],
        )
        if adjust is not None:
            adjust(result, phase)
        record = {**result, "transaction": self.transaction, **progress}
        self.observe(
            result["selection_value"], elapsed_seconds=result["seconds"], **record
        )
        self.telemetry.append("evaluations", record)

    def optimization_state(self, *, include_steps=False):
        return {
            role: {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                **({"steps": self.steps[role]} if include_steps else {}),
            }
            for role, (optimizer, scheduler) in self.optimization.items()
        }

    def checkpoint(self, schedule, payload=None):
        # A scheduled evaluation may just have triggered early stopping.
        if not (schedule.checkpoint or self.stopping.triggered):
            return
        extra = payload() if payload is not None else {}
        self.context.commit(
            {
                **extra,
                "version": ContractVersions.POLICY_PHASE_STATE,
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

        if schedule.retain_snapshot or self.stopping.triggered:
            from shutil import copyfile

            from sequifier.artifacts.phase import PhaseArtifacts
            from sequifier.io.atomic import atomic_path

            destination = (
                PhaseArtifacts.for_phase(
                    self.context.run.outputs.phase_directory,
                    self.context.index,
                    self.context.phase.name,
                ).snapshots
                / f"transaction-{self.transaction:08d}.pt"
            )
            with atomic_path(destination, durable=True) as temporary:
                copyfile(self.context.run.checkpoint_path, temporary)
        self.last_checkpoint_time = schedule.timestamp
        if schedule.timed_snapshot:
            self.last_snapshot_time = schedule.timestamp

    def finish(self, clocks, provenance):
        if self.context.metrics is not None:
            self.context.metrics.flush(self)
        scope = stop_scope(self.context.phase.stopping, self.stopping.triggered)
        actor, weights, result = self.selection.finish(
            self.context,
            self.transaction,
            clocks,
            {
                **provenance,
                "trainable_parameters": self.mask_ids,
                "resolved_seeds": {
                    "run": self.context.identity.run_seed,
                    "phase_sampling": self.context.identity.sampling_seed,
                },
            },
            scope,
        )
        continuation: OptimizerContinuation | None = None
        if self.context.preserve_optimizer and scope != "end_plan":
            continuation = {
                "algorithm": self.context.phase.algorithm,
                "networks": {
                    name: weights if name == "actor" else clone_weights(module)
                    for name, module in self.networks.items()
                },
                "optimization": self.optimization_state(),
                "trainable_parameters": self.mask_ids,
            }
        return PhaseExecution(actor, weights, result, continuation)
