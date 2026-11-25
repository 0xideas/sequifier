"""Shared policy transactions, representation checks and phase result contracts."""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypedDict, TypeVar

import torch
from torch.distributed.tensor import DTensor

from sequifier.artifacts.model_artifact import ModelExecutionConfig
from sequifier.artifacts.reports import HandoffCandidate, PhaseResult
from sequifier.config.policy_plan import PhaseConfig
from sequifier.contract_versions import ContractVersions
from sequifier.fingerprints import fingerprint
from sequifier.rl.compilation import compile_critics
from sequifier.rl.phase_optimization import PhaseOptimization, PhaseRoles
from sequifier.rl.policy import Policy
from sequifier.rl.role_execution import ActionForward, InferenceRounds
from sequifier.rl.telemetry import TelemetryJournal
from sequifier.runtime.plan import metric_improved, stop_scope
from sequifier.runtime.random_state import capture_rng_state, restore_rng_state
from sequifier.runtime.stopping import StoppingState
from sequifier.training.distributed_strategy import (
    DistributedStrategy,
    FullyShardedStrategy,
    LocalStrategy,
    on_writer,
)

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
    global_step_offset: int = 0
    strategy: DistributedStrategy = field(default_factory=LocalStrategy)

    @property
    def index(self):
        return self.identity.index


class Selection:
    """Strict comparisons retain the earliest finite best on ties."""

    def __init__(self, phase, monitor, direction, state=None):
        self.phase, self.monitor, self.direction = phase, monitor, direction
        self.best = HandoffCandidate(**state["best"]) if state else HandoffCandidate()
        self.last_metric = state["last_metric"] if state else None

    def observe(self, actor, metric, transaction, capture=clone_weights):
        if metric is not None and not math.isfinite(metric):
            raise ValueError(f"non-finite selection metric {self.monitor}")
        self.last_metric = metric
        if metric is None:
            return
        if metric_improved(metric, self.best.metric, self.direction):
            self.best = HandoffCandidate(
                float(metric), transaction, capture(actor.network)
            )

    def state_dict(self):
        return {"best": vars(self.best), "last_metric": self.last_metric}

    def finish(
        self,
        context,
        transaction: int,
        clocks,
        provenance,
        scope,
        capture=clone_weights,
    ):
        actor = context.actor.eval()
        selected = self.best.select(
            self.phase,
            lambda: HandoffCandidate(
                self.last_metric, transaction, capture(actor.network)
            ),
        )
        weights = selected.require_weights()
        if self.phase.handoff == "best" and not isinstance(
            context.strategy, FullyShardedStrategy
        ):
            actor.network.load_state_dict(weights, strict=True)
        result = on_writer(
            context.strategy,
            lambda: PhaseResult.from_phase(
                context.index,
                self.phase,
                self.phase.evaluation.monitor,
                stop_scope=scope,
                selection=selected,
                clocks=clocks,
                provenance=provenance,
            ),
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
        self.strategy = context.strategy
        self.rounds = InferenceRounds(self.strategy)
        self.collection_seconds = self.update_seconds = self.checkpoint_seconds = 0.0
        roles = PhaseRoles.build(
            context.actor, context.phase, restoring=context.resume_state is not None
        )
        compile_critics(context.run.parent.run.execution.policy_compile_mode, roles)
        self.modules = roles.trainable
        self.networks = {**self.modules, **roles.frozen}
        self.mask_ids = context.optimization.masks
        self.configs = context.optimization.configs
        context.optimization.apply_masks(roles)
        saved = context.resume_state
        field = "phase"
        if saved is None and context.phase.optimizer_state == "continue":
            saved = context.continuation
            field = "optimization"
            if saved is None:
                raise ValueError(
                    "continued phase requires the preceding optimizer state"
                )
            require_equal(
                context.phase.algorithm, saved["algorithm"], "optimization.algorithm"
            )
        if saved is not None:
            require_equal(
                sorted(self.networks), sorted(saved["networks"]), f"{field}.networks"
            )
            for role, module in self.networks.items():
                module.load_state_dict(saved["networks"][role])
        self.calls = {}
        for role, module in self.networks.items():
            entry = ActionForward(context.actor) if role == "actor" else module
            prepare = (
                self.strategy.prepare_network
                if role in self.modules
                else self.strategy.prepare_frozen
            )
            self.calls[role] = self.rounds.bind(role, prepare(entry))
        context.actor.role_call = self.calls["actor"]
        if context.reference is not None:
            reference = self.strategy.prepare_frozen(ActionForward(context.reference))
            context.reference.role_call = self.rounds.bind("reference", reference)
        self.optimization = context.optimization.optimizers(roles)
        if saved is not None:
            self._restore_optimizers(saved, field)
        self.steps = dict.fromkeys(self.modules, 0)
        self.sampler = torch.Generator().manual_seed(context.identity.sampling_seed)
        self.state = context.resume_state or {}
        streams = self.state.get("rank_states")
        if streams is not None:
            if len(streams) != self.strategy.world_size:
                raise ValueError("Policy checkpoint requires its original world size")
            self.state = {**self.state, **streams[self.strategy.rank]}
        self.epoch: int | None = None
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

    def _restore_optimizers(self, state, field):
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
        for role, (optimizer, scheduler) in self.optimization.items():
            require_equal(
                self.mask_ids[role],
                state["trainable_parameters"][role],
                f"{field}.trainability.{role}",
            )
            saved = state["optimization"][role]
            self.strategy.restore_optimizer_state(
                self.modules[role], optimizer, saved["optimizer"]
            )
            scheduler.load_state_dict(saved["scheduler"])

    @property
    def global_step(self):
        return self.context.global_step_offset + self.transaction

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
                self.strategy.clip_gradients(
                    module, self.configs[role].gradient_clip_norm or float("inf")
                )
            )
            for role, module in self.modules.items()
        }
        if self.strategy.sum(
            int(any(not math.isfinite(norm) for norm in norms.values()))
        ):
            raise ValueError("Non-finite policy gradients")
        for optimizer, _ in self.optimization.values():
            optimizer.step()
        for role, module in self.modules.items():
            finite = all(
                torch.isfinite(
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                ).all()
                for parameter in module.parameters()
            )
            if self.strategy.sum(int(not finite)):
                raise ValueError(
                    f"{role} optimizer produced non-finite parameters; transaction was not committed"
                )
        for role, (_, scheduler) in self.optimization.items():
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
        timed, latest = self.strategy.broadcast((timed, latest))
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
        metric = self.strategy.broadcast(metric)
        self.selection.observe(
            self.context.actor, metric, self.transaction, self.capture_weights
        )
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
        with self.rounds.scope():
            values = evaluator()
        values = {key: self.strategy.sum(value) for key, value in values.items()}
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
    ):
        """Commit an offline update after all participating optimizers have stepped."""
        self.transaction += 1
        elapsed = time.monotonic() - self.iteration_started
        self.update_seconds += elapsed
        self.record_training(
            final=final,
            elapsed_seconds=elapsed,
            **training,
        )
        schedule = self.transaction_schedule(final)
        if schedule.evaluate:
            evaluate()
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
        with self.rounds.scope(defer_deadline=True):
            result = (
                evaluate_policy(
                    self.context.actor,
                    phase.evaluation,
                    forbidden_adapter=adapter,
                    expected_identity=self.context.adapter_contract[
                        f"{self.context.index}.evaluation"
                    ],
                )
                if self.strategy.rank == 0
                else None
            )
        result = self.strategy.broadcast(result)
        if adjust is not None:
            adjust(result, phase)
        record = {**result, "transaction": self.transaction, **progress}
        self.observe(
            result["selection_value"], elapsed_seconds=result["seconds"], **record
        )
        self.telemetry.append("evaluations", record)

    def capture_weights(self, module):
        return self.strategy.capture_model_state(module)

    def partition(self, indices, offset=0):
        """A valid forward on an empty rank contributes zero to the global sum."""
        first = (self.strategy.rank - offset) % self.strategy.world_size
        local = indices[first :: self.strategy.world_size]
        return (local, self.strategy.world_size) if len(local) else (indices[:1], 0)

    def optimization_state(self, *, include_steps=False):
        return {
            role: {
                "optimizer": self.strategy.capture_optimizer_state(
                    self.modules[role], optimizer
                ),
                "scheduler": scheduler.state_dict(),
                **({"steps": self.steps[role]} if include_steps else {}),
            }
            for role, (optimizer, scheduler) in self.optimization.items()
        }

    def checkpoint(self, schedule, payload=None):
        # A scheduled evaluation may just have triggered early stopping.
        if not (schedule.checkpoint or self.stopping.triggered):
            return
        started = time.monotonic()
        extra = payload() if payload is not None else {}
        random = capture_rng_state(self.context.actor.device)
        local = {
            key: extra[key]
            for key in ("collector", "environment_state")
            if key in extra
        }
        rank_states = (
            self.strategy.gather_for_writer({**local, "random": random})
            if self.strategy.world_size > 1
            else None
        )
        self.context.commit(
            {
                **extra,
                **({"rank_states": rank_states} if rank_states is not None else {}),
                "version": ContractVersions.POLICY_PHASE_STATE,
                "networks": {
                    name: self.capture_weights(module)
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
                "reference": self.capture_weights(self.context.reference.network)
                if self.context.reference is not None
                else None,
                "reference_identity": self.context.reference_identity,
                "random": random,
            }
        )

        if self.strategy.rank == 0 and (
            schedule.retain_snapshot or self.stopping.triggered
        ):
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
        self.checkpoint_seconds += time.monotonic() - started
        self.last_checkpoint_time = schedule.timestamp
        if schedule.timed_snapshot:
            self.last_snapshot_time = schedule.timestamp

    def finish(self, clocks, provenance):
        if self.context.metrics is not None:
            self.context.metrics.flush(self)
        clocks = {**clocks, "global_step": self.global_step}
        scope = stop_scope(self.context.phase.stopping, self.stopping.triggered)
        if self.strategy.world_size > 1:
            provenance = {
                **provenance,
                "session_seconds_by_rank": self.strategy.gather_for_writer(
                    {
                        "collection": self.collection_seconds,
                        "update": self.update_seconds,
                        "checkpoint": self.checkpoint_seconds,
                        "scoring_forward_host": self.rounds.scoring_seconds,
                    }
                ),
            }
        actor, weights, result = self.selection.finish(
            self.context,
            self.transaction,
            clocks,
            {
                **provenance,
                "model_interface": self.context.actor.interface_name,
                "trainable_parameters": self.mask_ids,
                "resolved_seeds": {
                    "run": self.context.identity.run_seed,
                    "phase_sampling": self.context.identity.sampling_seed,
                },
            },
            scope,
            self.capture_weights,
        )
        continuation: OptimizerContinuation | None = None
        if self.context.preserve_optimizer and scope != "end_plan":
            continuation = {
                "algorithm": self.context.phase.algorithm,
                "networks": {
                    name: weights if name == "actor" else self.capture_weights(module)
                    for name, module in self.networks.items()
                },
                "optimization": self.optimization_state(),
                "trainable_parameters": self.mask_ids,
            }
        self.rounds.modules.clear()
        actor.role_call = None
        if self.context.reference is not None:
            self.context.reference.role_call = None
        if isinstance(self.strategy, FullyShardedStrategy):
            # Leave FSDP wrappers and phase-owned optimizers behind at the handoff.
            from sequifier.rl.preparation import build_actor

            random = capture_rng_state(self.strategy.device)
            sharded_actor = actor
            actor = build_actor(
                self.context.run, device=torch.device("cpu"), initialize=False
            )
            actor.transfer_report = sharded_actor.transfer_report
            actor.ppo_policy_version = sharded_actor.ppo_policy_version
            actor.grpo_policy_version = sharded_actor.grpo_policy_version
            if self.strategy.rank == 0:
                actor.network.load_state_dict(weights)
            restore_rng_state(self.strategy.device, random)
        return PhaseExecution(actor, weights, result, continuation)
