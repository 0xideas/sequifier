"""Training orchestration over an explicitly composed :class:`TrainingRun`."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sequifier.artifacts.backbone_repository import publish_revision
from sequifier.artifacts.manifests import write_manifest
from sequifier.artifacts.state_dict import canonicalize_state_dict
from sequifier.evaluation.service import EvaluationContext, EvaluationResult
from sequifier.export.service import ExportOptions
from sequifier.integration.contexts import (
    BatchPrepared,
    ForwardCompleted,
    LossComputed,
    ModelReady,
    RunCompleted,
    StepIdentity,
    ValidationCompleted,
)
from sequifier.logging_paths import model_log_directory
from sequifier.training.checkpoint_service import CheckpointRequest
from sequifier.training.metrics_service import TrainingMetrics
from sequifier.training.optimization import UpdatePolicy
from sequifier.training.runtime import SourceScheduler, build_source_runtime


@dataclass(frozen=True)
class RunResult:
    completion_reason: str
    exports: tuple[Path, ...]
    publication: dict[str, Any]


class TrainingEngine:
    """Coordinate run services without owning model or optimizer mechanics."""

    def _identity(
        self, run: Any, epoch: int, batch: int, accumulation_index: int
    ) -> StepIdentity:
        return StepIdentity(
            epoch=epoch,
            batch=batch,
            global_batch_step=run.state.global_batch_step + 1,
            optimizer_step=run.optimization.optimizer_step,
            accumulation_index=accumulation_index,
            accumulation_steps=run.optimization.gradient_policy.accumulation_steps,
            rank=run.distributed.rank,
            world_size=run.distributed.world_size,
        )

    def _evaluation_context(
        self,
        run: Any,
        *,
        kind: str,
        phase_index: int,
        phase_epoch: int,
        epoch: int,
        batch: int,
        batches_total: int,
    ) -> EvaluationContext:
        return EvaluationContext(
            run_id=run.state.run_id,
            session_id=run.state.session_id,
            phase_index=phase_index,
            phase_epoch=phase_epoch,
            epoch=epoch,
            training_batch=batch,
            training_batches_total=batches_total,
            global_step=run.state.global_batch_step,
            device=torch.device(run.config.device),
            rank=run.distributed.rank,
            world_size=run.distributed.world_size,
            distributed_strategy=run.distributed,
            kind=kind,
        )

    def _evaluate(self, run: Any, context: EvaluationContext) -> EvaluationResult:
        result = run.evaluation.evaluate(
            run.network, run.config.evaluation_sources, run.datasets, context
        )
        run.metrics.record_evaluation(
            result,
            context=context,
            learning_rate=float(run.optimization.optimizer.param_groups[0]["lr"]),
        )
        if run.integrations.enabled:
            for source in result.sources.values():
                run.integrations.emit(
                    ValidationCompleted(
                        access=run.optimization.access(run.network),
                        total_loss=source.total_loss,
                        target_losses=source.target_losses,
                        evaluation_kind=context.kind,
                    )
                )
        return result

    def _check_pruning(self, run: Any) -> None:
        if os.getenv("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN") is None:
            return
        should_prune = False
        if run.distributed.rank == 0:
            path = (
                model_log_directory(run.config.project_root, run.config.model_name)
                / f"{run.config.model_name}.prune"
            )
            should_prune = path.exists()
        if run.distributed.world_size > 1:
            signal = torch.tensor(
                [int(should_prune)], device=run.config.device, dtype=torch.int32
            )
            torch.distributed.broadcast(signal, src=0)
            should_prune = bool(signal.item())
        if should_prune:
            raise SystemExit(143)

    def _update_monitor(self, run: Any, result: EvaluationResult) -> tuple[float, bool]:
        monitor = run.config.evaluation_monitor
        if monitor is None or monitor.source not in result.sources:
            return float("nan"), False
        value = result.sources[monitor.source].total_loss
        if monitor.mode == "max" and run.state.best_validation_loss == float("inf"):
            run.state.best_validation_loss = float("-inf")
        improved = (
            value < run.state.best_validation_loss
            if monitor.mode == "min"
            else value > run.state.best_validation_loss
        )
        if improved:
            run.state.best_validation_loss = value
            run.state.epochs_without_improvement = 0
            state = run.distributed.capture_model_state(run.network)
            if run.distributed.rank == 0:
                run.state.best_model_state_dict = state
        else:
            run.state.epochs_without_improvement += 1
        return value, improved

    def _publish(self, run: Any, network: Any, epoch: int) -> dict[str, Any]:
        repository = run.config.model.backbone.repository
        if repository is None:
            return {"success": False, "reason": "repository_not_configured"}
        if not repository.publish:
            return {"success": False, "reason": "publication_disabled"}
        return publish_revision(
            network.backbone,
            run.config.model.backbone,
            run.config.project_root,
            parent_revision_id=run.state.backbone_parent_revision_id,
            source_run_id=run.state.run_id,
            source_epoch=epoch,
        )

    def _manifest(
        self,
        run: Any,
        result: RunResult,
        *,
        status: str = "complete",
    ) -> None:
        if run.distributed.rank != 0:
            return
        path = run.checkpoints.store.latest_path.parent / "manifest.json"
        write_manifest(
            path,
            {
                "artifact_type": "sequifier_run_manifest",
                "format_version": 2,
                "run_id": run.state.run_id,
                "session_id": run.state.session_id,
                "status": status,
                "completion_reason": result.completion_reason,
                "source_epoch": run.state.epoch,
                "exports": [str(path) for path in result.exports],
                "backbone_parent_revision_id": run.state.backbone_parent_revision_id,
                "backbone_publication": result.publication,
            },
        )

    def run(self, run: Any) -> RunResult:
        config = run.config
        state = run.state
        accumulation_count = 0
        active_dataset: str | None = None
        last_identity: StepIdentity | None = None
        stop_requested = False
        completion_reason = "normal_completion"
        metric_windows: dict[str, dict[str, Any]] = {}
        last_latest = time.monotonic()
        last_snapshot = time.monotonic()
        last_snapshot_step = state.global_batch_step
        access = run.optimization.access(run.network)
        run.integrations.emit(ModelReady(access=access))
        run.optimization.optimizer.zero_grad(set_to_none=True)

        def flush() -> None:
            nonlocal accumulation_count, stop_requested
            if accumulation_count == 0:
                return
            if active_dataset is None or last_identity is None:
                raise RuntimeError("Gradient accumulation has no dataset identity.")
            result = run.optimization.complete_step(
                run.network,
                last_identity,
                run.integrations,
                UpdatePolicy(
                    frozen_parameter_ids=run.datasets.resolve(
                        active_dataset
                    ).frozen_parameter_ids,
                    gradient_divisor=accumulation_count,
                ),
            )
            state.optimizer_step = run.optimization.optimizer_step
            state.accumulation_index = 0
            accumulation_count = 0
            stop_requested = stop_requested or result.stop_requested

        def flush_metrics(
            dataset_name: str, epoch: int, batch: int, batches_total: int
        ) -> None:
            window = metric_windows.pop(dataset_name, None)
            if window is None:
                return
            dataset = run.datasets.resolve(dataset_name)
            total, targets = run.loss.finalize_accounting(
                window["sums"], window["count"], dataset, allow_empty=True
            )
            run.metrics.record_training(
                TrainingMetrics(
                    dataset=dataset_name,
                    part=window["part"],
                    epoch=epoch,
                    batch=batch,
                    batches_total=batches_total,
                    global_step=state.global_batch_step,
                    window_batches=window["batches"],
                    total_loss=float(total.item()),
                    target_losses={
                        name: float(value.item()) for name, value in targets.items()
                    },
                    learning_rate=float(
                        run.optimization.optimizer.param_groups[0]["lr"]
                    ),
                    seconds_per_batch=window["seconds"] / window["batches"],
                )
            )

        current_epoch = max(
            1,
            state.epoch
            + (0 if state.phase_epoch and not state.phase_epoch_complete else 1),
        )
        current_batch = sum(state.iterator_positions.values())
        current_batches_total = 1
        try:
            if (
                state.global_batch_step == 0
                and config.evaluation_sources
                and config.global_training.calculate_validation_loss_on_initialization
            ):
                first_sources = [
                    build_source_runtime(source, run.datasets)
                    for source in config.training_plan[0].sources
                ]
                self._evaluate(
                    run,
                    self._evaluation_context(
                        run,
                        kind="initial",
                        phase_index=0,
                        phase_epoch=0,
                        epoch=0,
                        batch=0,
                        batches_total=sum(
                            source.num_batches() for source in first_sources
                        ),
                    ),
                )
            for phase_index, phase in enumerate(config.training_plan):
                if phase_index < state.phase_index:
                    continue
                sources = [
                    build_source_runtime(source, run.datasets)
                    for source in phase.sources
                ]
                scheduler = SourceScheduler(
                    phase,
                    sources,
                    seed=config.seed + phase_index,
                    phase_index=phase_index,
                )
                if phase_index == state.phase_index and state.source_scheduler_state:
                    scheduler.load_state_dict(state.source_scheduler_state)
                start_epoch = (
                    state.phase_epoch + (1 if state.phase_epoch_complete else 0)
                    if phase_index == state.phase_index
                    else 1
                )
                for phase_epoch in range(max(1, start_epoch), phase.epochs + 1):
                    resuming = (
                        state.phase_index == phase_index
                        and state.phase_epoch == phase_epoch
                        and not state.phase_epoch_complete
                        and bool(state.source_scheduler_state)
                    )
                    current_epoch = state.epoch if resuming else state.epoch + 1
                    state.phase_index = phase_index
                    state.phase_epoch = phase_epoch
                    state.phase_epoch_complete = False
                    run.callable_network.train()
                    current_batches_total = sum(
                        source.num_batches() for source in sources
                    )
                    current_batch = (
                        sum(state.iterator_positions.values()) if resuming else 0
                    )
                    for runtime_batch in scheduler.iter_epoch(phase_epoch):
                        self._check_pruning(run)
                        current_batch += 1
                        dataset = run.datasets.resolve(runtime_batch.dataset)
                        if (
                            active_dataset is not None
                            and active_dataset != dataset.name
                        ):
                            flush()
                        active_dataset = dataset.name
                        started = time.perf_counter()
                        prepared = run.loss.prepare_batch(
                            runtime_batch.batch, dataset, torch.device(config.device)
                        )
                        dataset.metrics.training_batches += 1
                        identity = self._identity(
                            run, current_epoch, current_batch, accumulation_count
                        )
                        prepared_event = BatchPrepared(
                            access=access,
                            identity=identity,
                            inputs=prepared.features,
                            targets=prepared.targets,
                            metadata=prepared.metadata,
                        )
                        if run.integrations.enabled:
                            run.integrations.emit(prepared_event)
                        trace = run.integrations.forward_trace(prepared_event)
                        output = run.callable_network(
                            prepared.features,
                            prepared.metadata,
                            interface_name=dataset.interface_name,
                            trace=trace,
                        )
                        if run.integrations.enabled:
                            run.integrations.emit(
                                ForwardCompleted(
                                    access=access,
                                    identity=identity,
                                    outputs=output.logits,
                                    captures={}
                                    if trace is None
                                    else dict(trace.captures),
                                )
                            )
                        loss = run.loss.calculate(
                            output, prepared, dataset, run.network
                        )
                        if run.integrations.enabled:
                            run.integrations.emit(
                                LossComputed(
                                    access=access,
                                    identity=identity,
                                    loss=loss.backward_loss.detach(),
                                    backward_loss=loss.backward_loss,
                                )
                            )
                        run.optimization.accumulate(
                            loss.backward_loss, identity, run.integrations, run.network
                        )
                        accumulation_count += 1
                        last_identity = identity
                        state.epoch = current_epoch
                        state.batch = current_batch
                        state.global_batch_step = identity.global_batch_step
                        state.accumulation_index = accumulation_count - 1
                        state.source_scheduler_state = scheduler.state_dict()
                        position_key = f"{runtime_batch.dataset}.{runtime_batch.part}"
                        state.iterator_positions[position_key] = (
                            state.iterator_positions.get(position_key, 0) + 1
                        )
                        dataset.parts[runtime_batch.part].iterator_positions[
                            "training"
                        ] = state.iterator_positions[position_key]
                        window = metric_windows.setdefault(
                            dataset.name,
                            {
                                "sums": {
                                    name: value.new_zeros(())
                                    for name, value in loss.accounting_sums.items()
                                },
                                "count": loss.accounting_count.new_zeros(()).to(
                                    next(iter(loss.accounting_sums.values())).dtype
                                ),
                                "batches": 0,
                                "seconds": 0.0,
                                "part": runtime_batch.part,
                            },
                        )
                        for name, value in loss.accounting_sums.items():
                            window["sums"][name] += value
                        window["count"] += loss.accounting_count.to(
                            window["count"].dtype
                        )
                        window["batches"] += 1
                        window["seconds"] += time.perf_counter() - started
                        if window["part"] != runtime_batch.part:
                            window["part"] = None
                        if window["batches"] >= config.global_training.log_interval:
                            flush_metrics(
                                dataset.name,
                                current_epoch,
                                current_batch,
                                current_batches_total,
                            )
                        if (
                            accumulation_count
                            >= run.optimization.gradient_policy.accumulation_steps
                        ):
                            flush()
                        now = time.monotonic()
                        snapshot_due = bool(
                            config.global_training.save_interval_batches is not None
                            and state.global_batch_step - last_snapshot_step
                            >= config.global_training.save_interval_batches
                        ) or bool(
                            config.global_training.save_interval_minutes is not None
                            and now - last_snapshot
                            >= config.global_training.save_interval_minutes * 60
                        )
                        latest_due = bool(
                            config.global_training.save_latest_interval_minutes
                            is not None
                            and now - last_latest
                            >= config.global_training.save_latest_interval_minutes * 60
                        )
                        if run.distributed.world_size > 1:
                            checkpoint_due = torch.tensor(
                                [
                                    int(latest_due) if run.distributed.rank == 0 else 0,
                                    int(snapshot_due)
                                    if run.distributed.rank == 0
                                    else 0,
                                ],
                                device=config.device,
                                dtype=torch.int32,
                            )
                            torch.distributed.broadcast(checkpoint_due, src=0)
                            latest_due = bool(checkpoint_due[0].item())
                            snapshot_due = bool(checkpoint_due[1].item())
                        if latest_due or snapshot_due:
                            flush()
                        if latest_due:
                            run.checkpoints.save(
                                CheckpointRequest("latest", identity), run
                            )
                            last_latest = time.monotonic()
                        if snapshot_due:
                            if (
                                config.global_training.save_interval_val_loss
                                and config.evaluation_sources
                            ):
                                self._evaluate(
                                    run,
                                    self._evaluation_context(
                                        run,
                                        kind="interval",
                                        phase_index=phase_index,
                                        phase_epoch=phase_epoch,
                                        epoch=current_epoch,
                                        batch=current_batch,
                                        batches_total=current_batches_total,
                                    ),
                                )
                                run.callable_network.train()
                            run.checkpoints.save(
                                CheckpointRequest(
                                    f"epoch-{current_epoch}-batch-{current_batch}",
                                    identity,
                                ),
                                run,
                            )
                            last_snapshot = time.monotonic()
                            last_snapshot_step = state.global_batch_step
                        if stop_requested:
                            completion_reason = "integration_requested_stop"
                            break
                    flush()
                    for dataset_name in tuple(metric_windows):
                        flush_metrics(
                            dataset_name,
                            current_epoch,
                            current_batch,
                            current_batches_total,
                        )
                    state.phase_epoch_complete = True
                    evaluation = (
                        self._evaluate(
                            run,
                            self._evaluation_context(
                                run,
                                kind="epoch_end",
                                phase_index=phase_index,
                                phase_epoch=phase_epoch,
                                epoch=current_epoch,
                                batch=current_batch,
                                batches_total=current_batches_total,
                            ),
                        )
                        if config.evaluation_sources
                        else EvaluationResult()
                    )
                    if run.optimization.scheduler_policy.step_on == "epoch":
                        run.optimization.step_scheduler()
                    state.iterator_positions = {}
                    state.source_scheduler_state = scheduler.state_dict()
                    self._update_monitor(run, evaluation)
                    if current_epoch % config.global_training.save_interval_epochs == 0:
                        run.checkpoints.save(
                            CheckpointRequest(f"epoch-{current_epoch}", last_identity),
                            run,
                        )
                    patience = config.global_training.early_stopping_epochs
                    if (
                        patience is not None
                        and config.evaluation_monitor is not None
                        and state.epochs_without_improvement >= patience
                    ):
                        stop_requested = True
                        completion_reason = "early_stopping"
                    if stop_requested:
                        break
                if stop_requested:
                    break
        except BaseException:
            flush()
            run.optimization.optimizer.zero_grad(set_to_none=True)
            if last_identity is not None:
                run.checkpoints.save(CheckpointRequest("latest", last_identity), run)
            raise

        run.distributed.barrier()
        last_state = run.distributed.capture_model_state(run.network)
        best_state = state.best_model_state_dict or last_state
        last_export = run.exporting.export(
            run.network, last_state, ExportOptions("last", state.epoch)
        )
        best_export = run.exporting.export(
            run.network,
            canonicalize_state_dict(best_state),
            ExportOptions("best", state.epoch),
        )
        publication = (
            self._publish(run, last_export.network, state.epoch)
            if run.distributed.rank == 0
            else {"success": False, "reason": "nonzero_rank"}
        )
        result = RunResult(
            completion_reason,
            best_export.paths + last_export.paths,
            publication,
        )
        self._manifest(run, result)
        run.integrations.emit(
            RunCompleted(access=access, completion_reason=completion_reason)
        )
        return result
