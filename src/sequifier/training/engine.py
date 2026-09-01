from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from sequifier.integration.callbacks import IntegrationManager
from sequifier.integration.contexts import (
    BackwardCompleted,
    GradientsClipped,
    GradientsUnscaled,
    ModelReady,
    OptimizerStepCompleted,
    OptimizerStepStarting,
    StepIdentity,
    TrainingAccess,
    TrainingEvent,
)
from sequifier.integration.controls import apply_training_directive
from sequifier.model.parameter_catalog import ParameterCatalog
from sequifier.training.state import TrainingState
from sequifier.typechecking import beartype


@dataclass
class TrainingEngine:
    model: nn.Module
    optimizer: Optimizer
    scheduler: Any
    scaler: GradScaler
    state: TrainingState
    integrations: IntegrationManager

    @beartype
    def __post_init__(self) -> None:
        access_model = getattr(self.model, "network", self.model)
        self.parameter_catalog = ParameterCatalog(access_model)
        self.access = TrainingAccess(
            model=access_model,
            parameter_catalog=self.parameter_catalog,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.stop_requested = False
        self._skip_next_scheduler_step = False
        self.best_model_state_dict: dict[str, Tensor] | None = getattr(
            self.model, "_resume_best_model_state_dict", None
        )

    @property
    @beartype
    def rank(self) -> int:
        rank = getattr(self.model, "rank", None)
        return int(rank or 0)

    @property
    @beartype
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @beartype
    def identity(
        self,
        *,
        epoch: int,
        batch: int,
        num_batches: int,
        accumulation_steps: int | None,
    ) -> StepIdentity:
        steps = accumulation_steps or 1
        global_batch_step = (epoch - 1) * num_batches + batch
        return StepIdentity(
            epoch=epoch,
            batch=batch,
            global_batch_step=global_batch_step,
            optimizer_step=self.state.optimizer_step,
            accumulation_index=(batch - 1) % steps,
            accumulation_steps=steps,
            rank=self.rank,
            world_size=self.world_size,
        )

    @beartype
    def update_batch_state(self, identity: StepIdentity) -> None:
        self.state.epoch = identity.epoch
        self.state.batch = identity.batch
        self.state.global_batch_step = identity.global_batch_step
        self.state.accumulation_index = identity.accumulation_index

    @beartype
    def emit(self, event: TrainingEvent) -> None:
        self.integrations.emit(event)

    @beartype
    def model_ready(self) -> None:
        self.emit(ModelReady(access=self.access))

    @beartype
    def step_scheduler(self) -> bool:
        if self._skip_next_scheduler_step:
            self._skip_next_scheduler_step = False
            return False
        if (
            hasattr(self.scheduler, "total_steps")
            and self.scheduler.last_epoch >= self.scheduler.total_steps
        ):
            return False
        self.scheduler.step()
        return True

    @beartype
    def backward_and_step(
        self,
        *,
        backward_loss: Any,
        identity: StepIdentity,
        optimizer_step_due: bool,
        gradient_clip_norm: float | None,
    ) -> bool:
        self.scaler.scale(backward_loss).backward()
        if self.integrations.enabled:
            self.emit(
                BackwardCompleted(
                    access=self.access,
                    identity=identity,
                    gradients_are_scaled=self.scaler.is_enabled(),
                    optimizer_step_due=optimizer_step_due,
                )
            )
        if not optimizer_step_due:
            return False
        return self._complete_optimizer_step(
            identity=identity,
            gradient_clip_norm=gradient_clip_norm,
        )

    @beartype
    def accumulate_loss(self, loss: Tensor, identity: StepIdentity) -> None:
        """Backpropagate one unnormalized accumulation microbatch."""

        self.scaler.scale(loss).backward()
        if self.integrations.enabled:
            self.emit(
                BackwardCompleted(
                    access=self.access,
                    identity=identity,
                    gradients_are_scaled=self.scaler.is_enabled(),
                    optimizer_step_due=False,
                )
            )

    @beartype
    def flush_accumulation(
        self,
        *,
        identity: StepIdentity,
        microbatch_count: int,
        frozen_parameter_ids: frozenset[int] = frozenset(),
        gradient_clip_norm: float | None = None,
        scheduler_step_on: str = "epoch",
    ) -> bool:
        """Normalize, mask, clip, and step one dataset-pure gradient window."""

        if microbatch_count <= 0:
            return False
        step_applied = self._complete_optimizer_step(
            identity=identity,
            gradient_clip_norm=gradient_clip_norm,
            gradient_divisor=microbatch_count,
            frozen_parameter_ids=frozen_parameter_ids,
            scheduler_step_on=scheduler_step_on,
        )
        self.state.accumulation_index = 0
        return step_applied

    @beartype
    def _complete_optimizer_step(
        self,
        *,
        identity: StepIdentity,
        gradient_clip_norm: float | None,
        gradient_divisor: int = 1,
        frozen_parameter_ids: frozenset[int] = frozenset(),
        scheduler_step_on: str | None = None,
    ) -> bool:
        """Complete the single transaction that may apply an optimizer update."""

        self.scaler.unscale_(self.optimizer)
        divisor = float(gradient_divisor)
        parameters = tuple(self.model.parameters())
        for parameter in parameters:
            if id(parameter) in frozen_parameter_ids:
                parameter.grad = None
            elif divisor != 1.0 and parameter.grad is not None:
                parameter.grad.div_(divisor)
        unscaled = GradientsUnscaled(access=self.access, identity=identity)
        if self.integrations.enabled:
            self.emit(unscaled)
        directive = self.integrations.directive(unscaled)
        if directive is not None:
            apply_training_directive(
                self.optimizer,
                directive,
                scheduler=self.scheduler,
            )

        clip_norm = gradient_clip_norm
        if directive is not None:
            if directive.disable_gradient_clipping:
                clip_norm = None
            elif directive.gradient_clip_norm is not None:
                clip_norm = directive.gradient_clip_norm
        if clip_norm is not None:
            parameters = [
                parameter for parameter in parameters if parameter.grad is not None
            ]
            total_norm = nn.utils.clip_grad_norm_(parameters, clip_norm)
            if self.integrations.enabled:
                self.emit(
                    GradientsClipped(
                        access=self.access,
                        identity=identity,
                        max_norm=float(clip_norm),
                        total_norm=total_norm,
                    )
                )

        skip_step = bool(directive is not None and directive.skip_optimizer_step)
        if self.integrations.enabled:
            self.emit(
                OptimizerStepStarting(
                    access=self.access,
                    identity=identity,
                    skip_optimizer_step=skip_step,
                    reason=None if directive is None else directive.reason,
                )
            )
        step_applied = False
        if not skip_step:
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            step_applied = (
                not self.scaler.is_enabled()
                or self.scaler.get_scale() >= previous_scale
            )
            if step_applied:
                self.state.optimizer_step += 1
                if scheduler_step_on == "batch":
                    if directive is None or not directive.skip_scheduler_step:
                        self.step_scheduler()
                elif directive is not None and directive.skip_scheduler_step:
                    self._skip_next_scheduler_step = True
                completed = StepIdentity(
                    epoch=identity.epoch,
                    batch=identity.batch,
                    global_batch_step=identity.global_batch_step,
                    optimizer_step=self.state.optimizer_step,
                    accumulation_index=identity.accumulation_index,
                    accumulation_steps=identity.accumulation_steps,
                    rank=identity.rank,
                    world_size=identity.world_size,
                )
                if self.integrations.enabled:
                    self.emit(
                        OptimizerStepCompleted(access=self.access, identity=completed)
                    )
        else:
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if directive is not None and directive.stop_after_step:
            self.stop_requested = True
        return step_applied

    @beartype
    def run_plan(
        self,
        config: Any,
        dataset_runtimes: dict[str, Any],
        ddp_model: Any = None,
    ) -> str:
        """Execute ordered sequential/interleaved phases with dataset-pure steps."""

        from sequifier.io.batch import SequifierBatch
        from sequifier.training.runtime import SourceScheduler, build_source_runtime

        model: Any = self.model
        model_to_call = ddp_model if ddp_model is not None else model
        accumulation_steps = config.global_training.accumulation_steps or 1
        accumulation_count = 0
        active_dataset = None
        last_identity = None
        last_latest_save_time = time.monotonic()
        last_snapshot_save_time = time.monotonic()
        last_snapshot_global_step = self.state.global_batch_step
        last_boundary_state = self.state.snapshot()
        backward_started = False
        batch_committed = False
        flush_in_progress = False
        completion_reason = "normal_completion"
        metric_windows: dict[str, dict[str, Any]] = {}
        current_run_epoch = max(
            1,
            self.state.epoch
            if self.state.phase_epoch and not self.state.phase_epoch_complete
            else self.state.epoch + 1,
        )
        current_epoch_batch = sum(self.state.iterator_positions.values())
        current_epoch_batches_total = 1
        self.model_ready()
        self.optimizer.zero_grad(set_to_none=True)

        @beartype
        def flush() -> None:
            nonlocal accumulation_count, flush_in_progress, last_boundary_state
            if accumulation_count == 0:
                return
            if active_dataset is None or last_identity is None:
                raise RuntimeError("Accumulation state has no active dataset")
            flush_in_progress = True
            try:
                self.flush_accumulation(
                    identity=last_identity,
                    microbatch_count=accumulation_count,
                    frozen_parameter_ids=dataset_runtimes[
                        active_dataset
                    ].frozen_parameter_ids,
                    gradient_clip_norm=config.global_training.gradient_clip,
                    scheduler_step_on=config.global_training.scheduler_step_on,
                )
            finally:
                flush_in_progress = False
            accumulation_count = 0
            last_boundary_state = self.state.snapshot()

        @beartype
        def update_monitor(evaluation_results: dict[str, float]) -> tuple[float, bool]:
            monitor = config.evaluation_monitor
            if monitor is None or monitor.source not in evaluation_results:
                return float("nan"), False
            monitored = evaluation_results[monitor.source]
            if monitor.mode == "max" and self.state.best_validation_loss == float(
                "inf"
            ):
                self.state.best_validation_loss = float("-inf")
            improved = (
                monitored < self.state.best_validation_loss
                if monitor.mode == "min"
                else monitored > self.state.best_validation_loss
            )
            if improved:
                self.state.best_validation_loss = monitored
                self.state.epochs_without_improvement = 0
                if self.rank == 0:
                    self.best_model_state_dict = {
                        name.replace("_orig_mod.", "").replace(
                            ".module.", "."
                        ): value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            else:
                self.state.epochs_without_improvement += 1
            return monitored, improved

        @beartype
        def flush_metric_window(
            dataset_name: str,
            epoch: int,
            batch: int,
            batches_total: int,
        ) -> None:
            window = metric_windows.pop(dataset_name, None)
            if window is None:
                return
            total_loss, target_losses = model._finalize_loss_components(
                window["sums"],
                window["count"],
                window["target_names"],
                "training metrics",
                raise_on_empty=False,
            )
            if self.rank != 0:
                return
            learning_rate = self.optimizer.param_groups[0]["lr"]
            seconds_per_batch = window["seconds"] / window["batches"]
            writer = model.metric_writers_by_dataset[dataset_name]
            writer.write_training(
                run_id=model.run_id,
                session_id=model.session_id,
                epoch=epoch,
                batch=batch,
                batches_total=batches_total,
                global_step=self.state.global_batch_step,
                window_batches=window["batches"],
                total_loss=total_loss,
                target_losses=target_losses,
                learning_rate=learning_rate,
                seconds_per_batch=seconds_per_batch,
                dataset=dataset_name,
                part=window["part"],
            )
            model.logger.bind(log_channel="metric").info(
                f"Epoch {epoch:3d} | Batch {batch:5d}/{batches_total:5d} | "
                f"Dataset: {dataset_name} | Loss: "
                f"{float(total_loss.detach().cpu().item()): .2e} | "
                f"LR: {float(learning_rate): .2e} | "
                f"S/Batch {float(seconds_per_batch): .2e}"
            )

        try:
            is_fresh_run = (
                self.state.global_batch_step == 0
                and self.state.phase_epoch == 0
                and not self.state.source_scheduler_state
            )
            if (
                is_fresh_run
                and config.evaluation_sources
                and config.global_training.calculate_validation_loss_on_initialization
            ):
                first_phase = config.training_plan[0]
                initial_batches_total = sum(
                    build_source_runtime(source, dataset_runtimes).num_batches()
                    for source in first_phase.sources
                )
                model.evaluate_sources(
                    config.evaluation_sources,
                    dataset_runtimes,
                    phase_index=0,
                    phase_epoch=0,
                    evaluation_kind="initial",
                    run_epoch=0,
                    training_batch=0,
                    training_batches_total=initial_batches_total,
                    global_step=self.state.global_batch_step,
                )

            for phase_index, phase in enumerate(config.training_plan):
                if phase_index < self.state.phase_index:
                    continue
                sources = [
                    build_source_runtime(source, dataset_runtimes)
                    for source in phase.sources
                ]
                scheduler = SourceScheduler(
                    phase,
                    sources,
                    seed=config.seed + phase_index,
                    phase_index=phase_index,
                )
                if (
                    self.state.phase_index == phase_index
                    and self.state.source_scheduler_state
                ):
                    scheduler.load_state_dict(self.state.source_scheduler_state)
                start_epoch = (
                    self.state.phase_epoch
                    + (1 if self.state.phase_epoch_complete else 0)
                    if self.state.phase_index == phase_index
                    else 1
                )
                start_epoch = max(1, start_epoch)
                for phase_epoch in range(start_epoch, phase.epochs + 1):
                    resuming_epoch = (
                        self.state.phase_index == phase_index
                        and self.state.phase_epoch == phase_epoch
                        and not self.state.phase_epoch_complete
                        and bool(self.state.source_scheduler_state)
                    )
                    current_run_epoch = (
                        self.state.epoch if resuming_epoch else self.state.epoch + 1
                    )
                    self.state.phase_index = phase_index
                    self.state.phase_epoch = phase_epoch
                    self.state.phase_epoch_complete = False
                    model_to_call.train()
                    batches_total = sum(source.num_batches() for source in sources)
                    current_epoch_batches_total = batches_total
                    current_epoch_batch = (
                        sum(self.state.iterator_positions.values())
                        if self.state.source_scheduler_state
                        else 0
                    )
                    for runtime_batch in scheduler.iter_epoch(phase_epoch):
                        model._check_and_terminate()
                        backward_started = False
                        batch_committed = False
                        batch_started = time.perf_counter()
                        current_epoch_batch += 1
                        dataset_name = runtime_batch.dataset
                        if (
                            active_dataset is not None
                            and dataset_name != active_dataset
                        ):
                            flush()
                        if dataset_name != active_dataset:
                            model.activate_dataset(
                                dataset_name, dataset_runtimes[dataset_name]
                            )
                            active_dataset = dataset_name

                        batch = runtime_batch.batch
                        if not isinstance(batch, SequifierBatch):
                            raise TypeError(
                                "Training sources must yield SequifierBatch"
                            )
                        data = {
                            key: value.to(model.device, non_blocking=True)
                            for key, value in batch.inputs.items()
                            if key in model.input_columns
                        }
                        targets = {
                            key: value.to(model.device, non_blocking=True)
                            for key, value in batch.targets.items()
                            if key in model.target_column_types
                        }
                        metadata = {
                            key: value.to(model.device, non_blocking=True)
                            for key, value in batch.metadata.items()
                        }
                        data, targets, metadata = model.objective.prepare_batch(
                            data, targets, metadata
                        )
                        next_global_batch_step = self.state.global_batch_step + 1
                        next_accumulation_count = accumulation_count + 1
                        identity = StepIdentity(
                            epoch=current_run_epoch,
                            batch=current_epoch_batch,
                            global_batch_step=next_global_batch_step,
                            optimizer_step=self.state.optimizer_step,
                            accumulation_index=next_accumulation_count - 1,
                            accumulation_steps=accumulation_steps,
                            rank=self.rank,
                            world_size=self.world_size,
                        )
                        output = model_to_call(
                            data, metadata=metadata, return_logits=True
                        )
                        loss, _, local_sums, local_count = (
                            model._calculate_training_loss(output, targets, metadata)
                        )
                        backward_started = True
                        self.accumulate_loss(loss, identity)
                        accumulation_count = next_accumulation_count
                        last_identity = identity
                        self.state.global_batch_step = next_global_batch_step
                        self.update_batch_state(identity)
                        self.state.source_scheduler_state = scheduler.state_dict()
                        position_ref = runtime_batch.dataset + "." + runtime_batch.part
                        self.state.iterator_positions[position_ref] = (
                            self.state.iterator_positions.get(position_ref, 0) + 1
                        )
                        batch_committed = True
                        window = metric_windows.get(dataset_name)
                        if window is None:
                            window_sums, window_count = model._new_loss_accumulators(
                                list(model.target_columns)
                            )
                            window = {
                                "sums": window_sums,
                                "count": window_count,
                                "target_names": list(model.target_columns),
                                "batches": 0,
                                "seconds": 0.0,
                                "part": runtime_batch.part,
                            }
                            metric_windows[dataset_name] = window
                        model._accumulate_loss_components(
                            window["sums"],
                            window["count"],
                            local_sums,
                            local_count,
                        )
                        window["batches"] += 1
                        window["seconds"] += time.perf_counter() - batch_started
                        if window["part"] != runtime_batch.part:
                            window["part"] = None
                        if window["batches"] >= model.log_interval:
                            flush_metric_window(
                                dataset_name,
                                current_run_epoch,
                                current_epoch_batch,
                                batches_total,
                            )
                        if accumulation_count == accumulation_steps:
                            flush()
                        now = time.monotonic()
                        snapshot_due = (
                            config.global_training.save_interval_batches is not None
                            and self.state.global_batch_step - last_snapshot_global_step
                            >= config.global_training.save_interval_batches
                        )
                        snapshot_due = snapshot_due or bool(
                            config.global_training.save_interval_minutes is not None
                            and now - last_snapshot_save_time
                            >= config.global_training.save_interval_minutes * 60
                        )
                        latest_due = bool(
                            config.global_training.save_latest_interval_minutes
                            is not None
                            and now - last_latest_save_time
                            >= config.global_training.save_latest_interval_minutes * 60
                        )
                        if config.global_training.distributed:
                            checkpoint_directives = torch.tensor(
                                [int(latest_due), int(snapshot_due)],
                                dtype=torch.int32,
                                device=model.device,
                            )
                            dist.broadcast(checkpoint_directives, src=0)
                            latest_due = bool(checkpoint_directives[0].item())
                            snapshot_due = bool(checkpoint_directives[1].item())
                        if latest_due or snapshot_due:
                            flush()
                        if latest_due:
                            model._save(
                                current_run_epoch,
                                max(0, current_epoch_batch - 1),
                                np.float32(float("nan")),
                                suffix="latest",
                                best_val_loss=self.state.best_validation_loss,
                                n_epochs_no_improvement=(
                                    self.state.epochs_without_improvement
                                ),
                                best_model_state_dict=self.best_model_state_dict,
                                num_batches=batches_total,
                                training_engine=self,
                            )
                            last_latest_save_time = time.monotonic()
                        if snapshot_due:
                            interval_loss = float("nan")
                            if (
                                config.global_training.save_interval_val_loss
                                and config.evaluation_sources
                            ):
                                interval_results = model.evaluate_sources(
                                    config.evaluation_sources,
                                    dataset_runtimes,
                                    phase_index=phase_index,
                                    phase_epoch=phase_epoch,
                                    evaluation_kind="interval",
                                    run_epoch=current_run_epoch,
                                    training_batch=current_epoch_batch,
                                    training_batches_total=batches_total,
                                    global_step=self.state.global_batch_step,
                                )
                                active_dataset = None
                                monitor = config.evaluation_monitor
                                if monitor is not None:
                                    interval_loss = interval_results[monitor.source]
                            model._save(
                                current_run_epoch,
                                max(0, current_epoch_batch - 1),
                                np.float32(interval_loss),
                                suffix=(
                                    f"epoch-{current_run_epoch}-batch-"
                                    f"{current_epoch_batch}"
                                ),
                                best_val_loss=self.state.best_validation_loss,
                                n_epochs_no_improvement=(
                                    self.state.epochs_without_improvement
                                ),
                                best_model_state_dict=self.best_model_state_dict,
                                num_batches=batches_total,
                                training_engine=self,
                            )
                            last_snapshot_save_time = time.monotonic()
                            last_snapshot_global_step = self.state.global_batch_step
                        if self.stop_requested:
                            completion_reason = "integration_requested_stop"
                            break
                    flush()
                    for dataset_name in sorted(metric_windows):
                        flush_metric_window(
                            dataset_name,
                            current_run_epoch,
                            current_epoch_batch,
                            batches_total,
                        )
                    self.state.phase_epoch_complete = True
                    if config.evaluation_sources:
                        evaluation_results = model.evaluate_sources(
                            config.evaluation_sources,
                            dataset_runtimes,
                            phase_index=phase_index,
                            phase_epoch=phase_epoch,
                            evaluation_kind="epoch_end",
                            run_epoch=current_run_epoch,
                            training_batch=current_epoch_batch,
                            training_batches_total=batches_total,
                            global_step=self.state.global_batch_step,
                        )
                        active_dataset = None
                    else:
                        evaluation_results = {}
                    if config.global_training.scheduler_step_on == "epoch":
                        self.step_scheduler()
                    self.state.iterator_positions = {}
                    self.state.source_scheduler_state = scheduler.state_dict()
                    monitored = float("nan")
                    monitor = config.evaluation_monitor
                    monitored, _ = update_monitor(evaluation_results)
                    self.state.epoch = current_run_epoch
                    if (
                        current_run_epoch % config.global_training.save_interval_epochs
                        == 0
                    ):
                        model._save(
                            current_run_epoch,
                            max(0, current_epoch_batch - 1),
                            np.float32(monitored),
                            suffix=f"epoch-{current_run_epoch}",
                            best_val_loss=self.state.best_validation_loss,
                            n_epochs_no_improvement=(
                                self.state.epochs_without_improvement
                            ),
                            best_model_state_dict=self.best_model_state_dict,
                            num_batches=batches_total,
                            training_engine=self,
                        )
                        last_snapshot_save_time = time.monotonic()
                        last_snapshot_global_step = self.state.global_batch_step
                    patience = config.global_training.early_stopping_epochs
                    if (
                        patience is not None
                        and monitor is not None
                        and self.state.epochs_without_improvement >= patience
                    ):
                        self.stop_requested = True
                        completion_reason = "early_stopping"
                    if self.stop_requested:
                        break
                if self.stop_requested:
                    break
        except BaseException:
            if flush_in_progress:
                raise
            if backward_started and not batch_committed:
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                self.state.restore(last_boundary_state)
            else:
                flush()
            if last_identity is not None:
                model._save(
                    current_run_epoch,
                    max(0, current_epoch_batch - 1),
                    np.float32(float("nan")),
                    suffix="latest",
                    best_val_loss=self.state.best_validation_loss,
                    n_epochs_no_improvement=self.state.epochs_without_improvement,
                    best_model_state_dict=self.best_model_state_dict,
                    num_batches=current_epoch_batches_total,
                    training_engine=self,
                )
            raise
        finally:
            flush()
        return completion_reason
