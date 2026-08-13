from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.distributed as dist
from torch import nn
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
    RunCompleted,
    StepIdentity,
    TrainingAccess,
    TrainingEvent,
)
from sequifier.integration.controls import apply_training_directive
from sequifier.model.parameter_catalog import ParameterCatalog
from sequifier.training.distributed import broadcast_publication_result
from sequifier.training.lifecycle import publish_final_backbone, write_terminal_manifest
from sequifier.training.metrics import StructuredMetricWriters
from sequifier.training.state import TrainingState


@dataclass
class TrainingEngine:
    model: nn.Module
    objective: Any
    criteria: dict[str, nn.Module]
    optimizer: Optimizer
    scheduler: Any
    scaler: GradScaler
    state: TrainingState
    integrations: IntegrationManager

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

    @property
    def rank(self) -> int:
        rank = getattr(self.model, "rank", None)
        return int(rank or 0)

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

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

    def update_batch_state(self, identity: StepIdentity) -> None:
        self.state.epoch = identity.epoch
        self.state.batch = identity.batch
        self.state.global_batch_step = identity.global_batch_step
        self.state.accumulation_index = identity.accumulation_index

    def emit(self, event: TrainingEvent) -> None:
        self.integrations.emit(event)

    def model_ready(self) -> None:
        self.emit(ModelReady(access=self.access))

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

        self.scaler.unscale_(self.optimizer)
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
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
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
                if directive is not None and directive.skip_scheduler_step:
                    self._skip_next_scheduler_step = True
                completed_identity = StepIdentity(
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
                        OptimizerStepCompleted(
                            access=self.access, identity=completed_identity
                        )
                    )
        else:
            step_applied = False
            self.scaler.update()
        self.optimizer.zero_grad()
        if directive is not None and directive.stop_after_step:
            self.stop_requested = True
        return step_applied

    def run(
        self, train_loader: Any, validation_loader: Any, ddp_model: Any = None
    ) -> None:
        model: Any = self.model
        setattr(model, "_training_engine", self)
        self.model_ready()
        if model.rank == 0 and model.metric_writers is None:
            model.metric_writers = StructuredMetricWriters(
                model.project_root,
                model.model_name,
                model.rank,
                class_share_columns=model.class_share_log_columns,
            )
        model.logger.info(
            f"--- Starting Training for model: {model.model_name} | "
            f"run: {model.run_id} | session: {model.session_id} ---"
        )

        best_val_loss = float(model._resume_best_val_loss)
        n_epochs_no_improvement = model._resume_n_epochs_no_improvement
        last_epoch = model.start_epoch - 1
        best_model_state = model._resume_best_model_state_dict
        completion_reason = "normal_completion"
        total_loss = np.float32(np.nan)

        try:
            model.last_latest_save_time = time.time()
            model.last_batch_save_time = time.time()
            model.last_batch_save_global_step = (model.start_epoch - 1) * len(
                train_loader
            ) + model.start_batch
            if (
                model.start_epoch == 1
                and model.hparams.training_spec.calculate_validation_loss_on_initialization
            ):
                total_loss, total_losses, class_counts = model._evaluate(
                    validation_loader, ddp_model
                )
                model._log_epoch_results(
                    0,
                    0,
                    0.0,
                    total_loss,
                    total_losses,
                    class_counts,
                    0,
                    len(train_loader),
                    "initial",
                )

            for epoch in range(
                model.start_epoch, model.hparams.training_spec.epochs + 1
            ):
                if (
                    model.early_stopping_epochs is not None
                    and n_epochs_no_improvement >= model.early_stopping_epochs
                ):
                    completion_reason = "early_stopping"
                    break
                if epoch > model.start_epoch and np.isnan(total_loss):
                    raise RuntimeError("Validation loss became NaN.")

                epoch_start_time = time.time()
                train_loader.dataset.set_epoch(epoch)
                validation_loader.dataset.set_epoch(epoch)
                model._train_epoch(
                    train_loader,
                    validation_loader,
                    epoch,
                    ddp_model,
                    best_val_loss,
                    n_epochs_no_improvement,
                    best_model_state,
                )
                if self.stop_requested:
                    completion_reason = "integration_requested_stop"
                    last_epoch = epoch
                    break

                total_loss, total_losses, class_counts = model._evaluate(
                    validation_loader, ddp_model
                )
                model._log_epoch_results(
                    epoch,
                    len(train_loader),
                    time.time() - epoch_start_time,
                    total_loss,
                    total_losses,
                    class_counts,
                    epoch * len(train_loader),
                    len(train_loader),
                    "epoch_end",
                )

                if total_loss < best_val_loss:
                    best_val_loss = float(total_loss)
                    best_model_state = model._get_full_state_dict(ddp_model)
                    n_epochs_no_improvement = 0
                else:
                    n_epochs_no_improvement += 1
                self.state.epoch = epoch
                self.state.best_validation_loss = best_val_loss
                self.state.epochs_without_improvement = n_epochs_no_improvement

                if model.scheduler_step_on == "epoch":
                    self.step_scheduler()
                if epoch % model.save_interval_epochs == 0:
                    model._save(
                        epoch,
                        len(train_loader) - 1,
                        total_loss,
                        ddp_model=ddp_model,
                        suffix=f"epoch-{epoch}",
                        best_val_loss=best_val_loss,
                        n_epochs_no_improvement=n_epochs_no_improvement,
                        best_model_state_dict=best_model_state,
                        num_batches=len(train_loader),
                    )
                last_epoch = epoch
                model._check_and_terminate()
        except KeyboardInterrupt:
            completion_reason = "keyboard_interruption"
            model.logger.warning("Training interrupted; exporting final state.")
        except BaseException as error:
            if model.rank == 0:
                is_pruned = isinstance(error, SystemExit) and error.code == 143
                write_terminal_manifest(
                    model,
                    status="pruned" if is_pruned else "failed",
                    completion_reason=("optuna_pruning" if is_pruned else "exception"),
                    source_epoch=last_epoch,
                    exports_succeeded=False,
                    publication={"success": False, "reason": "not_attempted"},
                )
            raise

        if model.hparams.training_spec.distributed:
            dist.barrier()
        last_model_state = model._get_full_state_dict(ddp_model)
        if best_model_state is None:
            if model.rank == 0:
                model.logger.info(
                    "No validation improvement... Saving last model as 'best'."
                )
            best_model_state = last_model_state

        finalization: dict[str, Any] | None = None
        if model.rank == 0:
            try:
                exported_last_model = model._export(
                    last_model_state, "last", last_epoch, clean=True
                )
                model._export(best_model_state, "best", last_epoch, clean=True)
                if exported_last_model is None:
                    raise RuntimeError("Rank 0 did not construct an export model.")
            except Exception as error:
                finalization = {
                    "exports_succeeded": False,
                    "publication": {"success": False, "reason": "not_attempted"},
                    "error": f"{type(error).__name__}: {error}",
                }
                write_terminal_manifest(
                    model,
                    status="failed",
                    completion_reason="export_failure",
                    source_epoch=last_epoch,
                    exports_succeeded=False,
                    publication=finalization["publication"],
                )
            else:
                try:
                    publication = publish_final_backbone(
                        exported_last_model, source_epoch=last_epoch
                    )
                except Exception as error:
                    publication = {
                        "success": False,
                        "reason": "publication_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                finalization = {
                    "exports_succeeded": True,
                    "publication": publication,
                }
                write_terminal_manifest(
                    model,
                    status="complete",
                    completion_reason=completion_reason,
                    source_epoch=last_epoch,
                    exports_succeeded=True,
                    publication=publication,
                )

        if model.hparams.training_spec.distributed:
            if model.rank is None:
                raise RuntimeError("Distributed training requires a process rank.")
            finalization = broadcast_publication_result(finalization, model.rank)
        if finalization is None or not finalization["exports_succeeded"]:
            error = None if finalization is None else finalization.get("error")
            raise RuntimeError(f"Complete-model export failed: {error}")

        publication = finalization["publication"]
        if publication.get("success"):
            model.logger.info(
                f"Published backbone revision {publication['revision_id']}."
            )
        elif publication.get("reason") == "compare_and_swap_conflict":
            model.logger.warning(
                "Backbone publication lost a compare-and-swap race; "
                "complete model exports remain valid."
            )
        elif publication.get("reason") == "publication_error":
            model.logger.warning(
                "Complete model exports succeeded, but backbone publication "
                f"failed: {publication.get('error')}"
            )
        model.logger.info("--- Training Complete ---")
        self.emit(
            RunCompleted(
                access=self.access,
                completion_reason=completion_reason,
            )
        )
        if model.hparams.training_spec.distributed:
            dist.barrier()
