"""Exact run-checkpoint save and staged restore services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sequifier.artifacts.model_artifact import build_model_artifact
from sequifier.artifacts.model_export import model_execution_config
from sequifier.artifacts.run_checkpoint import (
    RUN_CHECKPOINT_FORMAT_VERSION,
    IntegrationState,
    RunCheckpoint,
    RunCheckpointStore,
    load_run_checkpoint,
)
from sequifier.integration.contexts import CheckpointSaved, CheckpointSaving
from sequifier.training.loader_state import LoaderStateService


@dataclass(frozen=True)
class CheckpointRequest:
    suffix: str | None = "latest"
    identity: Any | None = None


@dataclass(frozen=True)
class LoadedRunCheckpoint:
    checkpoint: RunCheckpoint
    path: Path


class CheckpointCompatibility:
    """Validate the current schema and exact-resume runtime compatibility."""

    def validate(self, checkpoint: RunCheckpoint) -> None:
        checkpoint.validate()

    def validate_for_restore(
        self, checkpoint: RunCheckpoint, current_config: Any
    ) -> None:
        self.validate(checkpoint)
        saved = self._resume_settings(
            checkpoint.training_config,
            model_config=checkpoint.model.model_config.values,
        )
        current = self._resume_settings(current_config)
        mismatches = self._mismatches(saved, current)
        if mismatches:
            raise ValueError(
                "Checkpoint model/dataset topology or training plan does not match "
                "the current run. Use model initialization for a new run. "
                + "; ".join(mismatches[:20])
            )

    @staticmethod
    def _resume_settings(
        config: Any, *, model_config: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        execution = dict(model_config or model_execution_config(config))
        execution.pop("embedding_layer_names", None)
        training = config.global_training
        datasets = {
            name: {
                "model_interface": dataset.model_interface,
                "parts": list(dataset.parts),
                "criterion": dataset.criterion,
                "class_weights": dataset.class_weights,
                "loss_weights": dataset.loss_weights,
                "freeze": dataset.freeze.model_dump(mode="python"),
            }
            for name, dataset in config.dataset_training.items()
        }
        return {
            "model": execution,
            "resume_training": {
                "seed": config.seed,
                "read_format": training.read_format,
                "batch_size": training.batch_size,
                "accumulation_steps": training.accumulation_steps,
                "learning_rate": training.learning_rate,
                "optimizer": training.optimizer.model_dump(mode="python"),
                "scheduler": training.scheduler.model_dump(mode="python"),
                "scheduler_step_on": training.scheduler_step_on,
                "gradient_clip": training.gradient_clip,
                "distributed": training.distributed,
                "data_parallelism": training.data_parallelism,
                "world_size": training.world_size,
                "num_workers": training.num_workers,
                "load_full_data_to_ram": training.load_full_data_to_ram,
            },
            "datasets": datasets,
            "training_plan": [
                phase.model_dump(mode="python") for phase in config.training_plan
            ],
        }

    @classmethod
    def _mismatches(
        cls,
        saved: Any,
        current: Any,
        path: str = "",
    ) -> list[str]:
        if isinstance(saved, Mapping) and isinstance(current, Mapping):
            mismatches: list[str] = []
            for key in sorted(set(saved) | set(current), key=str):
                child_path = f"{path}.{key}" if path else str(key)
                if key not in saved:
                    mismatches.append(f"{child_path}: missing from checkpoint")
                elif key not in current:
                    mismatches.append(f"{child_path}: missing from current run")
                else:
                    mismatches.extend(
                        cls._mismatches(saved[key], current[key], child_path)
                    )
            return mismatches
        if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
            if len(saved) != len(current):
                return [
                    f"{path}: checkpoint length={len(saved)}, "
                    f"current length={len(current)}"
                ]
            mismatches = []
            for index, (saved_item, current_item) in enumerate(zip(saved, current)):
                mismatches.extend(
                    cls._mismatches(
                        saved_item,
                        current_item,
                        f"{path}[{index}]",
                    )
                )
            return mismatches
        if saved != current:
            return [f"{path}: checkpoint={saved!r}, current={current!r}"]
        return []


class CheckpointService:
    def __init__(
        self,
        store: RunCheckpointStore,
        distributed: Any,
        random_state: Any,
        loader_state: LoaderStateService,
        compatibility: CheckpointCompatibility | None = None,
    ) -> None:
        self.store = store
        self.distributed = distributed
        self.random_state = random_state
        self.loader_state = loader_state
        self.compatibility = compatibility or CheckpointCompatibility()

    def save(self, request: CheckpointRequest, run: Any) -> Path | None:
        path = self.store.path_for(request.suffix)
        access = run.optimization.access(run.network)
        run.integrations.emit(
            CheckpointSaving(access=access, identity=request.identity, path=path)
        )
        model_state = self.distributed.capture_model_state(run.network)
        optimizer_state = self.distributed.capture_optimizer_state(
            run.network, run.optimization.optimizer
        )
        random_state = self.random_state.gather(self.distributed)
        loader_state = self.loader_state.state_dict(run.datasets)
        integration_state = run.integrations.checkpoint_state_dict()
        self.distributed.barrier()
        if self.distributed.rank != 0:
            return None
        checkpoint = RunCheckpoint(
            format_version=RUN_CHECKPOINT_FORMAT_VERSION,
            model=build_model_artifact(
                run.network,
                run.config,
                state_dict=model_state,
                provenance={
                    "run_id": run.state.run_id,
                    "session_id": run.state.session_id,
                },
            ),
            optimization=run.optimization.state_dict(optimizer_state),
            run_state=run.state.state_dict(),
            random_state=random_state,
            loader_state=loader_state,
            integration_state=IntegrationState(values=integration_state),
            training_config=run.config,
        )
        self.compatibility.validate(checkpoint)
        written = self.store.save(checkpoint, request.suffix)
        run.integrations.emit(
            CheckpointSaved(access=access, identity=request.identity, path=written)
        )
        return written


class CheckpointRestorer:
    def __init__(self, compatibility: CheckpointCompatibility | None = None) -> None:
        self.compatibility = compatibility or CheckpointCompatibility()

    def load(self, path: Path) -> LoadedRunCheckpoint:
        checkpoint = load_run_checkpoint(path)
        self.compatibility.validate(checkpoint)
        return LoadedRunCheckpoint(checkpoint, path)

    def validate_for_restore(
        self, loaded: LoadedRunCheckpoint, current_config: Any
    ) -> None:
        self.compatibility.validate_for_restore(loaded.checkpoint, current_config)

    def restore_model(
        self, loaded: LoadedRunCheckpoint, network: Any, distributed: Any
    ) -> None:
        distributed.restore_model_state(
            network, loaded.checkpoint.model.model_state_dict
        )

    def restore_optimization(
        self,
        loaded: LoadedRunCheckpoint,
        optimization: Any,
        network: Any,
        distributed: Any,
    ) -> None:
        distributed.restore_optimizer_state(
            network, optimization.optimizer, loaded.checkpoint.optimization.optimizer
        )
        optimization.load_non_optimizer_state(loaded.checkpoint.optimization)

    def restore_runtime(self, loaded: LoadedRunCheckpoint, run: Any) -> None:
        from sequifier.training.state import RunState

        restored = RunState.from_state_dict(loaded.checkpoint.run_state)
        if restored.optimizer_step != loaded.checkpoint.optimization.optimizer_step:
            raise ValueError(
                "Checkpoint optimizer-step values disagree between run and "
                "optimization state."
            )
        run.state = restored
        run.optimization.optimizer_step = run.state.optimizer_step
        run.integrations.load_state_dict(loaded.checkpoint.integration_state.values)
        run.loader_state.load_state_dict(
            run.datasets,
            loaded.checkpoint.loader_state,
            resume_from_iterator_start=not restored.phase_epoch_complete,
        )

    def restore_randomness(self, loaded: LoadedRunCheckpoint, run: Any) -> None:
        if len(loaded.checkpoint.random_state.states) != run.distributed.world_size:
            raise ValueError(
                "Checkpoint random-state world size does not match execution: "
                f"{len(loaded.checkpoint.random_state.states)} != "
                f"{run.distributed.world_size}."
            )
        state = run.random.select_for_rank(
            loaded.checkpoint.random_state, run.distributed.rank
        )
        run.random.restore(state)
