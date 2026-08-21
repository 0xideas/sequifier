import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
import time  # noqa: E402
import uuid  # noqa: E402
import warnings  # noqa: E402
from typing import Any, Optional, Union  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from beartype import beartype  # noqa: E402
from loguru import logger as loguru_logger  # noqa: E402
from packaging import version  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.amp.grad_scaler import GradScaler  # noqa: E402
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.distributed.fsdp import (  # noqa: E402
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )
else:
    from torch.distributed._composable.fsdp import (  # noqa: E402
        MixedPrecisionPolicy,  # type: ignore
        OffloadPolicy,  # type: ignore
        fully_shard,  # type: ignore
    )

from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.nn import ModuleDict  # noqa: E402
from torch.nn.functional import one_hot  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

torch._dynamo.config.suppress_errors = True

ClassCounts = dict[str, Tensor]
CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = {2, 3}
EMBEDDING_INDEX_DTYPES = (torch.int32, torch.int64)
NARROW_EMBEDDING_INDEX_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
)
WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES = (torch.uint32, torch.uint64)

from sequifier.artifacts.model_config import (  # noqa: E402
    resolved_config_from_model_config,
)
from sequifier.artifacts.model_export import (  # noqa: E402
    model_execution_config,
    pt_bundle,
)
from sequifier.artifacts.run_checkpoint import checkpoint_path  # noqa: E402
from sequifier.config.composable_train_config import (  # noqa: E402
    ResolvedSequifierConfig as TrainModel,
)
from sequifier.config.composable_train_config import load_train_config  # noqa: E402
from sequifier.config.train_config import legacy_load_train_config  # noqa: E402
from sequifier.distributed.env import setup_distributed_env  # noqa: E402
from sequifier.helpers import (  # noqa: E402
    conditional_beartype,
    configure_determinism,
    configure_logger,
    construct_index_maps,
    get_torch_dtype,
    normalize_path,
)
from sequifier.integration import (  # noqa: E402
    BatchPrepared,
    CheckpointSaved,
    CheckpointSaving,
    ForwardCompleted,
    IntegrationManager,
    IntegrationSpec,
    LossComputed,
    RunCompleted,
    ValidationCompleted,
)
from sequifier.io.batch import SequifierBatch  # noqa: E402
from sequifier.io.sequifier_dataset_from_file import (  # noqa: E402
    SequifierDatasetFromFile,
)
from sequifier.io.sequifier_dataset_from_folder_parquet import (  # noqa: E402
    SequifierDatasetFromFolderParquet,
)
from sequifier.io.sequifier_dataset_from_folder_parquet_lazy import (  # noqa: E402
    SequifierDatasetFromFolderParquetLazy,
)
from sequifier.io.sequifier_dataset_from_folder_pt import (  # noqa: E402
    SequifierDatasetFromFolderPt,
)
from sequifier.io.sequifier_dataset_from_folder_pt_lazy import (  # noqa: E402
    SequifierDatasetFromFolderPtLazy,
)
from sequifier.logging_paths import (  # noqa: E402
    model_artifact_path,
    model_log_directory,
)
from sequifier.model.dtypes import cast_floating_to_module_dtype  # noqa: E402
from sequifier.model.embedding import (  # noqa: E402
    ONNX_EMBEDDING_LAYER_NAMES_KEY,
    parse_embedding_layer_name,
)
from sequifier.model.factory import (  # noqa: E402
    build_transformer_network,
    compile_composable_training_model,
    compile_unique_layers,
    wrap_composable_ddp,
)
from sequifier.model.layers import RMSNorm  # noqa: E402
from sequifier.model.model import SequifierModel  # noqa: E402
from sequifier.model.network import ComposableTransformerNetwork  # noqa: E402
from sequifier.model.parameter_catalog import (  # noqa: E402
    ParameterCatalog,
    semantic_optimizer_groups,
)
from sequifier.model.tracing import activate_trace_context  # noqa: E402
from sequifier.optimizers.optimizers import get_optimizer_class  # noqa: E402
from sequifier.special_tokens import ONNX_CATEGORICAL_TARGET_CODECS_KEY  # noqa: E402
from sequifier.training.distributed import (  # noqa: E402
    broadcast_initial_state,
    broadcast_publication_result,
    verify_loaded_revision,
)
from sequifier.training.engine import TrainingEngine  # noqa: E402
from sequifier.training.initial_state import (  # noqa: E402
    load_model_initial_state,
    select_initial_state,
)
from sequifier.training.lifecycle import (  # noqa: E402
    publish_final_backbone,
    write_terminal_manifest,
)
from sequifier.training.metrics import StructuredMetricWriters  # noqa: E402
from sequifier.training.session import TrainingSession  # noqa: E402
from sequifier.training.state import TrainingState  # noqa: E402


def cleanup():
    """Destroy the active distributed process group."""
    dist.destroy_process_group()


def _smallest_embedding_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the narrowest dtype accepted by torch embedding for this integer dtype."""
    if dtype in EMBEDDING_INDEX_DTYPES:
        return dtype
    if dtype in NARROW_EMBEDDING_INDEX_DTYPES:
        return torch.int32
    if dtype in WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES:
        return torch.int64
    raise TypeError(f"Embedding indices must use an integer dtype, got {dtype}.")


@beartype
def _class_index_tensor(indices: Tensor) -> Tensor:
    """Return integer class indices in the dtype required by PyTorch losses."""
    _smallest_embedding_safe_dtype(indices.dtype)
    if indices.dtype == torch.int64:
        return indices
    return indices.to(dtype=torch.int64)


@beartype
def create_dummy_data_and_metadata(
    config: Any, local_rank: int
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    dummy_data = {}
    for col in config.input_columns:
        dtype = torch.int64 if col in config.categorical_columns else torch.float32
        dummy_data[col] = torch.ones(
            (config.training_spec.batch_size, config.window_view.context_length),
            dtype=dtype,
            device=local_rank,
        )

    dummy_metadata = {
        "attention_valid_mask": torch.ones(
            (config.training_spec.batch_size, config.window_view.context_length),
            dtype=torch.bool,
            device=local_rank,
        )
    }
    return dummy_data, dummy_metadata


def _canonical_parameter_name(name: str) -> str:
    """Return a parameter name independent of compile and component DDP wrappers."""
    canonical = name.replace("_orig_mod.", "")
    if canonical.startswith("module."):
        canonical = canonical[len("module.") :]
    return canonical.replace(".module.", ".")


def _canonical_parameter_names(value: Any) -> Any:
    """Canonicalize a checkpoint parameter-name list when it has the expected form."""
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        return value
    return [_canonical_parameter_name(name) for name in value]


def _run_training_session(
    *,
    model: Any,
    config: TrainModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    checkpoint: dict[str, Any] | None,
    ddp_model: nn.Module | None,
    integration_specs: tuple[IntegrationSpec, ...],
    integration_instances: tuple[Any, ...],
) -> None:
    for index, group in enumerate(model.optimizer.param_groups):
        group.setdefault(
            "group_id",
            "all" if len(model.optimizer.param_groups) == 1 else f"group-{index}",
        )
    saved_state = (checkpoint.get("training_state") or {}) if checkpoint else {}
    if not isinstance(saved_state, dict):
        saved_state = {}
    state = TrainingState(
        epoch=int(saved_state.get("epoch", max(0, model.start_epoch - 1))),
        batch=int(saved_state.get("batch", model.start_batch)),
        global_batch_step=int(saved_state.get("global_batch_step", 0)),
        optimizer_step=int(saved_state.get("optimizer_step", 0)),
        accumulation_index=int(saved_state.get("accumulation_index", 0)),
        best_validation_loss=float(
            saved_state.get("best_validation_loss", model._resume_best_val_loss)
        ),
        epochs_without_improvement=int(
            saved_state.get(
                "epochs_without_improvement", model._resume_n_epochs_no_improvement
            )
        ),
        run_id=model.run_id,
        session_id=model.session_id,
    )
    integrations = IntegrationManager(
        specs=integration_specs,
        instances=integration_instances,
        rank=int(model.rank or 0),
        world_size=(
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        ),
        distributed=config.training_spec.distributed,
    )
    integrations.validate_execution(
        torch_compile=config.training_spec.torch_compile,
        data_parallelism=config.training_spec.data_parallelism,
    )
    engine = TrainingEngine(
        model=model,
        objective=model.objective,
        criteria=model.criterion,
        optimizer=model.optimizer,
        scheduler=model.scheduler,
        scaler=model.scaler,
        state=state,
        integrations=integrations,
    )
    session = TrainingSession(
        config=config,
        model=model,
        engine=engine,
        train_loader=train_loader,
        validation_loader=valid_loader,
        integrations=integrations,
    )
    session.restore_integration_state(checkpoint)
    session.run(ddp_model=ddp_model)


def _optimizer_parameters(model: Any, *, semantic_grouping: bool) -> Any:
    parameters = tuple(model.parameters_to_optimize())
    if not semantic_grouping:
        return parameters
    return semantic_optimizer_groups(
        ParameterCatalog(model), parameters={id(parameter) for parameter in parameters}
    )


def _train_composable_worker(
    local_rank: int,
    world_size: int,
    config: Any,
    global_rank: int,
    integration_specs: tuple[IntegrationSpec, ...],
    integration_instances: tuple[Any, ...],
    semantic_optimizer_grouping: bool,
) -> None:
    """Build dataset/source runtimes and execute the canonical training plan."""

    from sequifier.training.runtime import build_dataset_runtimes

    configure_logger(
        config.project_root,
        config.model_name,
        global_rank,
        dataset_names=tuple(config.dataset_training_spec),
        rank_specific=config.global_training_spec.distributed,
    )
    if config.global_training_spec.distributed:
        if config.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        setup_distributed_env(
            global_rank,
            local_rank,
            world_size,
            config.global_training_spec.backend,
        )
    configure_determinism(config.seed, config.global_training_spec.enforce_determinism)
    model = TransformerModel(config, rank=global_rank, local_rank=local_rank)
    if config.global_training_spec.distributed:
        selected_source = broadcast_initial_state(
            select_initial_state(config) if global_rank == 0 else None,
            global_rank,
        )
    else:
        selected_source = select_initial_state(config)
    checkpoint = load_model_initial_state(model, selected_source)
    if config.global_training_spec.distributed:
        verify_loaded_revision(model._backbone_parent_revision_id)
    network = model.network
    if not isinstance(network, ComposableTransformerNetwork):
        raise TypeError("Canonical training requires a composable network")
    dataset_runtimes = build_dataset_runtimes(
        config, network, torch.device(model.device)
    )
    model.activate_dataset(
        next(iter(dataset_runtimes)), next(iter(dataset_runtimes.values()))
    )
    if global_rank == 0:
        dataset_count = len(config.dataset_training_spec)
        evaluated_datasets = {source.dataset for source in config.evaluation_sources}
        model.metric_writers_by_dataset = {
            name: StructuredMetricWriters(
                config.project_root,
                config.model_name,
                global_rank,
                class_share_columns=dataset.class_share_log_columns,
                dataset_name=name,
                dataset_count=dataset_count,
                validation_enabled=name in evaluated_datasets,
            )
            for name, dataset in config.dataset_training_spec.items()
        }
    model._semantic_optimizer_grouping = semantic_optimizer_grouping
    if checkpoint is not None:
        model._validate_checkpoint_compatibility(checkpoint, 0)

    callable_model: nn.Module = model
    ddp_model = None
    if (
        config.global_training_spec.distributed
        and config.global_training_spec.data_parallelism == "FSDP"
    ):
        mesh = init_device_mesh("cuda", (world_size,))
        model._data_parallel_group = mesh.get_group()
        fsdp_kwargs: dict[str, Any] = {"mesh": mesh}
        if config.global_training_spec.layer_autocast:
            amp_dtype = get_torch_dtype(
                config.global_training_spec.layer_type_dtypes.get("linear", "bfloat16")
                if config.global_training_spec.layer_type_dtypes
                else "bfloat16"
            )
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=amp_dtype,
                reduce_dtype=amp_dtype,
                output_dtype=amp_dtype,
            )
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy()
        if config.global_training_spec.fsdp_cpu_offload:
            fsdp_kwargs["offload_policy"] = OffloadPolicy()

        sharded_layer_ids: set[int] = set()
        for layer in model.layers:
            if id(layer) in sharded_layer_ids:
                continue
            fully_shard(layer, **fsdp_kwargs)
            sharded_layer_ids.add(id(layer))
        fully_shard(model, **fsdp_kwargs)
        dist.barrier()

        params_to_optimize = _optimizer_parameters(
            model, semantic_grouping=semantic_optimizer_grouping
        )
        model.initialize_optimizer(params=params_to_optimize)
        if checkpoint is not None:
            set_optimizer_state_dict(
                model,
                model.optimizer,
                optim_state_dict=checkpoint["optimizer_state_dict"],
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        if config.global_training_spec.torch_compile == "inner":
            compile_unique_layers(model.layers)
        dummy_data, dummy_metadata = create_dummy_data_and_metadata(config, local_rank)
        with torch.no_grad():
            _ = model(dummy_data, dummy_metadata, False)
        dist.barrier()
    else:
        params_to_optimize = _optimizer_parameters(
            model, semantic_grouping=semantic_optimizer_grouping
        )
        model.initialize_optimizer(params=params_to_optimize)
        if config.global_training_spec.torch_compile != "none":
            callable_model = compile_composable_training_model(model, config)
    if checkpoint is not None:
        if config.global_training_spec.data_parallelism != "FSDP":
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        model._apply_checkpoint_training_state(
            checkpoint.get("scaler_state_dict"),
            checkpoint.get("best_val_loss", float("inf")),
            checkpoint.get("n_epochs_no_improvement", 0),
            checkpoint.get("best_model_state_dict"),
            checkpoint.get("rng_state"),
            checkpoint.get("data_loader_generator_states"),
            checkpoint.get("run_id"),
        )
    if (
        config.global_training_spec.distributed
        and config.global_training_spec.data_parallelism == "DDP"
    ):
        ddp_model = wrap_composable_ddp(model, config, local_rank)
        if ddp_model is not None:
            callable_model = ddp_model

    saved_state = checkpoint.get("training_state", {}) if checkpoint else {}
    state_fields = TrainingState.__dataclass_fields__
    state = TrainingState(
        **{key: value for key, value in saved_state.items() if key in state_fields}
    )
    if not checkpoint:
        state.run_id = model.run_id
    model.run_id = state.run_id
    state.session_id = model.session_id
    integrations = IntegrationManager(
        specs=integration_specs,
        instances=integration_instances,
        rank=global_rank,
        world_size=world_size,
        distributed=config.global_training_spec.distributed,
    )
    integrations.validate_execution(
        torch_compile=config.global_training_spec.torch_compile,
        data_parallelism=config.global_training_spec.data_parallelism,
    )
    if checkpoint is not None:
        integrations.load_state_dict(checkpoint.get("integration_state"))
        model._restore_rng_state()
    engine = TrainingEngine(
        model=model,
        objective=model.objective,
        criteria=model.criterion,
        optimizer=model.optimizer,
        scheduler=model.scheduler,
        scaler=model.scaler,
        state=state,
        integrations=integrations,
    )
    object.__setattr__(model, "_training_engine", engine)
    model.logger.info(
        f"--- Starting Training for model: {model.model_name} | "
        f"run: {model.run_id} | session: {model.session_id} ---"
    )
    completion_reason = "normal_completion"
    try:
        completion_reason = engine.run_plan(
            config,
            dataset_runtimes,
            ddp_model=(callable_model if callable_model is not model else None),
        )
    except KeyboardInterrupt:
        completion_reason = "keyboard_interruption"
        model.logger.warning("Training interrupted; exporting final state.")
    except BaseException as error:
        if global_rank == 0:
            is_pruned = isinstance(error, SystemExit) and error.code == 143
            write_terminal_manifest(
                model,
                status="pruned" if is_pruned else "failed",
                completion_reason=("optuna_pruning" if is_pruned else "exception"),
                source_epoch=state.epoch,
                exports_succeeded=False,
                publication={"success": False, "reason": "not_attempted"},
            )
        raise
    if config.global_training_spec.distributed:
        dist.barrier()
    last_model_state = model._get_full_state_dict(
        callable_model if callable_model is not model else None
    )
    best_model_state = engine.best_model_state_dict
    if best_model_state is None:
        if global_rank == 0:
            model.logger.info(
                "No validation improvement... Saving last model as 'best'."
            )
        best_model_state = last_model_state

    finalization: dict[str, Any] | None = None
    if global_rank == 0:
        try:
            exported_last_model = model._export(
                last_model_state, "last", state.epoch, clean=True
            )
            model._export(best_model_state, "best", state.epoch, clean=True)
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
                source_epoch=state.epoch,
                exports_succeeded=False,
                publication=finalization["publication"],
            )
        else:
            try:
                publication = publish_final_backbone(
                    exported_last_model, source_epoch=state.epoch
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
                source_epoch=state.epoch,
                exports_succeeded=True,
                publication=publication,
            )

    if config.global_training_spec.distributed:
        finalization = broadcast_publication_result(finalization, global_rank)
    if finalization is None or not finalization["exports_succeeded"]:
        error = None if finalization is None else finalization.get("error")
        raise RuntimeError(f"Complete-model export failed: {error}")

    publication = finalization["publication"]
    if publication.get("success"):
        model.logger.info(f"Published backbone revision {publication['revision_id']}.")
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
    engine.emit(
        RunCompleted(
            access=engine.access,
            completion_reason=completion_reason,
        )
    )
    if config.global_training_spec.distributed:
        dist.barrier()
        cleanup()


@beartype
@loguru_logger.catch(message="Training worker failed", reraise=True)
def train_worker(
    local_rank: int,
    world_size: int,
    config: Any,
    from_folder: bool,
    global_rank: int,
    torch_compile: str,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
):
    """Run one local distributed-training worker."""
    if hasattr(config, "dataset_training_spec"):
        return _train_composable_worker(
            local_rank,
            world_size,
            config,
            global_rank,
            integration_specs,
            integration_instances,
            semantic_optimizer_grouping,
        )
    logger = configure_logger(config.project_root, config.model_name, global_rank)
    data_path = config.data_path
    if data_path is None:
        raise ValueError("data_path must be provided or resolved from metadata")

    if config.training_spec.distributed:
        if config.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        setup_distributed_env(
            global_rank, local_rank, world_size, config.training_spec.backend
        )

    if from_folder:
        if config.read_format == "pt":
            if config.training_spec.load_full_data_to_ram:
                train_dataset = SequifierDatasetFromFolderPt(data_path, config)
                valid_dataset = SequifierDatasetFromFolderPt(
                    config.validation_data_path, config
                )
            else:
                train_dataset = SequifierDatasetFromFolderPtLazy(data_path, config)
                valid_dataset = SequifierDatasetFromFolderPtLazy(
                    config.validation_data_path, config
                )
        elif config.read_format == "parquet":
            if config.training_spec.load_full_data_to_ram:
                train_dataset = SequifierDatasetFromFolderParquet(data_path, config)
                valid_dataset = SequifierDatasetFromFolderParquet(
                    config.validation_data_path, config
                )
            else:
                train_dataset = SequifierDatasetFromFolderParquetLazy(data_path, config)
                valid_dataset = SequifierDatasetFromFolderParquetLazy(
                    config.validation_data_path, config
                )
        else:
            raise Exception("Not allowed")

    else:
        if config.training_spec.distributed:
            raise ValueError(
                "Distributed training is not supported with single-file datasets."
            )
        train_dataset = SequifierDatasetFromFile(data_path, config)
        valid_dataset = SequifierDatasetFromFile(config.validation_data_path, config)

    configure_determinism(config.seed, config.training_spec.enforce_determinism)

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(config.seed + 10_001)
    valid_loader_generator = torch.Generator()
    valid_loader_generator.manual_seed(config.seed + 10_002)

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Batching is handled natively by the IterableDataset
        sampler=None,  # Sharding is handled natively by the IterableDataset
        num_workers=config.training_spec.num_workers,
        pin_memory=config.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
        generator=train_loader_generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        sampler=None,
        num_workers=config.training_spec.num_workers,
        pin_memory=config.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
        generator=valid_loader_generator,
    )

    model = TransformerModel(config, rank=global_rank, local_rank=local_rank)
    model._semantic_optimizer_grouping = semantic_optimizer_grouping
    if config.training_spec.distributed:
        run_id_object = [model.run_id if global_rank == 0 else None]
        dist.broadcast_object_list(run_id_object, src=0)
        model.run_id = str(run_id_object[0])
    model._data_loader_generators = {
        "train": train_loader_generator,
        "valid": valid_loader_generator,
    }
    base_model = model

    if config.training_spec.distributed:
        selected_source = broadcast_initial_state(
            select_initial_state(config) if global_rank == 0 else None,
            global_rank,
        )
    else:
        selected_source = select_initial_state(config)
    checkpoint = load_model_initial_state(model, selected_source)
    if config.training_spec.distributed:
        verify_loaded_revision(model._backbone_parent_revision_id)
    if checkpoint is not None:
        model._validate_checkpoint_compatibility(checkpoint, len(train_loader))

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Initialize Optimizer
    if not config.training_spec.distributed:
        params_to_optimize = _optimizer_parameters(
            model, semantic_grouping=semantic_optimizer_grouping
        )
        model.initialize_optimizer(params=params_to_optimize)

        if checkpoint is not None:
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            base_model.start_epoch, base_model.start_batch = _checkpoint_start_position(
                checkpoint, len(train_loader)
            )
            model._apply_checkpoint_training_state(
                checkpoint.get("scaler_state_dict"),
                checkpoint.get("best_val_loss", float("inf")),
                checkpoint.get("n_epochs_no_improvement", 0),
                checkpoint.get("best_model_state_dict"),
                checkpoint.get("rng_state"),
                checkpoint.get("data_loader_generator_states"),
                checkpoint.get("run_id"),
            )
        else:
            model.start_epoch = 1
            model.start_batch = 0
            if model._freezing_active:
                trainable_params = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters "
                    f"({format_number(trainable_params)} trainable)."
                )
            else:
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters."
                )

        if config.device.startswith("cuda"):
            if torch_compile == "outer":
                model = torch.compile(model)
            elif torch_compile == "inner":
                compile_unique_layers(model.layers)

        if checkpoint is not None:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()

        _run_training_session(
            model=base_model,
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=checkpoint,
            ddp_model=model if model is not base_model else None,
            integration_specs=integration_specs,
            integration_instances=integration_instances,
        )
    elif config.training_spec.data_parallelism == "FSDP":
        mesh = init_device_mesh(
            "cuda", (world_size,)
        )  # 1D mesh for standard ZeRO-3 full sharding
        model._data_parallel_group = mesh.get_group()

        fsdp_kwargs = {"mesh": mesh}
        if config.training_spec.layer_autocast:
            amp_dtype = get_torch_dtype(
                config.training_spec.layer_type_dtypes.get("linear", "bfloat16")
                if config.training_spec.layer_type_dtypes
                else "bfloat16"
            )

            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=amp_dtype,
                reduce_dtype=amp_dtype,
                output_dtype=amp_dtype,
            )
        else:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy()

        if config.training_spec.fsdp_cpu_offload:
            fsdp_kwargs["offload_policy"] = OffloadPolicy()

        sharded_layer_ids: set[int] = set()
        for layer in model.layers:
            layer_id = id(layer)
            if layer_id in sharded_layer_ids:
                continue
            fully_shard(layer, **fsdp_kwargs)
            sharded_layer_ids.add(layer_id)

        fully_shard(model, **fsdp_kwargs)
        dist.barrier()

        params_to_optimize = _optimizer_parameters(
            model, semantic_grouping=semantic_optimizer_grouping
        )
        model.initialize_optimizer(params=params_to_optimize)

        if checkpoint is not None:
            model.start_epoch, model.start_batch = _checkpoint_start_position(
                checkpoint, len(train_loader)
            )
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            set_optimizer_state_dict(
                base_model,
                base_model.optimizer,
                optim_state_dict=checkpoint["optimizer_state_dict"],
                options=options,
            )
            base_model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            model._apply_checkpoint_training_state(
                checkpoint.get("scaler_state_dict"),
                checkpoint.get("best_val_loss", float("inf")),
                checkpoint.get("n_epochs_no_improvement", 0),
                checkpoint.get("best_model_state_dict"),
                checkpoint.get("rng_state"),
                checkpoint.get("data_loader_generator_states"),
                checkpoint.get("run_id"),
            )

        else:
            model.start_epoch = 1
            model.start_batch = 0
            if model._freezing_active:
                trainable_params = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters "
                    f"({format_number(trainable_params)} trainable)."
                )
            else:
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters."
                )

        if config.device.startswith("cuda"):
            if torch_compile == "inner":
                compile_unique_layers(model.layers)

        if config.device.startswith("cuda"):
            dummy_data, dummy_metadata = create_dummy_data_and_metadata(
                config, local_rank
            )
            with torch.no_grad():
                _ = model(dummy_data, dummy_metadata, False)

            dist.barrier()

        if checkpoint is not None:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()

        _run_training_session(
            model=base_model,
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=checkpoint,
            ddp_model=base_model,
            integration_specs=integration_specs,
            integration_instances=integration_instances,
        )
        cleanup()
    elif config.training_spec.data_parallelism == "DDP":  # DDP
        params_to_optimize = _optimizer_parameters(
            model, semantic_grouping=semantic_optimizer_grouping
        )
        model.initialize_optimizer(params=params_to_optimize)

        if checkpoint is not None:
            base_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            base_model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            base_model.start_epoch, base_model.start_batch = _checkpoint_start_position(
                checkpoint, len(train_loader)
            )
            base_model._apply_checkpoint_training_state(
                checkpoint.get("scaler_state_dict"),
                checkpoint.get("best_val_loss", float("inf")),
                checkpoint.get("n_epochs_no_improvement", 0),
                checkpoint.get("best_model_state_dict"),
                checkpoint.get("rng_state"),
                checkpoint.get("data_loader_generator_states"),
                checkpoint.get("run_id"),
            )
        else:
            model.start_epoch = 1
            model.start_batch = 0
            if model._freezing_active:
                trainable_params = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters "
                    f"({format_number(trainable_params)} trainable)."
                )
            else:
                logger.info(
                    "Initializing new model with "
                    f"{format_number(pytorch_total_params)} parameters."
                )

        if config.device.startswith("cuda"):
            if torch_compile == "outer":
                model = torch.compile(model)

        device_ids = [local_rank] if config.device.startswith("cuda") else None
        ddp_model = DDP(model, device_ids=device_ids, find_unused_parameters=False)

        if config.device.startswith("cuda"):
            dummy_data, dummy_metadata = create_dummy_data_and_metadata(
                config, local_rank
            )

            if config.training_spec.layer_autocast:
                with (
                    torch.no_grad(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    _ = ddp_model(dummy_data, dummy_metadata, False)
            else:
                with torch.no_grad():
                    _ = ddp_model(dummy_data, dummy_metadata, False)

            dist.barrier()
        if checkpoint is not None:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()
        _run_training_session(
            model=base_model,
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=checkpoint,
            ddp_model=ddp_model,
            integration_specs=integration_specs,
            integration_instances=integration_instances,
        )
        cleanup()
    else:
        raise ValueError("For data_parallelism, only 'FSDP' and 'DDP' are supported")


@beartype
def _mp_train_worker_wrapper(
    local_rank: int,
    world_size: int,
    config: Any,
    from_folder: bool,
    torch_compile: str,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    semantic_optimizer_grouping: bool = False,
):
    train_worker(
        local_rank,
        world_size,
        config,
        from_folder,
        global_rank=local_rank,
        torch_compile=torch_compile,
        integration_specs=integration_specs,
        semantic_optimizer_grouping=semantic_optimizer_grouping,
    )


@beartype
def run_training(
    config: Any,
    *,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
) -> None:
    if hasattr(config, "dataset_training_spec"):
        spec = config.global_training_spec
        if spec.distributed and integration_instances:
            raise ValueError(
                "Distributed runs require IntegrationSpec; direct instances cannot "
                "be transferred to workers."
            )
        torch.set_float32_matmul_precision(spec.float32_matmul_precision)
        world_size = spec.world_size
        if spec.distributed:
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                global_rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                train_worker(
                    local_rank,
                    world_size,
                    config,
                    False,
                    global_rank,
                    spec.torch_compile,
                    integration_specs,
                    (),
                    semantic_optimizer_grouping,
                )
            else:
                mp.spawn(
                    _mp_train_worker_wrapper,
                    args=(
                        world_size,
                        config,
                        False,
                        spec.torch_compile,
                        integration_specs,
                        semantic_optimizer_grouping,
                    ),
                    nprocs=world_size,
                    join=True,
                )
        else:
            train_worker(
                0,
                1,
                config,
                False,
                0,
                spec.torch_compile,
                integration_specs,
                integration_instances,
                semantic_optimizer_grouping,
            )
        return
    data_path = config.data_path
    if data_path is None:
        raise ValueError("data_path must be provided or resolved from metadata")
    if config.training_spec.distributed and integration_instances:
        raise ValueError(
            "Distributed runs require IntegrationSpec; direct instances cannot "
            "be transferred to workers."
        )

    torch.set_float32_matmul_precision(config.training_spec.float32_matmul_precision)

    world_size = config.training_spec.world_size
    from_folder = os.path.isdir(normalize_path(data_path, config.project_root))

    if config.training_spec.distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Launched via torchrun / srun for multi-node distributed training
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            train_worker(
                local_rank,
                world_size,
                config,
                from_folder,
                global_rank,
                config.training_spec.torch_compile,
                integration_specs,
                (),
                semantic_optimizer_grouping,
            )
        else:
            # Single-node multi-GPU fallback using mp.spawn
            try:
                mp.spawn(
                    _mp_train_worker_wrapper,
                    args=(
                        world_size,
                        config,
                        from_folder,
                        config.training_spec.torch_compile,
                        integration_specs,
                        semantic_optimizer_grouping,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            except mp.ProcessExitedException as e:
                # Catch the specific PyTorch exception and check the exit_code attribute
                if e.exit_code == 143:
                    sys.exit(143)
                else:
                    raise e
    else:
        train_worker(
            0,
            1,
            config,
            from_folder,
            0,
            config.training_spec.torch_compile,
            integration_specs,
            integration_instances,
            semantic_optimizer_grouping,
        )


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    """Load train config and launch local or distributed training."""
    config_path = args.config_path or "configs/train.yaml"
    loader = (
        legacy_load_train_config
        if os.getenv("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN") == "1"
        else load_train_config
    )
    config = loader(config_path, args_config, args.skip_metadata)
    run_training(config)


@beartype
def format_number(number: int | float | np.float32) -> str:
    value = float(number)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return f"{value: .2e}"


def _get_evaluation_loss_mask(metadata: dict[str, Tensor]) -> Tensor:
    """Build the effective loss mask from token, objective, and sample masks."""
    valid_mask = metadata["target_valid_mask"].bool()

    if "bert_mask" in metadata:
        valid_mask = valid_mask & metadata["bert_mask"].bool()

    if "sample_valid_mask" in metadata:
        sample_valid_mask = metadata["sample_valid_mask"].bool()

        if sample_valid_mask.ndim != 1:
            raise ValueError("sample_valid_mask must have shape [batch_size].")
        if sample_valid_mask.shape[0] != valid_mask.shape[0]:
            raise ValueError(
                "sample_valid_mask batch dimension does not match target_valid_mask."
            )

        valid_mask = valid_mask & sample_valid_mask.unsqueeze(1)

    return valid_mask


@beartype
def _checkpoint_start_position(
    checkpoint: dict[str, Any], num_batches: int
) -> tuple[int, int]:
    """Return the next epoch/batch position after a saved checkpoint."""
    if checkpoint["batch"] + 1 >= num_batches:
        return checkpoint["epoch"] + 1, 0
    return checkpoint["epoch"], checkpoint["batch"] + 1


def _update_file_metadata_hash(hasher: Any, file_path: str) -> None:
    """Hash file identity metadata without reading the file contents."""
    normalized_path = os.path.abspath(file_path)
    file_stat = os.stat(normalized_path)
    hasher.update(normalized_path.encode("utf-8"))
    hasher.update(str(file_stat.st_size).encode("utf-8"))
    hasher.update(str(file_stat.st_mtime_ns).encode("utf-8"))


@beartype
def accumulate_class_counts(
    counts: ClassCounts,
    output: dict[str, Tensor],
    valid_mask: Tensor,
    n_classes: dict[str, int],
) -> None:
    """Accumulates predicted class counts over valid evaluation tokens."""
    flattened_mask = valid_mask.bool().T.contiguous().reshape(-1)

    for col, running_counts in counts.items():
        if col not in output:
            raise RuntimeError(f"Output is missing class-share column {col!r}.")

        predicted_ids = output[col].argmax(dim=-1).contiguous().reshape(-1)

        if predicted_ids.numel() != flattened_mask.numel():
            raise RuntimeError(
                f"Prediction/mask size mismatch for {col!r}: "
                f"{predicted_ids.numel()} predictions versus "
                f"{flattened_mask.numel()} mask entries."
            )

        valid_predictions = predicted_ids[flattened_mask]

        if valid_predictions.numel() == 0:
            continue

        batch_counts = torch.bincount(
            valid_predictions.to(torch.int64),
            minlength=n_classes[col],
        )

        if batch_counts.numel() != running_counts.numel():
            raise RuntimeError(
                f"Class-count size mismatch for {col!r}: "
                f"{batch_counts.numel()} counts versus "
                f"{running_counts.numel()} expected classes."
            )

        running_counts.add_(batch_counts)


class TransformerEmbeddingModel(nn.Module):
    """Embedding-only wrapper for TransformerModel."""

    def __init__(self, transformer_model: "TransformerModel"):
        super().__init__()
        self.transformer_model = transformer_model
        self.logger = self.transformer_model.logger

    @beartype
    def _copy_model(self):
        """Deep-copy without copying the logger handle."""
        logger_ref = self.transformer_model.logger
        del self.transformer_model.logger
        del self.logger
        model_copy = copy.deepcopy(self)
        model_copy.transformer_model._initialize_log_file()
        self.transformer_model.logger = logger_ref
        self.logger = self.transformer_model.logger
        return model_copy

    @conditional_beartype
    def forward(self, src: dict[str, Tensor], metadata: dict[str, Tensor]):
        """Return embedding output from the wrapped model."""
        return self.transformer_model.forward_embed(src, metadata=metadata)


class _OnnxExportWrapper(nn.Module):
    def __init__(
        self,
        model: Union["TransformerModel", TransformerEmbeddingModel],
        feature_columns: list[str],
    ):
        super().__init__()
        self.model = model
        self.feature_columns = feature_columns

    def forward(self, *inputs: Tensor):
        features = dict(zip(self.feature_columns, inputs[:-1]))
        metadata = {"attention_valid_mask": inputs[-1]}
        return self.model(features, metadata=metadata)


class TransformerModel(SequifierModel):
    """Sequifier transformer plus train/eval/export routines."""

    @beartype
    def __init__(
        self, hparams: Any, rank: Optional[int] = None, local_rank: Optional[int] = None
    ):
        """Build model modules and training state from config."""
        super().__init__()
        self.project_root = hparams.project_root
        self._composable = hasattr(hparams, "dataset_training_spec")
        self.active_dataset_name = (
            next(iter(hparams.dataset_training_spec)) if self._composable else None
        )
        self.active_interface_name = (
            hparams.dataset_training_spec[self.active_dataset_name].model_interface
            if self._composable
            else None
        )
        self.model_type = "Transformer"

        self.rank = rank

        self.model_name = hparams.model_name or uuid.uuid4().hex[:8]
        self.run_id = uuid.uuid4().hex
        self.session_id = uuid.uuid4().hex
        self.metric_writers: Optional[StructuredMetricWriters] = None
        self.metric_writers_by_dataset: dict[str, StructuredMetricWriters] = {}
        self._log_dataset_names = (
            tuple(hparams.dataset_training_spec) if self._composable else ()
        )
        self._rank_specific_logs = bool(hparams.training_spec.distributed)

        self._initialize_log_file()

        self.input_columns = hparams.input_columns
        self.categorical_columns = [
            col
            for col in hparams.categorical_columns
            if self.input_columns is None or col in self.input_columns
        ]
        self.real_columns = [
            col
            for col in hparams.real_columns
            if self.input_columns is None or col in self.input_columns
        ]
        self.logger.info(f"{self.categorical_columns = }")
        self.logger.info(f"{self.real_columns = }")

        self.target_columns = hparams.target_columns
        self.target_column_types = hparams.target_column_types
        self.loss_weights = hparams.training_spec.loss_weights
        self.storage_layout = hparams.storage_layout
        self.window_view = hparams.window_view
        self.context_length = hparams.window_view.context_length
        self.n_classes = hparams.n_classes
        self.inference_batch_size = hparams.inference_batch_size
        self.log_interval = hparams.training_spec.log_interval
        self.class_share_log_columns = hparams.training_spec.class_share_log_columns
        self.index_maps = construct_index_maps(
            hparams.id_maps, self.class_share_log_columns, True
        )
        self.export_embedding_model = hparams.export_embedding_model
        self.embedding_layer_names = tuple(hparams.embedding_layer_names)
        embedding_selectors = tuple(
            parse_embedding_layer_name(name) for name in self.embedding_layer_names
        )
        self._embedding_backbone_layer_indices = tuple(
            selector.index
            for selector in embedding_selectors
            if selector.source == "backbone_layer" and selector.index is not None
        )
        self._embedding_capture_final_norm = any(
            selector.source == "backbone_final_norm" for selector in embedding_selectors
        )
        decoder_block_indices: dict[str, list[int]] = {}
        for selector in embedding_selectors:
            if (
                selector.source == "decoder_hidden_block"
                and selector.branch is not None
                and selector.index is not None
            ):
                decoder_block_indices.setdefault(selector.branch, []).append(
                    selector.index
                )
        self._embedding_decoder_block_indices = {
            branch: tuple(indices) for branch, indices in decoder_block_indices.items()
        }
        self.export_generative_model = hparams.export_generative_model
        self.export_onnx = hparams.export_onnx
        self.export_pt = hparams.export_pt
        self.export_with_dropout = False
        self.early_stopping_epochs = hparams.training_spec.early_stopping_epochs
        self.hparams = hparams
        self.device = hparams.device
        self.device_max_concat_length = hparams.training_spec.device_max_concat_length

        if hparams.device.startswith("cuda"):
            if local_rank is not None:
                self.device = f"cuda:{local_rank}"
            elif self.rank is not None:  # Backwards compatibility
                self.device = f"cuda:{self.rank}"
            else:
                self.device = hparams.device
        else:
            self.device = hparams.device

        built_model = build_transformer_network(
            hparams,
            device=torch.device(self.device),
            logger=self.logger,
        )
        network = built_model.network
        self.objective = built_model.objective
        self.dim_model = network.dim_model
        self.backbone = network.backbone
        if isinstance(network, ComposableTransformerNetwork):
            self.interfaces = network.interfaces
            assert self.active_interface_name is not None
            route = network.resolve_interface(self.active_interface_name)
            self.prediction_length = route.prediction_length
            self.decoding_support = route.decoding_support
            interface_metadata = built_model.runtime_metadata.interfaces[
                self.active_interface_name
            ]
        else:
            self.ingestion = network.ingestion
            self.ingestion_adapter = network.ingestion_adapter
            self.decoder = network.decoder
            self.prediction_length = network.prediction_length
            self.decoding_support = network.decoding_support
            interface_metadata = next(
                iter(built_model.runtime_metadata.interfaces.values())
            )
        self.decoded_context_length = self.context_length - self.decoding_support + 1
        self.interface_runtime_metadata = built_model.runtime_metadata.interfaces
        self.target_decoder_ids = interface_metadata.target_decoder_ids
        self.target_n_classes = interface_metadata.target_n_classes
        self.target_global_to_decoder = interface_metadata.target_global_to_decoder
        self.softmax: dict[str, nn.Module] = {}
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                self.softmax[target_column] = nn.LogSoftmax(dim=-1)
            elif target_column_type != "real":
                raise ValueError(
                    f"Target column type {target_column_type} not in "
                    "['categorical', 'real']"
                )

        self.criterion = self._init_criterion(hparams=hparams)
        self.batch_size = hparams.training_spec.batch_size
        self.accumulation_steps = hparams.training_spec.accumulation_steps

        self.register_buffer(
            "src_mask",
            network.attention_mask_policy.detach().clone(),
            persistent=False,
        )
        self._freezing_active = (
            any(
                dataset.freezing.active
                for dataset in hparams.dataset_training_spec.values()
            )
            if self._composable
            else any(
                config.has_freezing_policy
                for config in (
                    hparams.model_spec.ingestion,
                    hparams.model_spec.backbone,
                    hparams.model_spec.decoder,
                )
            )
        )

        self.scheduler_step_on = hparams.training_spec.scheduler_step_on

        self.save_interval_epochs = hparams.training_spec.save_interval_epochs
        self.save_latest_interval_minutes = (
            hparams.training_spec.save_latest_interval_minutes
        )
        self.save_interval_minutes = hparams.training_spec.save_interval_minutes
        self.save_interval_batches = hparams.training_spec.save_interval_batches
        self.save_interval_val_loss = hparams.training_spec.save_interval_val_loss
        use_scaler = False
        if hparams.training_spec.layer_type_dtypes:
            if "float16" in hparams.training_spec.layer_type_dtypes.values():
                use_scaler = True

        self.scaler = GradScaler(device=self.device.split(":")[0], enabled=use_scaler)
        self._resume_best_val_loss = float("inf")
        self._resume_n_epochs_no_improvement = 0
        self._resume_best_model_state_dict = None
        self._resume_rng_state = None
        self._resume_data_loader_generator_states = None
        self._data_loader_generators: dict[str, torch.Generator] = {}
        self.start_epoch = 1
        self.start_batch = 0
        self._backbone_parent_revision_id: Optional[str] = None

        self._apply_layer_dtypes()

        self.to(self.device)
        for criterion in self.criterion.values():
            criterion.to(self.device)

        object.__setattr__(self, "network", network)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in {"ingestion", "ingestion_adapter", "decoder"}:
                modules = self.__dict__.get("_modules", {})
                interfaces = modules.get("interfaces")
                interface_name = self.__dict__.get("active_interface_name")
                if interfaces is not None and interface_name is not None:
                    return getattr(interfaces[interface_name], name)
            raise

    def activate_dataset(self, name: str, runtime: Any | None = None) -> None:
        """Activate one dataset's interface, loss, and metric policy."""

        if not self._composable:
            return
        dataset = self.hparams.dataset_training_spec[name]
        self.active_dataset_name = name
        self.active_interface_name = dataset.model_interface
        interface = dataset.interface
        self.input_columns = interface.input_columns
        self.categorical_columns = interface.categorical_columns
        self.real_columns = interface.real_columns
        self.target_columns = interface.target_columns
        self.target_column_types = interface.target_column_types
        self.n_classes = interface.n_classes
        self.storage_layout = interface.storage_layout
        self.window_view = interface.window_view
        self.context_length = interface.window_view.context_length
        route = self.interfaces[self.active_interface_name]
        self.prediction_length = route.prediction_length
        self.decoding_support = route.decoding_support
        self.decoded_context_length = self.context_length - self.decoding_support + 1
        metadata = self.interface_runtime_metadata[self.active_interface_name]
        self.target_decoder_ids = metadata.target_decoder_ids
        self.target_n_classes = metadata.target_n_classes
        self.target_global_to_decoder = metadata.target_global_to_decoder
        self.softmax = {
            target: nn.LogSoftmax(dim=-1)
            for target, target_type in self.target_column_types.items()
            if target_type == "categorical"
        }
        self.loss_weights = dataset.loss_weights
        self.class_share_log_columns = dataset.class_share_log_columns
        self.index_maps = construct_index_maps(
            interface.id_maps, self.class_share_log_columns, True
        )
        if runtime is not None:
            self.criterion = runtime.criteria

    def evaluate_sources(
        self,
        source_configs: list[Any],
        dataset_runtimes: dict[str, Any],
        *,
        phase_index: int,
        phase_epoch: int,
    ) -> dict[str, float]:
        """Evaluate configured dataset/part sources using count-weighted losses."""

        from sequifier.training.runtime import build_source_runtime

        results = {}
        was_training = self.training
        accounting_dtype = (
            torch.float32 if torch.device(self.device).type == "mps" else torch.float64
        )
        baseline_metrics = getattr(self, "_baseline_metrics_by_source", None)
        if baseline_metrics is None:
            baseline_metrics = {}
            self._baseline_metrics_by_source = baseline_metrics
        self.eval()
        try:
            with torch.no_grad():
                for source_config in source_configs:
                    evaluation_started = time.perf_counter()
                    source = build_source_runtime(source_config, dataset_runtimes)
                    source.set_epoch(phase_epoch, "validation")
                    self.activate_dataset(
                        source_config.dataset,
                        dataset_runtimes[source_config.dataset],
                    )
                    loss_sums = {
                        target: torch.zeros(
                            (), dtype=accounting_dtype, device=self.device
                        )
                        for target in self.target_columns
                    }
                    class_counts = {
                        column: torch.zeros(
                            self.target_n_classes[column],
                            dtype=torch.int64,
                            device=self.device,
                        )
                        for column in self.class_share_log_columns
                    }
                    valid_count = torch.zeros((), dtype=torch.int64, device=self.device)
                    calculate_baseline = source_config.ref not in baseline_metrics
                    baseline_sums = {
                        target: torch.zeros(
                            (), dtype=accounting_dtype, device=self.device
                        )
                        for target in self.target_columns
                    }
                    baseline_count = torch.zeros(
                        (), dtype=torch.int64, device=self.device
                    )
                    for batch_index, runtime_batch in enumerate(
                        source.iter_batches("validation")
                    ):
                        batch = runtime_batch.batch
                        data = {
                            key: value.to(self.device, non_blocking=True)
                            for key, value in batch.inputs.items()
                            if key in self.input_columns
                        }
                        targets = {
                            key: value.to(self.device, non_blocking=True)
                            for key, value in batch.targets.items()
                            if key in self.target_column_types
                        }
                        metadata = {
                            key: value.to(self.device, non_blocking=True)
                            for key, value in batch.metadata.items()
                        }
                        data, targets, metadata = self.objective.prepare_batch(
                            data,
                            targets,
                            metadata,
                            eval_seed=(
                                self.hparams.seed
                                + phase_index * 1_000_003
                                + phase_epoch * 10_007
                                + batch_index
                            ),
                        )
                        output = self(data, metadata=metadata, return_logits=True)
                        valid_mask = self.objective.build_loss_mask(metadata)
                        if calculate_baseline:
                            pseudo_output = {
                                target: self._transform_val(
                                    target,
                                    self.objective.baseline_prediction_values(
                                        target,
                                        data,
                                        targets,
                                        self.target_column_types[target],
                                    ),
                                )
                                for target in self.target_columns
                            }
                            baseline_targets = {
                                target: self.objective.baseline_target_values(
                                    target, targets
                                )
                                for target in self.target_columns
                            }
                            batch_baseline_sums, batch_baseline_count = (
                                self._calculate_loss_components(
                                    pseudo_output,
                                    baseline_targets,
                                    valid_mask,
                                )
                            )
                            for target, value in batch_baseline_sums.items():
                                baseline_sums[target] += value.to(accounting_dtype)
                            baseline_count += batch_baseline_count
                        targets, valid_mask = self.objective.transform_targets_for_loss(
                            targets, valid_mask
                        )
                        local_sums, local_count = self._calculate_local_loss_components(
                            output, targets, valid_mask
                        )
                        for target, value in local_sums.items():
                            loss_sums[target] += value.detach().to(accounting_dtype)
                        valid_count += local_count.detach()
                        accumulate_class_counts(
                            class_counts,
                            output,
                            self._loss_valid_mask(valid_mask),
                            self.target_n_classes,
                        )
                    if self.hparams.global_training_spec.distributed:
                        dist.all_reduce(valid_count, op=dist.ReduceOp.SUM)
                        for value in loss_sums.values():
                            dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        for value in class_counts.values():
                            dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        if calculate_baseline:
                            dist.all_reduce(baseline_count, op=dist.ReduceOp.SUM)
                            for value in baseline_sums.values():
                                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    if valid_count.item() == 0:
                        raise RuntimeError(
                            f"Evaluation source {source_config.ref!r} has no valid targets"
                        )
                    denominator = valid_count.to(accounting_dtype)
                    target_losses = {
                        target: float((value / denominator).item())
                        for target, value in loss_sums.items()
                    }
                    results[source_config.ref] = float(
                        torch.stack(
                            tuple(
                                value * self._loss_weight(target) / denominator
                                for target, value in loss_sums.items()
                            )
                        )
                        .sum()
                        .item()
                    )
                    if calculate_baseline:
                        if baseline_count.item() == 0:
                            raise RuntimeError(
                                f"Evaluation source {source_config.ref!r} has no "
                                "valid baseline targets"
                            )
                        baseline_denominator = baseline_count.to(accounting_dtype)
                        baseline_target_losses = {
                            target: float((value / baseline_denominator).item())
                            for target, value in baseline_sums.items()
                        }
                        baseline_loss = float(
                            torch.stack(
                                tuple(
                                    value
                                    * self._loss_weight(target)
                                    / baseline_denominator
                                    for target, value in baseline_sums.items()
                                )
                            )
                            .sum()
                            .item()
                        )
                        baseline_metrics[source_config.ref] = (
                            baseline_loss,
                            baseline_target_losses,
                        )
                    baseline_loss, baseline_target_losses = baseline_metrics[
                        source_config.ref
                    ]
                    if self.rank == 0:
                        class_distributions = {}
                        for column, counts in class_counts.items():
                            total_count = int(counts.sum().item())
                            distribution = []
                            if total_count:
                                for decoder_id, count_tensor in enumerate(counts):
                                    count = int(count_tensor.item())
                                    if count == 0:
                                        continue
                                    global_id = self.target_decoder_ids[column][
                                        decoder_id
                                    ]
                                    distribution.append(
                                        {
                                            "class_id": global_id,
                                            "class_label": self.index_maps[column][
                                                global_id
                                            ],
                                            "count": count,
                                            "total_count": total_count,
                                            "share": count / total_count,
                                        }
                                    )
                            class_distributions[column] = distribution
                        writer = self.metric_writers_by_dataset[source_config.dataset]
                        writer.write_validation(
                            run_id=self.run_id,
                            session_id=self.session_id,
                            evaluation_kind=(
                                "dataset" if source_config.part is None else "part"
                            ),
                            epoch=phase_epoch,
                            batch=self._training_engine.state.global_batch_step,
                            batches_total=source.num_batches("validation"),
                            global_step=(self._training_engine.state.global_batch_step),
                            total_loss=results[source_config.ref],
                            target_losses=target_losses,
                            baseline_loss=baseline_loss,
                            baseline_target_losses=baseline_target_losses,
                            class_distributions=class_distributions,
                            learning_rate=self.optimizer.param_groups[0]["lr"],
                            elapsed_seconds=(time.perf_counter() - evaluation_started),
                            dataset=source_config.dataset,
                            part=source_config.part,
                        )
        finally:
            self.train(was_training)
        return results

    @property
    def encoder(self) -> ModuleDict:
        return getattr(self.ingestion, "encoder", ModuleDict())

    @property
    def pos_encoder(self):
        return getattr(self.ingestion, "pos_encoder", None)

    @property
    def layers(self) -> nn.ModuleList:
        return self.backbone.layers

    @property
    def real_columns_direct(self) -> list[str]:
        return getattr(self.ingestion, "real_columns_direct", [])

    def _ingestion_direct_real_dtype(self) -> torch.dtype:
        return self.backbone.layers[0].ff.get_first_layer_dtype()

    @beartype
    def initialize_optimizer(self, params: Any = None) -> None:
        """Create optimizer and scheduler from training config."""
        if params is None:
            params = self.parameters_to_optimize()

        opt_kwargs = dict(self.hparams.training_spec.optimizer)
        self.optimizer = self._get_optimizer(
            params=params, **self._filter_key(opt_kwargs, "name")
        )

        sched_kwargs = dict(self.hparams.training_spec.scheduler)
        self.scheduler = self._get_scheduler(**self._filter_key(sched_kwargs, "name"))
        self.scheduler_step_on = self.hparams.training_spec.scheduler_step_on

    def parameters_to_optimize(self):
        """Return the legacy iterator or the active policy's trainable parameters."""

        if not self._freezing_active:
            return self.parameters()
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(
                "The configured freezing policies leave the model with no "
                "trainable parameters."
            )
        return trainable_parameters

    @beartype
    def _apply_layer_dtypes(self) -> None:
        """Cast configured layer classes to requested dtypes."""
        layer_config = self.hparams.training_spec.layer_type_dtypes

        if not layer_config:
            return

        self.logger.info(f"Applying custom layer dtypes: {layer_config}")

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_decoder = name.startswith("decoder.") or ".decoder." in name
                if is_decoder and "decoder" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["decoder"]))
                elif "linear" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["linear"]))

            elif isinstance(module, nn.Embedding) and "embedding" in layer_config:
                target_dtype = get_torch_dtype(layer_config["embedding"])
                module.to(dtype=target_dtype)

            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                conv_dtype = (
                    layer_config.get("conv")
                    or layer_config.get("linear")
                    or layer_config.get("embedding")
                )
                if conv_dtype is not None:
                    module.to(dtype=get_torch_dtype(conv_dtype))

            elif isinstance(module, nn.MultiheadAttention):
                attention_dtype = layer_config.get("linear")
                if attention_dtype is not None:
                    module.to(dtype=get_torch_dtype(attention_dtype))

            elif isinstance(module, (nn.LayerNorm, RMSNorm)) and "norm" in layer_config:
                target_dtype = get_torch_dtype(layer_config["norm"])
                module.to(dtype=target_dtype)

        if "linear" in layer_config:
            target_dtype = get_torch_dtype(layer_config["linear"])
            for criterion in self.criterion.values():
                if hasattr(criterion, "weight") and criterion.weight is not None:
                    criterion.weight.data = criterion.weight.data.to(dtype=target_dtype)

    @beartype
    def _init_criterion(self, hparams: Any) -> dict[str, nn.Module]:
        """Build unreduced per-target loss modules."""
        criterion: dict[str, nn.Module] = {}
        for target_column in self.target_columns:
            criterion_name = hparams.training_spec.criterion[target_column]
            if hasattr(torch.nn, criterion_name):
                criterion_class = getattr(torch.nn, criterion_name)
            else:
                raise ValueError(f"Criterion {criterion_name} not found in torch.nn")

            criterion_kwargs = {}
            if (
                hparams.training_spec.class_weights is not None
                and target_column in hparams.training_spec.class_weights
            ):
                class_weights = Tensor(
                    hparams.training_spec.class_weights[target_column]
                )
                if self.target_column_types[target_column] == "categorical":
                    if class_weights.numel() == self.n_classes[target_column]:
                        class_weights = class_weights[
                            self.target_decoder_ids[target_column]
                        ]
                    elif class_weights.numel() != self.target_n_classes[target_column]:
                        raise ValueError(
                            f"class_weights[{target_column!r}] has incompatible length."
                        )
                criterion_kwargs["weight"] = class_weights

            criterion_kwargs["reduction"] = "none"

            criterion[target_column] = criterion_class(**criterion_kwargs)
        return criterion

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        """Return a causal attention mask."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    @staticmethod
    def _filter_key(dict_: dict[str, Any], key: str) -> dict[str, Any]:
        """Return a copy without key."""
        return {k: v for k, v in dict_.items() if k != key}

    @conditional_beartype
    def _build_attention_mask(self, valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        batch_size, context_length = valid_mask.shape
        device = valid_mask.device

        expected_context_length = self.src_mask.shape[-1]
        if context_length != expected_context_length:
            raise ValueError(
                f"valid_mask sequence length ({context_length}) must match "
                f"model sequence length ({expected_context_length})."
            )

        base_mask = self.src_mask.to(device=device, dtype=dtype)
        base_mask = base_mask.view(1, 1, context_length, context_length)

        invalid_keys = ~valid_mask.bool()

        padding_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_length,
            device=device,
            dtype=dtype,
        )

        padding_mask = padding_mask.masked_fill(
            invalid_keys[:, None, None, :],
            torch.finfo(dtype).min,
        )

        return base_mask + padding_mask

    @conditional_beartype
    def forward_inner(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        """Encode inputs into contextual hidden states."""
        valid_mask = metadata["attention_valid_mask"].bool()  # type: ignore
        ingestion_output = self.ingestion(src, metadata)
        ingestion_output = self.ingestion_adapter(
            cast_floating_to_module_dtype(ingestion_output, self.ingestion_adapter)
        )
        mask = self._build_attention_mask(valid_mask, dtype=ingestion_output.dtype)
        hidden = self.encode_ingested(ingestion_output, metadata, mask)
        return hidden.transpose(0, 1)

    @conditional_beartype
    def forward_embed(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        """Return configured final-step activations as one concatenated embedding."""
        valid_mask = metadata["attention_valid_mask"].bool()  # type: ignore
        ingestion_output = self.ingestion(src, metadata)
        ingestion_output = self.ingestion_adapter(
            cast_floating_to_module_dtype(ingestion_output, self.ingestion_adapter)
        )
        mask = self._build_attention_mask(valid_mask, dtype=ingestion_output.dtype)
        hidden, backbone_activations = self.backbone.forward_with_activations(
            ingestion_output.masked_fill(~valid_mask[:, :, None], 0.0),
            mask,
            self._embedding_backbone_layer_indices,
            self._embedding_capture_final_norm,
        )
        hidden = hidden.masked_fill(~valid_mask[:, :, None], 0.0)

        activations_by_name: dict[str, Tensor] = {}
        for layer_index in self._embedding_backbone_layer_indices:
            activation = backbone_activations[layer_index]
            activations_by_name[f"backbone.layers.{layer_index}"] = (
                activation.masked_fill(~valid_mask[:, :, None], 0.0).transpose(0, 1)
            )
        if self._embedding_capture_final_norm:
            activations_by_name["backbone.final_norm"] = hidden.transpose(0, 1)

        if self._embedding_decoder_block_indices:
            decoder_input = self._decoder_input_windows(hidden.transpose(0, 1))
            decoder_activations = self.decoder.hidden_block_activations(
                decoder_input,
                self._embedding_decoder_block_indices,
            )
            decoder_valid_mask = valid_mask[:, self.decoding_support - 1 :]
            for (branch_name, block_index), activation in decoder_activations.items():
                name = f"decoder.branches.{branch_name}.hidden_blocks.{block_index}"
                activations_by_name[name] = activation.masked_fill(
                    ~decoder_valid_mask.transpose(0, 1)[:, :, None], 0.0
                )

        selected_activations = [
            activations_by_name[name][-self.prediction_length :]
            for name in self.embedding_layer_names
        ]
        return torch.cat(selected_activations, dim=-1)

    @conditional_beartype
    def forward_train(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return raw decoded outputs for all target columns."""
        if self._composable:
            output = self.network(
                src, metadata, interface_name=self.active_interface_name
            )
        else:
            output = self.network(src, metadata)
        return {
            target_column: logits.transpose(0, 1)
            for target_column, logits in output.logits.items()
        }

    @conditional_beartype
    def _decoder_input_windows(self, output: Tensor) -> Tensor:
        """Return support-window decoder inputs from sequence-first states."""
        if output.shape[0] != self.context_length:
            raise ValueError(
                f"Decoder expected {self.context_length} hidden-state positions, "
                f"got {output.shape[0]}."
            )
        if self.decoding_support == 1:
            return output

        batch_first = output.transpose(0, 1)
        windows = batch_first.unfold(1, self.decoding_support, 1)
        windows = windows.permute(1, 0, 3, 2).contiguous()
        return windows.reshape(
            self.decoded_context_length,
            output.shape[1],
            self.decoding_support * self.dim_model,
        )

    @conditional_beartype
    def decode(self, target_column: str, output: Tensor) -> Tensor:
        """Project hidden states through one target decoder."""
        return self.decoder.decode(target_column, self._decoder_input_windows(output))

    @conditional_beartype
    def apply_softmax(self, target_column: str, output: Tensor) -> Tensor:
        """Apply LogSoftmax only for categorical targets."""
        if self.target_column_types[target_column] == "real":
            return output
        else:
            return self.softmax[target_column](output.float())

    @conditional_beartype
    def forward(
        self,
        src: dict[str, Tensor],
        metadata: dict[str, Tensor],
        return_logits: Union[bool, Tensor] = False,
    ) -> dict[str, Tensor]:
        """Return final-step logits or predictions for inference/eval."""
        output = self.forward_train(src, metadata)
        if return_logits:
            return output
        return {
            target_column: self.apply_softmax(
                target_column, out[-self.prediction_length :, :, :]
            )
            for target_column, out in output.items()
        }

    def _get_full_state_dict(
        self, ddp_model: Optional[nn.Module] = None
    ) -> dict[str, Tensor]:
        model_to_extract = ddp_model if ddp_model is not None else self
        if self.hparams.training_spec.data_parallelism == "FSDP":
            # FSDP2 uses StateDictOptions to gather the full state dict to rank 0 CPU
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(model_to_extract, options=options)

            # Only return on Rank 0 to save memory, as configured by StateDictOptions
            if self.rank == 0:
                return {
                    k.replace("_orig_mod.", ""): v.clone()
                    for k, v in state_dict.items()
                }
            return {}
        else:
            return {
                _canonical_parameter_name(k): v.cpu().clone()
                for k, v in self.state_dict().items()
            }

    @beartype
    def _check_and_terminate(self):
        """Exit 143 when rank 0 broadcasts an Optuna prune sentinel."""
        if os.getenv("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN") is not None:
            should_prune = 0
            if self.rank == 0:
                prune_file = os.path.join(
                    model_log_directory(self.project_root, self.model_name),
                    f"{self.model_name}.prune",
                )
                if os.path.exists(prune_file):
                    should_prune = 1

            if self.hparams.training_spec.distributed:
                signal_tensor = torch.tensor(
                    [should_prune], dtype=torch.int32, device=self.device
                )
                dist.broadcast(signal_tensor, src=0)
                should_prune = signal_tensor.item()

            if should_prune:
                if self.rank == 0:
                    self.logger.info(
                        "Pruning signal received from Optuna orchestrator. "
                        "Tearing down cooperatively."
                    )
                if self.hparams.training_spec.distributed:
                    cleanup()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

                sys.exit(143)

    @beartype
    def _checkpoint_compatibility_metadata(
        self, num_batches: Optional[int]
    ) -> dict[str, Any]:
        """Return resume-critical settings stored with each new checkpoint."""
        if self._composable:
            datasets = {}
            for name, dataset in self.hparams.dataset_training_spec.items():
                first_part = next(iter(dataset.parts.values()))
                metadata = first_part.metadata
                interface = dataset.interface
                datasets[name] = {
                    "model_interface": dataset.model_interface,
                    "parts": list(dataset.parts),
                    "schema": {
                        "storage_layout": asdict(interface.storage_layout),
                        "column_data_types": {
                            column: interface.column_data_types[column]
                            for column in interface.input_columns
                        },
                        "id_maps": {
                            column: interface.id_maps[column]
                            for column in interface.categorical_columns
                        },
                        "special_token_ids": interface.special_token_ids,
                        "normalize_real_columns": metadata.normalize_real_columns,
                        "normalization_statistics": (
                            {
                                column: metadata.selected_columns_statistics.get(
                                    column, {}
                                )
                                for column in interface.real_columns
                            }
                            if metadata.normalize_real_columns
                            else {}
                        ),
                    },
                    "criterion": dataset.criterion,
                    "class_weights": dataset.class_weights,
                    "loss_weights": dataset.loss_weights,
                    "freezing": dataset.freezing.model_dump(mode="json"),
                }
            settings = {
                "model": model_execution_config(self.hparams),
                "datasets": datasets,
                "training_plan": [
                    phase.model_dump(mode="json")
                    for phase in self.hparams.training_plan
                ],
            }
            encoded = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
            return {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "config_fingerprint": hashlib.sha256(encoded).hexdigest(),
                "resume_settings": settings,
                "provenance": {
                    name: {
                        part_name: {
                            "metadata_config_path": part.metadata_config_path,
                            "training_data_path": part.training_data_path,
                            "validation_data_path": part.validation_data_path,
                        }
                        for part_name, part in dataset.parts.items()
                    }
                    for name, dataset in self.hparams.dataset_training_spec.items()
                },
            }
        training_spec = self.hparams.training_spec
        bert_spec = (
            training_spec.bert_spec.model_dump(mode="json")
            if training_spec.bert_spec is not None
            else None
        )
        next_occurrence_config = (
            training_spec.next_occurrence_config.model_dump(mode="json")
            if training_spec.next_occurrence_config is not None
            else None
        )
        model_spec = self.hparams.model_spec.model_dump(mode="json")
        for component_name in ("ingestion", "backbone", "decoder"):
            component = model_spec[component_name]
            if component.get("freezing") is None:
                component.pop("freezing", None)
            if component.get("freezing_except") is None:
                component.pop("freezing_except", None)

        compatibility_settings = {
            "model_name": self.model_name,
            "read_format": self.hparams.read_format,
            "num_batches": num_batches,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "learning_rate": training_spec.learning_rate,
            "scheduler_step_on": self.scheduler_step_on,
            "scheduler": dict(training_spec.scheduler),
            "optimizer": dict(training_spec.optimizer),
            "semantic_optimizer_grouping": getattr(
                self, "_semantic_optimizer_grouping", False
            ),
            "distributed": training_spec.distributed,
            "data_parallelism": training_spec.data_parallelism,
            "world_size": (
                dist.get_world_size(group=self._data_parallel_process_group())
                if self._distributed_is_initialized()
                else training_spec.world_size
            ),
            "training_objective": self.hparams.training_objective,
            "seed": self.hparams.seed,
            "bert_spec": bert_spec,
            "next_occurrence_config": next_occurrence_config,
            "criterion": training_spec.criterion,
            "class_weights": training_spec.class_weights,
            "loss_weights": training_spec.loss_weights,
            "layer_type_dtypes": training_spec.layer_type_dtypes,
            "layer_autocast": training_spec.layer_autocast,
            "num_workers": training_spec.num_workers,
            "load_full_data_to_ram": training_spec.load_full_data_to_ram,
            "fsdp_cpu_offload": training_spec.fsdp_cpu_offload,
            "storage_layout": asdict(self.storage_layout),
            "window_view": asdict(self.window_view),
            "model_window_stride": self.hparams.model_window_stride,
            "column_data_types": self.hparams.column_data_types,
            "categorical_columns": self.categorical_columns,
            "real_columns": self.real_columns,
            "input_columns": self.input_columns,
            "target_columns": self.target_columns,
            "target_column_types": self.target_column_types,
            "categorical_decoder_special_tokens": getattr(
                self.hparams, "categorical_decoder_special_tokens", {}
            ),
            "categorical_target_codecs": self.target_decoder_ids,
            "n_classes": self.n_classes,
            "id_maps": self.hparams.id_maps,
            "special_token_ids": self.hparams.special_token_ids,
            "feature_layout": (
                self.hparams.feature_layout.model_dump(mode="json")
                if self.hparams.feature_layout is not None
                else None
            ),
            "model_spec": model_spec,
        }
        if self._freezing_active:
            compatibility_settings["trainable_parameter_names"] = [
                _canonical_parameter_name(name)
                for name, parameter in self.named_parameters(remove_duplicate=True)
                if parameter.requires_grad
            ]
        provenance = {
            "data_path": normalize_path(self.hparams.data_path, self.project_root),
            "validation_data_path": normalize_path(
                self.hparams.validation_data_path, self.project_root
            ),
            "metadata_config_path": normalize_path(
                self.hparams.metadata_config_path, self.project_root
            ),
        }
        fingerprint_input = json.dumps(
            compatibility_settings, sort_keys=True, default=str
        ).encode("utf-8")
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config_fingerprint": hashlib.sha256(fingerprint_input).hexdigest(),
            "resume_settings": compatibility_settings,
            "provenance": provenance,
        }

    @beartype
    def _validate_checkpoint_compatibility(
        self, checkpoint: dict[str, Any], num_batches: int
    ) -> None:
        """Reject checkpoints whose resume-critical settings no longer match."""
        checkpoint_metadata = checkpoint.get("checkpoint_metadata")
        if checkpoint_metadata is None:
            self.logger.warning(
                "Checkpoint has no compatibility metadata; "
                "continuing with legacy resume behavior."
            )
            return
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError("Checkpoint compatibility metadata must be a dictionary.")

        format_version = checkpoint_metadata.get("format_version")
        if format_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
            raise ValueError(
                "Unsupported checkpoint format version "
                f"{format_version!r}; supported versions are "
                f"{sorted(SUPPORTED_CHECKPOINT_FORMAT_VERSIONS)!r}."
            )

        saved_settings = checkpoint_metadata.get("resume_settings")
        if not isinstance(saved_settings, dict):
            raise ValueError(
                "Checkpoint compatibility metadata is missing resume_settings."
            )
        saved_settings = dict(saved_settings)
        if "trainable_parameter_names" in saved_settings:
            saved_settings["trainable_parameter_names"] = _canonical_parameter_names(
                saved_settings["trainable_parameter_names"]
            )

        current_metadata = self._checkpoint_compatibility_metadata(num_batches)
        current_settings = current_metadata["resume_settings"]
        saved_trainable_parameters = saved_settings.get("trainable_parameter_names")
        current_trainable_parameters = current_settings.get("trainable_parameter_names")
        if saved_trainable_parameters != current_trainable_parameters:
            raise ValueError(
                "Checkpoint trainable parameters do not match the current freezing "
                "configuration. Resume requires an identical trainable parameter "
                "set; use backbone initialization to start a new optimizer."
            )
        mismatches = []
        for key, current_value in current_settings.items():
            saved_value = saved_settings.get(key)
            if saved_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
                )

        if mismatches and self._composable:
            raise ValueError(
                "Checkpoint model/dataset topology or training plan does not match "
                "the current run. Use model initialization for a new run. "
                + "; ".join(mismatches)
            )
        if mismatches:
            mismatch_text = "; ".join(mismatches)
            warnings.warn(
                "Checkpoint is not identical with the current training configuration. "
                "Ensure that this is the intended configuration. "
                f"{mismatch_text}"
            )

        saved_fingerprint = checkpoint_metadata.get("config_fingerprint")
        current_fingerprint = current_metadata["config_fingerprint"]
        normalized_saved_fingerprint_input = json.dumps(
            saved_settings, sort_keys=True, default=str
        ).encode("utf-8")
        normalized_saved_fingerprint = hashlib.sha256(
            normalized_saved_fingerprint_input
        ).hexdigest()
        if (
            saved_fingerprint != current_fingerprint
            and normalized_saved_fingerprint != current_fingerprint
        ):
            warnings.warn(
                "Checkpoint configuration fingerprint mismatch: "
                f"checkpoint={saved_fingerprint!r}, current={current_fingerprint!r}"
            )

    @beartype
    def _get_rng_state(self) -> dict[str, Any]:
        """Capture Python, NumPy, Torch CPU, and CUDA RNG state for this rank."""
        device = torch.device(self.device)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device=device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None,
        }

    @beartype
    def _collect_rng_states_for_checkpoint(self) -> Optional[list[Any]]:
        """Gather per-rank RNG states on rank 0 for checkpointing."""
        rng_state = self._get_rng_state()
        if not self.hparams.training_spec.distributed:
            return [rng_state]

        rng_states = (
            [None] * dist.get_world_size(group=self._data_parallel_process_group())
            if self.rank == 0
            else None
        )
        dist.gather_object(
            rng_state,
            object_gather_list=rng_states,
            dst=0,
            group=self._data_parallel_process_group(),
        )
        return rng_states

    @beartype
    def _select_rng_state_for_rank(self, rng_states: Any) -> Optional[dict[str, Any]]:
        """Return this rank's saved RNG state from a checkpoint payload."""
        if rng_states is None:
            return None
        if isinstance(rng_states, dict):
            return rng_states
        if not isinstance(rng_states, list) or len(rng_states) == 0:
            return None

        rank = self.rank or 0
        if rank < len(rng_states):
            return rng_states[rank]
        self.logger.warning(
            "Checkpoint has no RNG state for this rank; "
            "using rank 0 RNG state as a fallback."
        )
        return rng_states[0]

    @beartype
    def _get_data_loader_generator_states(self) -> dict[str, Tensor]:
        """Capture dedicated DataLoader generator states."""
        return {
            name: generator.get_state()
            for name, generator in self._data_loader_generators.items()
        }

    @beartype
    def _restore_data_loader_generator_states(self) -> None:
        """Restore dedicated DataLoader generator states when present."""
        states = self._resume_data_loader_generator_states
        if states is None:
            return
        if not isinstance(states, dict):
            self.logger.warning(
                "Checkpoint DataLoader generator state is not a dictionary; "
                "using freshly seeded DataLoader generators."
            )
            return

        for name, generator in self._data_loader_generators.items():
            state = states.get(name)
            if isinstance(state, Tensor):
                generator.set_state(state)

    @beartype
    def _apply_checkpoint_training_state(
        self,
        scaler_state_dict: Optional[dict[str, Any]],
        best_val_loss: Any,
        n_epochs_no_improvement: Any,
        best_model_state_dict: Any,
        rng_states: Any,
        data_loader_generator_states: Any,
        run_id: Any = None,
    ) -> None:
        """Restore non-model training state from a checkpoint payload."""
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)
        elif self.scaler.is_enabled():
            self.logger.warning(
                "Checkpoint has no GradScaler state; "
                "resuming with a freshly initialized scaler."
            )

        self._resume_best_val_loss = float(best_val_loss)
        self._resume_n_epochs_no_improvement = int(n_epochs_no_improvement)
        self._resume_best_model_state_dict = best_model_state_dict
        self._resume_rng_state = self._select_rng_state_for_rank(rng_states)
        self._resume_data_loader_generator_states = data_loader_generator_states
        if isinstance(run_id, str) and run_id:
            self.run_id = run_id

    @beartype
    def _restore_rng_state(self) -> None:
        """Apply the checkpoint RNG state after compile/warm-up work is finished."""
        rng_state = self._resume_rng_state
        if rng_state is None:
            self.logger.warning(
                "Checkpoint has no RNG state; stochastic training will "
                "continue from the current process RNG state."
            )
            return

        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        cuda_state = rng_state.get("cuda")
        device = torch.device(self.device)
        if (
            cuda_state is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state(cuda_state, device=device)

    @beartype
    def train_model(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        ddp_model: Optional[nn.Module] = None,
    ) -> None:
        """Run epochs, validation, checkpointing, export, and interruption cleanup."""
        training_engine: TrainingEngine | None = getattr(self, "_training_engine", None)
        if self.rank == 0 and self.metric_writers is None:
            self.metric_writers = StructuredMetricWriters(
                self.project_root,
                self.model_name,
                self.rank,
                class_share_columns=self.class_share_log_columns,
            )
        self.logger.info(
            f"--- Starting Training for model: {self.model_name} | "
            f"run: {self.run_id} | session: {self.session_id} ---"
        )

        best_val_loss: float = float(self._resume_best_val_loss)
        n_epochs_no_improvement = self._resume_n_epochs_no_improvement
        last_epoch = self.start_epoch - 1
        best_model_state = self._resume_best_model_state_dict
        completion_reason = "normal_completion"

        try:
            self.last_latest_save_time = time.time()
            self.last_batch_save_time = time.time()
            self.last_batch_save_global_step = (self.start_epoch - 1) * len(
                train_loader
            ) + self.start_batch

            if (
                self.start_epoch == 1
                and self.hparams.training_spec.calculate_validation_loss_on_initialization
            ):
                total_loss, total_losses, class_counts = self._evaluate(
                    valid_loader, ddp_model
                )
                elapsed = 0.0

                self._log_epoch_results(
                    0,
                    0,
                    elapsed,
                    total_loss,
                    total_losses,
                    class_counts,
                    0,
                    len(train_loader),
                    "initial",
                )
            for epoch in range(self.start_epoch, self.hparams.training_spec.epochs + 1):
                if (
                    self.early_stopping_epochs is not None
                    and n_epochs_no_improvement >= self.early_stopping_epochs
                ):
                    completion_reason = "early_stopping"
                    break
                if epoch > self.start_epoch and np.isnan(total_loss):  # type: ignore # noqa: F821
                    raise RuntimeError("Validation loss became NaN.")

                epoch_start_time = time.time()
                train_loader.dataset.set_epoch(epoch)
                valid_loader.dataset.set_epoch(epoch)

                self._train_epoch(
                    train_loader,
                    valid_loader,
                    epoch,
                    ddp_model,
                    best_val_loss,
                    n_epochs_no_improvement,
                    best_model_state,
                )

                if training_engine is not None and training_engine.stop_requested:
                    completion_reason = "integration_requested_stop"
                    last_epoch = epoch
                    break

                total_loss, total_losses, class_counts = self._evaluate(
                    valid_loader, ddp_model
                )
                elapsed = time.time() - epoch_start_time
                self._log_epoch_results(
                    epoch,
                    len(train_loader),
                    elapsed,
                    total_loss,
                    total_losses,
                    class_counts,
                    epoch * len(train_loader),
                    len(train_loader),
                    "epoch_end",
                )

                if total_loss < best_val_loss:
                    best_val_loss = float(total_loss)
                    best_model_state = self._get_full_state_dict(ddp_model)
                    n_epochs_no_improvement = 0
                else:
                    n_epochs_no_improvement += 1

                if training_engine is not None:
                    training_engine.state.epoch = epoch
                    training_engine.state.best_validation_loss = best_val_loss
                    training_engine.state.epochs_without_improvement = (
                        n_epochs_no_improvement
                    )

                if self.scheduler_step_on == "epoch":
                    if training_engine is not None:
                        training_engine.step_scheduler()
                    elif (
                        not hasattr(self.scheduler, "total_steps")
                        or self.scheduler.last_epoch < self.scheduler.total_steps
                    ):
                        self.scheduler.step()

                if epoch % self.save_interval_epochs == 0:
                    self._save(
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
                self._check_and_terminate()
        except KeyboardInterrupt:
            completion_reason = "keyboard_interruption"
            self.logger.warning("Training interrupted; exporting final state.")
        except BaseException as error:
            if self.rank == 0:
                is_pruned = isinstance(error, SystemExit) and error.code == 143
                write_terminal_manifest(
                    self,
                    status="pruned" if is_pruned else "failed",
                    completion_reason="optuna_pruning" if is_pruned else "exception",
                    source_epoch=last_epoch,
                    exports_succeeded=False,
                    publication={"success": False, "reason": "not_attempted"},
                )
            raise

        if self.hparams.training_spec.distributed:
            dist.barrier()

        # Complete export needs one full final model. FSDP extraction is collective.
        last_model_state = self._get_full_state_dict(ddp_model)
        if best_model_state is None:
            if self.rank == 0:
                self.logger.info(
                    "No validation improvement... Saving last model as 'best'."
                )
            best_model_state = last_model_state

        finalization: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                exported_last_model = self._export(
                    last_model_state, "last", last_epoch, clean=True
                )
                self._export(best_model_state, "best", last_epoch, clean=True)
                if exported_last_model is None:
                    raise RuntimeError("Rank 0 did not construct an export model.")
            except Exception as error:
                finalization = {
                    "exports_succeeded": False,
                    "publication": {"success": False, "reason": "not_attempted"},
                    "error": f"{type(error).__name__}: {error}",
                }
                write_terminal_manifest(
                    self,
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
                finalization = {"exports_succeeded": True, "publication": publication}
                write_terminal_manifest(
                    self,
                    status="complete",
                    completion_reason=completion_reason,
                    source_epoch=last_epoch,
                    exports_succeeded=True,
                    publication=publication,
                )

        if self.hparams.training_spec.distributed:
            if self.rank is None:
                raise RuntimeError("Distributed training requires a process rank.")
            finalization = broadcast_publication_result(finalization, self.rank)
        if finalization is None or not finalization["exports_succeeded"]:
            error = None if finalization is None else finalization.get("error")
            raise RuntimeError(f"Complete-model export failed: {error}")

        publication = finalization["publication"]
        if publication.get("success"):
            self.logger.info(
                f"Published backbone revision {publication['revision_id']}."
            )
        elif publication.get("reason") == "compare_and_swap_conflict":
            self.logger.warning(
                "Backbone publication lost a compare-and-swap race; "
                "complete model exports remain valid."
            )
        elif publication.get("reason") == "publication_error":
            self.logger.warning(
                "Complete model exports succeeded, but backbone "
                f"publication failed: {publication.get('error')}"
            )
        self.logger.info("--- Training Complete ---")

        if training_engine is not None:
            training_engine.emit(
                RunCompleted(
                    access=training_engine.access,
                    completion_reason=completion_reason,
                )
            )

        if self.hparams.training_spec.distributed:
            dist.barrier()

    @beartype
    def _train_epoch(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epoch: int,
        ddp_model: Optional[nn.Module] = None,
        best_val_loss: float = float("inf"),
        n_epochs_no_improvement: int = 0,
        best_model_state: Optional[dict[str, Tensor]] = None,
    ) -> None:
        """Run one train epoch with optional mid-epoch saves."""
        target_names = self._loss_target_names()
        train_loss_sums, train_token_count = self._new_loss_accumulators(target_names)

        batches_aggregated = 0

        start_time = time.time()
        num_batches = len(train_loader)
        start_batch = self.start_batch
        self.start_batch = 0
        set_dataset_start_batch = getattr(train_loader.dataset, "set_start_batch", None)
        dataset_handles_start_batch = callable(set_dataset_start_batch)
        if dataset_handles_start_batch:
            set_dataset_start_batch(start_batch)

        model_to_call = ddp_model if ddp_model is not None else self

        model_to_call.train()
        training_engine: TrainingEngine | None = getattr(self, "_training_engine", None)

        for batch_offset, batch in enumerate(train_loader):
            if not isinstance(batch, SequifierBatch):
                raise TypeError(
                    "Training DataLoader must yield SequifierBatch objects, "
                    f"got {type(batch).__name__}."
                )
            batch_count = (
                start_batch + batch_offset
                if dataset_handles_start_batch
                else batch_offset
            )
            if batch_count >= start_batch:
                data = batch.inputs
                targets = batch.targets
                metadata = batch.metadata
                data = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in data.items()
                    if k in self.input_columns
                }
                targets = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in targets.items()
                    if k in self.target_column_types
                }
                metadata = {
                    k: v.to(self.device, non_blocking=True) for k, v in metadata.items()
                }
                data, targets, metadata = self.objective.prepare_batch(
                    data, targets, metadata
                )
                identity = None
                prepared_event = None
                trace_context = None
                if training_engine is not None:
                    identity = training_engine.identity(
                        epoch=epoch,
                        batch=batch_count + 1,
                        num_batches=num_batches,
                        accumulation_steps=self.accumulation_steps,
                    )
                    training_engine.update_batch_state(identity)
                    if training_engine.integrations.enabled:
                        prepared_event = BatchPrepared(
                            access=training_engine.access,
                            identity=identity,
                            inputs=data,
                            targets=targets,
                            metadata=metadata,
                        )
                        training_engine.emit(prepared_event)
                        trace_context = training_engine.integrations.forward_trace(
                            prepared_event
                        )

                # Only use standard torch.autocast if FSDP MixedPrecision is NOT handling it natively
                with activate_trace_context(trace_context):
                    if (
                        self.hparams.training_spec.layer_autocast
                        and self.hparams.training_spec.data_parallelism != "FSDP"
                    ):
                        amp_dtype = get_torch_dtype(
                            self.hparams.training_spec.layer_type_dtypes.get(
                                "linear", "bfloat16"
                            )
                            if self.hparams.training_spec.layer_type_dtypes
                            else "bfloat16"
                        )
                        with torch.autocast(
                            device_type=self.device.split(":")[0], dtype=amp_dtype
                        ):
                            output = model_to_call(
                                data, metadata=metadata, return_logits=True
                            )
                            (
                                loss,
                                backward_components,
                                local_loss_sums,
                                local_token_count,
                            ) = self._calculate_training_loss(output, targets, metadata)
                    else:
                        output = model_to_call(
                            data, metadata=metadata, return_logits=True
                        )
                        (
                            loss,
                            backward_components,
                            local_loss_sums,
                            local_token_count,
                        ) = self._calculate_training_loss(output, targets, metadata)

                if training_engine is not None and training_engine.integrations.enabled:
                    training_engine.emit(
                        ForwardCompleted(
                            access=training_engine.access,
                            identity=identity,
                            outputs={
                                target: logits.transpose(0, 1)
                                for target, logits in output.items()
                            },
                            captures=(
                                {} if trace_context is None else trace_context.captures
                            ),
                        )
                    )

                if self.accumulation_steps is None:
                    accumulation_divisor = 1
                else:
                    window_start = (
                        batch_count // self.accumulation_steps
                    ) * self.accumulation_steps
                    accumulation_divisor = min(
                        self.accumulation_steps,
                        num_batches - window_start,
                    )

                backward_loss = loss / accumulation_divisor
                optimizer_step_due = (
                    self.accumulation_steps is None
                    or (batch_count + 1) % self.accumulation_steps == 0
                    or (batch_count + 1) == num_batches
                )
                if training_engine is not None and training_engine.integrations.enabled:
                    training_engine.emit(
                        LossComputed(
                            access=training_engine.access,
                            identity=identity,
                            loss=loss,
                            backward_loss=backward_loss,
                        )
                    )
                if training_engine is not None:
                    if identity is None:
                        raise RuntimeError(
                            "Training engine did not create step identity."
                        )
                    optimizer_step_performed = training_engine.backward_and_step(
                        backward_loss=backward_loss,
                        identity=identity,
                        optimizer_step_due=optimizer_step_due,
                        gradient_clip_norm=self.hparams.training_spec.gradient_clip,
                    )
                else:
                    self.scaler.scale(backward_loss).backward()
                    optimizer_step_performed = False
                self._accumulate_loss_components(
                    train_loss_sums,
                    train_token_count,
                    local_loss_sums,
                    local_token_count,
                )

                if optimizer_step_due and training_engine is None:
                    self.scaler.unscale_(self.optimizer)
                    clip_norm = self.hparams.training_spec.gradient_clip
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    optimizer_step_performed = True
                    self.optimizer.zero_grad()

                if optimizer_step_due:
                    if not optimizer_step_performed:
                        self.optimizer.zero_grad()

                batches_aggregated += 1
                if (batch_count + 1) % self.log_interval == 0:
                    avg_train_loss, avg_train_losses = self._finalize_loss_components(
                        train_loss_sums,
                        train_token_count,
                        target_names,
                        "training",
                        raise_on_empty=False,
                    )
                    if self.rank == 0:
                        learning_rate = self.scheduler.get_last_lr()[0]
                        s_per_batch = (time.time() - start_time) / max(
                            1, batches_aggregated
                        )
                        global_step = (epoch - 1) * num_batches + batch_count + 1
                        if self.metric_writers is None:
                            raise RuntimeError(
                                "Rank 0 structured metric writers are not initialized."
                            )
                        self.metric_writers.write_training(
                            run_id=self.run_id,
                            session_id=self.session_id,
                            epoch=epoch,
                            batch=batch_count + 1,
                            batches_total=num_batches,
                            global_step=global_step,
                            window_batches=batches_aggregated,
                            total_loss=avg_train_loss,
                            target_losses=avg_train_losses,
                            learning_rate=learning_rate,
                            seconds_per_batch=s_per_batch,
                        )
                        self.logger.bind(log_channel="metric").info(
                            f"Epoch {epoch:3d} | Batch {(batch_count + 1):5d}/"
                            f"{num_batches:5d} | Loss: "
                            f"{format_number(avg_train_loss.detach().cpu().item())} | "
                            f"LR: {format_number(learning_rate)} | "
                            f"S/Batch {format_number(s_per_batch)}"
                        )

                    train_loss_sums, train_token_count = self._new_loss_accumulators(
                        target_names
                    )
                    if self.rank == 0:
                        batches_aggregated = 0
                        self.start_batch = 0
                        start_time = time.time()
                    self._check_and_terminate()

                del data, targets, output, loss, backward_loss, backward_components

                if self.scheduler_step_on == "batch" and optimizer_step_performed:
                    if training_engine is not None:
                        training_engine.step_scheduler()
                    elif (
                        not hasattr(self.scheduler, "total_steps")
                        or self.scheduler.last_epoch < self.scheduler.total_steps
                    ):
                        self.scheduler.step()

                if optimizer_step_due:
                    should_save_latest = torch.tensor(
                        [0], dtype=torch.int32, device=self.device
                    )
                    should_save_batch = torch.tensor(
                        [0], dtype=torch.int32, device=self.device
                    )
                    val_loss_batch = torch.tensor(
                        [np.float32(np.nan)], dtype=torch.float32, device=self.device
                    )

                    current_time = time.time()
                    elapsed_since_batch_save = current_time - self.last_batch_save_time
                    current_global_step = (epoch - 1) * num_batches + (batch_count + 1)
                    batches_since_batch_save = (
                        current_global_step - self.last_batch_save_global_step
                    )

                    if not self.hparams.training_spec.distributed or self.rank == 0:
                        if self.save_latest_interval_minutes is not None and (
                            current_time - self.last_latest_save_time
                        ) >= (self.save_latest_interval_minutes * 60):
                            should_save_latest[0] = 1

                        if self.save_interval_minutes is not None and (
                            elapsed_since_batch_save
                        ) >= (self.save_interval_minutes * 60):
                            should_save_batch[0] = 1

                        if (
                            self.save_interval_batches is not None
                            and batches_since_batch_save >= self.save_interval_batches
                        ):
                            should_save_batch[0] = 1

                    if self.hparams.training_spec.distributed:
                        dist.broadcast(should_save_latest, src=0)
                        dist.broadcast(should_save_batch, src=0)
                        dist.barrier()

                    if should_save_batch.item() == 1:
                        if self.save_interval_val_loss:
                            val_loss, val_losses, class_counts = self._evaluate(
                                valid_loader, ddp_model
                            )

                            if (
                                not self.hparams.training_spec.distributed
                                or self.rank == 0
                            ):
                                self._log_epoch_results(
                                    epoch,
                                    batch_count + 1,
                                    elapsed_since_batch_save,
                                    val_loss,
                                    val_losses,
                                    class_counts,
                                    current_global_step,
                                    num_batches,
                                    "interval",
                                )
                                val_loss_batch[0] = float(val_loss)
                            self._check_and_terminate()
                        else:
                            val_loss_batch.fill_(torch.nan)

                    if self.hparams.training_spec.distributed:
                        dist.broadcast(val_loss_batch, src=0)

                    if should_save_latest.item() == 1:
                        self._save(
                            epoch,
                            batch_count,
                            np.float32(np.nan),
                            ddp_model,
                            suffix="latest",
                            best_val_loss=best_val_loss,
                            n_epochs_no_improvement=n_epochs_no_improvement,
                            best_model_state_dict=best_model_state,
                            num_batches=num_batches,
                        )
                        self.last_latest_save_time = time.time()

                    val_loss = np.float32(val_loss_batch.item())
                    if should_save_batch.item() != 0:
                        self._save(
                            epoch,
                            batch_count,
                            val_loss,  # type: ignore
                            ddp_model,
                            suffix=f"epoch-{epoch}-batch-{batch_count + 1}",
                            best_val_loss=best_val_loss,
                            n_epochs_no_improvement=n_epochs_no_improvement,
                            best_model_state_dict=best_model_state,
                            num_batches=num_batches,
                        )
                        self.last_batch_save_time = time.time()
                        self.last_batch_save_global_step = current_global_step

                if training_engine is not None and training_engine.stop_requested:
                    break

        if dataset_handles_start_batch:
            set_dataset_start_batch(0)

    @beartype
    def _calculate_loss(
        self,
        output: dict[str, Tensor],
        targets: dict[str, Tensor],
        metadata: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return backward-scaled loss and components for the current rank."""
        loss, backward_components, _, _ = self._calculate_training_loss(
            output, targets, metadata
        )
        return loss, backward_components

    @beartype
    def _calculate_training_loss(
        self,
        output: dict[str, Tensor],
        targets: dict[str, Tensor],
        metadata: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Return the normalized backward loss plus local metric primitives."""
        target_names = self._loss_target_names(targets)
        if not target_names:
            raise RuntimeError("Loss calculation failed; no target columns were found.")

        valid_mask = self.objective.build_loss_mask(metadata)
        targets, valid_mask = self.objective.transform_targets_for_loss(
            targets, valid_mask
        )

        local_sums, local_count = self._calculate_local_loss_components(
            output, targets, valid_mask
        )
        global_count = local_count.detach().clone()
        gradient_average_factor = self._gradient_reduction_factor()

        if gradient_average_factor > 1:
            dist.all_reduce(
                global_count,
                op=dist.ReduceOp.SUM,
                group=self._data_parallel_process_group(),
            )

        loss = None
        backward_components = {}
        denominator = global_count.clamp_min(1)
        for target_column in target_names:
            denominator_for_sum = denominator.to(dtype=local_sums[target_column].dtype)
            backward_components[target_column] = (
                local_sums[target_column]
                * self._loss_weight(target_column)
                * gradient_average_factor
                / denominator_for_sum
            )
            if loss is None:
                loss = backward_components[target_column].clone()
            else:
                loss += backward_components[target_column]

        if loss is None:
            raise RuntimeError(
                "Loss calculation failed; no loss tensors were generated."
            )

        decoder = getattr(self, "decoder", None)
        regularization_loss = getattr(decoder, "regularization_loss", None)
        if callable(regularization_loss):
            loss = loss + regularization_loss()

        return loss, backward_components, local_sums, local_count

    @beartype
    def _calculate_local_loss_components(
        self,
        output: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Return unweighted, unnormalized local loss sums and one token count."""
        target_names = self._loss_target_names(targets)
        if not target_names:
            raise RuntimeError("Loss calculation failed; no target columns were found.")

        valid_mask = self._loss_valid_mask(valid_mask)
        mask = valid_mask.bool().T.contiguous().reshape(-1)
        token_count = mask.sum(dtype=torch.int64)

        loss_sums = {}
        for target_column in target_names:
            output_tensor = self._loss_output_tensor(target_column, output)
            target_tensor = self._loss_target_tensor(
                target_column,
                targets,
                sequence_length=valid_mask.shape[1],
                valid_mask=mask,
            )

            if self.target_column_types[target_column] == "real":
                target_tensor = target_tensor.to(dtype=output_tensor.dtype)

            output_count = (
                output_tensor.shape[0]
                if self.target_column_types[target_column] == "categorical"
                else output_tensor.numel()
            )
            if output_count != mask.numel():
                raise RuntimeError(
                    "Loss/mask size mismatch for target column "
                    f"{target_column!r}: output has {output_count} elements "
                    f"but mask has {mask.numel()}."
                )
            if target_tensor.numel() != mask.numel():
                raise RuntimeError(
                    "Target/mask size mismatch for target column "
                    f"{target_column!r}: target has {target_tensor.numel()} "
                    f"elements but mask has {mask.numel()}."
                )

            raw_loss = self.criterion[target_column](output_tensor, target_tensor)
            if raw_loss.numel() != mask.numel():
                raise RuntimeError(
                    "Loss/mask size mismatch for target column "
                    f"{target_column!r}: loss has {raw_loss.numel()} elements "
                    f"but mask has {mask.numel()}."
                )
            loss_sums[target_column] = raw_loss.masked_select(mask).sum()

        return loss_sums, token_count

    @beartype
    def _loss_valid_mask(self, valid_mask: Tensor) -> Tensor:
        """Return the suffix of target positions with full decoder support."""
        decoded_context_length = getattr(
            self,
            "decoded_context_length",
            valid_mask.shape[1],
        )
        return valid_mask[:, -decoded_context_length:]

    @beartype
    def _loss_output_tensor(
        self,
        target_column: str,
        output: dict[str, Tensor],
    ) -> Tensor:
        """Return flattened decoder outputs aligned to loss positions."""
        target_column_type = self.target_column_types[target_column]
        output_values = output[target_column]
        decoded_context_length = getattr(
            self,
            "decoded_context_length",
            output_values.shape[0],
        )
        if (
            (target_column_type == "real" and output_values.ndim >= 2)
            or (target_column_type == "categorical" and output_values.ndim == 3)
        ) and output_values.shape[0] > decoded_context_length:
            output_values = output_values[-decoded_context_length:]

        if target_column_type == "categorical":
            return output_values.float().reshape(
                -1, getattr(self, "target_n_classes", self.n_classes)[target_column]
            )
        if target_column_type == "real":
            return output_values.to(dtype=torch.float32).reshape(-1)
        raise ValueError(
            f"Target column type {target_column_type} not in ['categorical', 'real']"
        )

    @beartype
    def _loss_target_tensor(
        self,
        target_column: str,
        targets: dict[str, Tensor],
        sequence_length: int,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return flattened targets for the configured training objective."""
        target_values = self.objective.target_values_for_loss(target_column, targets)
        target_values = target_values[:, -sequence_length:]
        target_tensor = target_values.T.contiguous().reshape(-1)
        if self.target_column_types[target_column] == "categorical":
            target_tensor = _class_index_tensor(target_tensor)
            if not hasattr(self, "target_global_to_decoder"):
                return target_tensor
            lookup = torch.tensor(
                self.target_global_to_decoder[target_column],
                device=target_tensor.device,
            )
            target_tensor = lookup[target_tensor]
            excluded = target_tensor < 0
            checked = excluded if valid_mask is None else excluded & valid_mask
            if bool(checked.any()):
                raise ValueError(
                    f"Categorical target {target_column!r} contains excluded "
                    "special tokens at valid loss positions."
                )
            target_tensor = target_tensor.masked_fill(excluded, 0)
        return target_tensor

    @beartype
    def _calculate_loss_components(
        self,
        output: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Return detached local loss sums and one shared token count for metrics."""
        targets, valid_mask = self.objective.transform_targets_for_loss(
            targets, valid_mask
        )
        loss_sums, token_count = self._calculate_local_loss_components(
            output, targets, valid_mask
        )
        return (
            {
                col: loss_sum.detach().to(dtype=self._metric_float_dtype())
                for col, loss_sum in loss_sums.items()
            },
            token_count.detach(),
        )

    @beartype
    def _metric_float_dtype(self) -> torch.dtype:
        """Return the highest precision floating dtype supported by this device."""
        if torch.device(self.device).type == "mps":
            return torch.float32
        return torch.float64

    @beartype
    def _loss_target_names(
        self, targets: Optional[dict[str, Tensor]] = None
    ) -> list[str]:
        """Return configured target columns in stable training-config order."""
        configured_targets = getattr(
            self, "target_columns", list(self.target_column_types.keys())
        )
        if targets is not None:
            missing_targets = [
                col
                for col in configured_targets
                if col in self.target_column_types and col not in targets
            ]
            if missing_targets:
                raise RuntimeError(f"Missing target columns: {sorted(missing_targets)}")

        return [col for col in configured_targets if col in self.target_column_types]

    @beartype
    def _loss_weight(self, target_column: str) -> float:
        """Return the configured scalar loss weight for a target column."""
        if self.loss_weights is None:
            return 1.0
        return float(self.loss_weights.get(target_column, 1.0))

    @beartype
    def _distributed_is_initialized(self) -> bool:
        """Return whether torch.distributed collectives are currently usable."""
        return dist.is_available() and dist.is_initialized()

    @beartype
    def _data_parallel_process_group(self) -> Optional[dist.ProcessGroup]:
        """Return the process group used by the data-parallel reducer."""
        return getattr(self, "_data_parallel_group", None)

    @beartype
    def _gradient_reduction_factor(self) -> int:
        """Return the gradient multiplier needed before averaged reducers run."""
        if not self._distributed_is_initialized():
            return 1

        training_spec = getattr(getattr(self, "hparams", None), "training_spec", None)
        data_parallelism = getattr(training_spec, "data_parallelism", None)
        if data_parallelism in {"DDP", "FSDP"}:
            return dist.get_world_size(group=self._data_parallel_process_group())

        return 1

    @beartype
    def _new_loss_accumulators(
        self, target_names: list[str]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Create detached sum/count accumulators for logging or validation."""
        dtype = self._metric_float_dtype()
        return (
            {
                col: torch.zeros((), device=self.device, dtype=dtype)
                for col in target_names
            },
            torch.zeros((), device=self.device, dtype=dtype),
        )

    @beartype
    def _accumulate_loss_components(
        self,
        sums: dict[str, Tensor],
        count: Tensor,
        batch_sums: dict[str, Tensor],
        batch_count: Tensor,
    ) -> None:
        """Accumulate detached local unweighted loss sums and token counts."""
        for col in batch_sums:
            sums[col] = sums[col] + batch_sums[col].detach().to(
                device=sums[col].device,
                dtype=sums[col].dtype,
            )
        count += batch_count.detach().to(device=count.device, dtype=count.dtype)

    @beartype
    def _finalize_loss_components(
        self,
        sums: dict[str, Tensor],
        count: Tensor,
        target_names: list[str],
        label: str,
        raise_on_empty: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Reduce local loss sums/counts and return weighted token means."""
        packed = torch.stack([sums[col] for col in target_names] + [count])

        if self._distributed_is_initialized():
            dist.all_reduce(
                packed,
                op=dist.ReduceOp.SUM,
                group=self._data_parallel_process_group(),
            )

        n_targets = len(target_names)
        reduced_sums = dict(zip(target_names, packed[:n_targets]))
        reduced_count = packed[n_targets]

        if reduced_count.detach().cpu().item() == 0:
            if raise_on_empty:
                raise RuntimeError(f"No valid {label} tokens found.")

            dtype = self._metric_float_dtype()
            losses = {
                col: torch.zeros((), device=self.device, dtype=dtype)
                for col in target_names
            }
            return torch.zeros((), device=self.device, dtype=dtype), losses

        losses = {}
        total = torch.zeros((), device=self.device, dtype=self._metric_float_dtype())
        for col in target_names:
            losses[col] = reduced_sums[col] / reduced_count * self._loss_weight(col)
            total = total + losses[col]
        return total, losses

    @beartype
    def _copy_model(self):
        """Deep-copy without copying the logger handle."""
        logger_ref = self.logger
        del self.logger
        model_copy = copy.deepcopy(self)
        model_copy._initialize_log_file()
        self.logger = logger_ref
        return model_copy

    @beartype
    def _transform_val(self, col: str, val: Tensor) -> Tensor:
        """Transform targets into baseline-loss output shape."""
        if self.target_column_types[col] == "categorical":
            if hasattr(self.decoder, "target_dtype"):
                target_dtype = self.decoder.target_dtype(col)
            else:
                target_dtype = self.decoder[col].weight.dtype
            global_ids = _class_index_tensor(val)
            if not hasattr(self, "target_global_to_decoder"):
                return one_hot(global_ids, self.n_classes[col]).to(dtype=target_dtype)
            mapped = torch.tensor(
                self.target_global_to_decoder[col], device=global_ids.device
            )[global_ids]
            return one_hot(mapped.clamp_min(0), self.target_n_classes[col]).to(
                dtype=target_dtype
            ) * (mapped >= 0).unsqueeze(-1)
        else:
            if self.target_column_types[col] != "real":
                raise ValueError(f"Column {col} must be 'real' if not 'categorical'.")
            return val

    @beartype
    def _evaluate(
        self, valid_loader: DataLoader, ddp_model: Optional[nn.Module] = None
    ) -> tuple[np.float32, dict[str, np.float32], ClassCounts]:
        """Evaluate validation loss and optional class-share counts."""

        model_to_call = ddp_model if ddp_model is not None else self
        target_names = self._loss_target_names()
        class_count_columns = list(dict.fromkeys(self.class_share_log_columns))
        target_decoder_ids = getattr(self, "target_decoder_ids", {})
        target_n_classes = getattr(self, "target_n_classes", self.n_classes)

        for col in class_count_columns:
            missing_class_ids = [
                class_id
                for class_id in target_decoder_ids.get(col, range(self.n_classes[col]))
                if class_id not in self.index_maps[col]
            ]
            if missing_class_ids:
                raise ValueError(
                    f"Class-share column {col!r} is missing index-map entries "
                    f"for class IDs {missing_class_ids}."
                )

        local_class_counts: ClassCounts = {
            col: torch.zeros(
                target_n_classes[col],
                dtype=torch.int64,
                device=self.device,
            )
            for col in class_count_columns
        }

        was_training = model_to_call.training
        model_to_call.eval()

        try:
            total_loss_sums, total_loss_count = self._new_loss_accumulators(
                target_names
            )

            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_loader):
                    if not isinstance(batch, SequifierBatch):
                        raise TypeError(
                            "Validation DataLoader must yield SequifierBatch objects, "
                            f"got {type(batch).__name__}."
                        )
                    data = batch.inputs
                    targets = batch.targets
                    metadata = batch.metadata
                    # Move data to the current process's assigned GPU
                    data = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in data.items()
                        if k in self.input_columns
                    }
                    targets = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in targets.items()
                        if k in self.target_column_types
                    }
                    metadata = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in metadata.items()
                    }
                    data, targets, metadata = self.objective.prepare_batch(
                        data,
                        targets,
                        metadata,
                        eval_seed=self.hparams.seed + batch_idx,
                    )

                    valid_mask = self.objective.build_loss_mask(metadata)

                    if (
                        self.hparams.training_spec.layer_autocast
                        and self.hparams.training_spec.data_parallelism != "FSDP"
                    ):
                        amp_dtype = get_torch_dtype(
                            self.hparams.training_spec.layer_type_dtypes.get(
                                "linear", "bfloat16"
                            )
                            if self.hparams.training_spec.layer_type_dtypes
                            else "bfloat16"
                        )
                        with torch.autocast(
                            device_type=self.device.split(":")[0], dtype=amp_dtype
                        ):
                            output = model_to_call(
                                data, metadata=metadata, return_logits=True
                            )
                            loss_sums, token_counts = self._calculate_loss_components(
                                output, targets, valid_mask
                            )
                    else:
                        output = model_to_call(
                            data, metadata=metadata, return_logits=True
                        )
                        loss_sums, token_counts = self._calculate_loss_components(
                            output, targets, valid_mask
                        )

                    self._accumulate_loss_components(
                        total_loss_sums,
                        total_loss_count,
                        loss_sums,
                        token_counts,
                    )
                    accumulate_class_counts(
                        local_class_counts,
                        output,
                        self._loss_valid_mask(valid_mask),
                        target_n_classes,
                    )

            total_loss_global, total_losses_global = self._finalize_loss_components(
                total_loss_sums, total_loss_count, target_names, "validation"
            )

            if self._distributed_is_initialized():
                for col in class_count_columns:
                    dist.all_reduce(
                        local_class_counts[col],
                        op=dist.ReduceOp.SUM,
                        group=self._data_parallel_process_group(),
                    )

            # Handle one-time baseline loss calculation with the same aggregation semantics.
            if not hasattr(self, "baseline_loss"):
                baseline_loss_sums, baseline_loss_count = self._new_loss_accumulators(
                    target_names
                )

                with torch.no_grad():
                    for batch_idx, batch in enumerate(valid_loader):
                        if not isinstance(batch, SequifierBatch):
                            raise TypeError(
                                "Validation DataLoader must yield SequifierBatch objects, "
                                f"got {type(batch).__name__}."
                            )
                        data = batch.inputs
                        targets = batch.targets
                        metadata = batch.metadata
                        data = {
                            k: v.to(self.device, non_blocking=True)
                            for k, v in data.items()
                            if k in self.input_columns
                        }
                        targets = {
                            k: v.to(self.device, non_blocking=True)
                            for k, v in targets.items()
                            if k in self.target_column_types
                        }
                        metadata = {
                            k: v.to(self.device, non_blocking=True)
                            for k, v in metadata.items()
                        }

                        _, _, metadata = self.objective.prepare_batch(
                            data,
                            targets,
                            metadata,
                            eval_seed=self.hparams.seed + batch_idx,
                        )

                        valid_mask = self.objective.build_loss_mask(metadata)

                        pseudo_output = {}
                        targets_for_baseline = {}
                        for col in self.target_columns:
                            if col in targets:
                                pseudo_output[col] = self._transform_val(
                                    col,
                                    self.objective.baseline_prediction_values(
                                        col,
                                        data,
                                        targets,
                                        self.target_column_types[col],
                                    ),
                                )
                                targets_for_baseline[col] = (
                                    self.objective.baseline_target_values(col, targets)
                                )

                        if len(pseudo_output) > 0:
                            loss_sums, token_counts = self._calculate_loss_components(
                                pseudo_output,
                                targets_for_baseline,
                                valid_mask,
                            )
                            self._accumulate_loss_components(
                                baseline_loss_sums,
                                baseline_loss_count,
                                loss_sums,
                                token_counts,
                            )

                baseline_loss, baseline_losses = self._finalize_loss_components(
                    baseline_loss_sums,
                    baseline_loss_count,
                    target_names,
                    "baseline validation",
                )
                self.baseline_loss = baseline_loss.detach().cpu().item()
                self.baseline_losses = {
                    col: loss.detach().cpu().item()
                    for col, loss in baseline_losses.items()
                }

            return (
                np.float32(total_loss_global.detach().cpu().item()),
                {
                    k: np.float32(v.detach().cpu().item())
                    for k, v in total_losses_global.items()
                },
                {
                    col: counts.detach().cpu()
                    for col, counts in local_class_counts.items()
                },
            )
        finally:
            model_to_call.train(was_training)
            torch.clear_autocast_cache()

    @beartype
    def _export(
        self,
        state_dict: dict[str, Tensor],
        suffix: str,
        epoch: int,
        clean: bool = False,
    ) -> Optional["TransformerModel"]:
        """Export configured model variants from rank 0."""
        if self.rank != 0:
            return None

        # Instantiate a clean, decoupled CPU model for the export phase
        if clean:
            export_hparams = copy.deepcopy(self.hparams)
            export_hparams.training_spec.torch_compile = "none"
            export_hparams.training_spec.distributed = False
            export_hparams.training_spec.data_parallelism = None
            export_hparams.training_spec.fsdp_cpu_offload = None
            export_hparams.device = "cpu"
        else:
            export_hparams = self.hparams

        export_model = TransformerModel(export_hparams)
        export_model.load_state_dict(state_dict)
        export_model._backbone_parent_revision_id = self._backbone_parent_revision_id
        export_model.eval()

        os.makedirs(os.path.join(self.project_root, "models"), exist_ok=True)

        if self.export_generative_model:
            if self._composable:
                for index, dataset_name in enumerate(
                    export_hparams.dataset_training_spec
                ):
                    export_model.activate_dataset(dataset_name)
                    self._export_model(
                        export_model,
                        suffix,
                        epoch,
                        dataset_name=dataset_name,
                        write_pt=index == 0,
                    )
            else:
                self._export_model(export_model, suffix, epoch)
        if self.export_embedding_model:
            if self._composable:
                for dataset_name in export_hparams.dataset_training_spec:
                    export_model.activate_dataset(dataset_name)
                    model2 = TransformerEmbeddingModel(export_model)
                    self._export_model(
                        model2,
                        f"{suffix}-embedding",
                        epoch,
                        dataset_name=dataset_name,
                        write_pt=False,
                    )
            else:
                model2 = TransformerEmbeddingModel(export_model)
                self._export_model(model2, f"{suffix}-embedding", epoch)
        return export_model

    @beartype
    def _export_model(
        self,
        model: Union["TransformerModel", "TransformerEmbeddingModel"],
        suffix: str,
        epoch: int,
        dataset_name: str | None = None,
        write_pt: bool = True,
    ) -> None:
        """Write one model as ONNX and/or PT."""
        os.makedirs(os.path.join(self.project_root, "models"), exist_ok=True)

        if self.export_onnx:
            route_model = (
                model.transformer_model
                if isinstance(model, TransformerEmbeddingModel)
                else model
            )
            is_different_type = any(
                p.dtype in [torch.float16, torch.bfloat16, torch.float64]
                for p in model.parameters()
            )
            model_to_export = model

            if is_different_type:
                self.logger.info(
                    "Casting model to float32 for ONNX export compatibility..."
                )
                # Safe to deepcopy since `model` is already a pure CPU, unwrapped PyTorch module here.
                model_to_export = model._copy_model().float()

            export_device = next(model_to_export.parameters()).device

            x_cat = {
                col: torch.randint(
                    0,
                    route_model.n_classes[col],
                    (route_model.inference_batch_size, route_model.context_length),
                ).to(export_device, non_blocking=True)
                for col in route_model.categorical_columns
            }

            dtype_real = torch.float32 if is_different_type else None
            x_real = {
                col: torch.rand(
                    route_model.inference_batch_size, route_model.context_length
                ).to(export_device, non_blocking=True, dtype=dtype_real)
                for col in route_model.real_columns
            }

            input_dict = {**x_cat, **x_real}
            attention_valid_mask = torch.ones(
                route_model.inference_batch_size,
                route_model.context_length,
                dtype=torch.bool,
                device=export_device,
            )
            attention_valid_mask[0, 0] = False

            feature_columns = list(input_dict.keys())
            x = tuple(input_dict[col] for col in feature_columns) + (
                attention_valid_mask,
            )
            export_wrapper = _OnnxExportWrapper(model_to_export, feature_columns)

            input_names = [f"{col}_in" for col in input_dict.keys()] + [
                "attention_valid_mask"
            ]

            # Determine output names based on the model type
            if hasattr(model_to_export, "transformer_model"):
                output_names = ["output"]
            else:
                output_names = [
                    f"{col}_out" if col in input_names else col
                    for col in sorted(model_to_export.target_columns)
                ]

            # Export the model
            export_path = str(
                model_artifact_path(
                    self.project_root,
                    self.model_name,
                    suffix,
                    "onnx",
                    dataset_name=dataset_name,
                    dataset_count=(
                        len(self.hparams.dataset_training_spec)
                        if self._composable
                        else 1
                    ),
                )
            )

            try:
                torch._logging.set_logs(onnx=logging.ERROR)
                logging.getLogger("torch.onnx").setLevel(logging.ERROR)
            except (ImportError, AttributeError):
                torch.onnx.disable_log()  # Fallback for older PyTorch versions
            with (
                warnings.catch_warnings(),
                open(os.devnull, "w") as fnull,
                contextlib.redirect_stdout(fnull),
                contextlib.redirect_stderr(fnull),
            ):  # Ignore ONLY the specific messages we understand and expect
                warnings.filterwarnings(
                    "ignore",
                    message=".*Exporting a model while it is in training mode.*",
                )

                # Ignore the internal PyTree deprecation bubbling up from Python 3.14/copyreg
                warnings.filterwarnings("ignore", category=FutureWarning)

                torch.onnx.export(
                    export_wrapper,
                    x,
                    export_path,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    training=torch._C._onnx.TrainingMode.EVAL,
                )

            onnx_model = onnx.load(export_path)
            codec_metadata = onnx_model.metadata_props.add()
            codec_metadata.key = ONNX_CATEGORICAL_TARGET_CODECS_KEY
            codec_metadata.value = json.dumps(route_model.target_decoder_ids)
            if isinstance(model_to_export, TransformerEmbeddingModel):
                embedding_metadata = onnx_model.metadata_props.add()
                embedding_metadata.key = ONNX_EMBEDDING_LAYER_NAMES_KEY
                embedding_metadata.value = json.dumps(
                    model_to_export.transformer_model.embedding_layer_names
                )
            onnx.save(onnx_model, export_path)

        if self.export_pt and write_pt:
            export_path = str(
                model_artifact_path(
                    self.project_root,
                    self.model_name,
                    suffix,
                    "pt",
                )
            )
            torch.save(
                pt_bundle(model, self.hparams),
                export_path,
            )

    @beartype
    def _save(
        self,
        epoch: int,
        batch: int,
        val_loss: np.float32,
        ddp_model: Optional[nn.Module] = None,
        suffix: Optional[str] = None,
        best_val_loss: float = float("inf"),
        n_epochs_no_improvement: int = 0,
        best_model_state_dict: Optional[dict[str, Tensor]] = None,
        num_batches: Optional[int] = None,
    ) -> None:
        """Save rank-0 checkpoint state."""
        latest_path = checkpoint_path(self.hparams)
        output_path = (
            latest_path
            if suffix == "latest"
            else latest_path.with_name(f"{self.model_name}-{suffix}.pt")
        )
        training_engine: TrainingEngine | None = getattr(self, "_training_engine", None)
        if training_engine is not None:
            training_engine.emit(
                CheckpointSaving(
                    access=training_engine.access,
                    identity=training_engine.identity(
                        epoch=max(1, epoch),
                        batch=max(1, batch + 1),
                        num_batches=max(1, num_batches or 1),
                        accumulation_steps=self.accumulation_steps,
                    ),
                    path=output_path,
                )
            )
        model_to_extract = ddp_model if ddp_model is not None else self

        if self.hparams.training_spec.data_parallelism == "FSDP":
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)

            # Get model state dict
            raw_model_state = get_model_state_dict(model_to_extract, options=options)
            model_state_dict = {
                _canonical_parameter_name(k): v for k, v in raw_model_state.items()
            }

            # Get optimizer state dict
            optim_state_dict = get_optimizer_state_dict(
                model_to_extract, self.optimizer, options=options
            )

        else:
            model_state_dict = self.state_dict()
            model_state_dict = {
                _canonical_parameter_name(k): v for k, v in self.state_dict().items()
            }
            optim_state_dict = copy.deepcopy(self.optimizer.state_dict())

        rng_state = self._collect_rng_states_for_checkpoint()
        data_loader_generator_states = self._get_data_loader_generator_states()
        integration_state = (
            training_engine.integrations.checkpoint_state_dict()
            if training_engine is not None
            else {}
        )

        if self.rank != 0:
            return

        latest_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "checkpoint_metadata": self._checkpoint_compatibility_metadata(num_batches),
            "epoch": epoch,
            "batch": batch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optim_state_dict,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_state": rng_state,
            "data_loader_generator_states": data_loader_generator_states,
            "run_id": self.run_id,
            "best_val_loss": float(best_val_loss),
            "n_epochs_no_improvement": int(n_epochs_no_improvement),
            "best_model_state_dict": best_model_state_dict,
            "backbone_parent_revision_id": self._backbone_parent_revision_id,
            "loss": val_loss,
            "training_config": self.hparams.model_dump(mode="python"),
            "training_state": (
                asdict(training_engine.state) if training_engine is not None else None
            ),
            "integration_state": (integration_state),
        }

        temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, output_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise
        if output_path != latest_path:
            latest_temp_path = latest_path.with_name(
                f".{latest_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                torch.save(checkpoint, latest_temp_path)
                os.replace(latest_temp_path, latest_path)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(latest_temp_path)
        self.logger.info(f"Saved checkpoint to {output_path}")
        if training_engine is not None:
            training_engine.emit(
                CheckpointSaved(
                    access=training_engine.access,
                    identity=training_engine.identity(
                        epoch=max(1, epoch),
                        batch=max(1, batch + 1),
                        num_batches=max(1, num_batches or 1),
                        accumulation_steps=self.accumulation_steps,
                    ),
                    path=output_path,
                )
            )

    @beartype
    def _get_optimizer(self, params: Any, **kwargs):
        """Instantiate the configured optimizer."""
        optimizer_class = get_optimizer_class(self.hparams.training_spec.optimizer.name)
        return optimizer_class(
            params, lr=self.hparams.training_spec.learning_rate, **kwargs
        )

    @beartype
    def _get_scheduler(self, **kwargs):
        """Instantiate the configured LR scheduler."""
        scheduler_name = self.hparams.training_spec.scheduler.name
        if hasattr(torch.optim.lr_scheduler, scheduler_name):
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        else:
            raise ValueError(
                f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler"
            )
        return scheduler_class(self.optimizer, **kwargs)

    @beartype
    def _initialize_log_file(self):
        """Attach the configured logger."""
        self.logger = configure_logger(
            self.project_root,
            self.model_name,
            self.rank,
            dataset_names=self._log_dataset_names,
            rank_specific=self._rank_specific_logs,
        )

    @beartype
    def _get_latest_model_name(self) -> Optional[str]:
        """Return the explicitly configured run checkpoint when it exists."""
        path = checkpoint_path(self.hparams)
        return str(path) if path.is_file() else None

    @beartype
    def _log_epoch_results(
        self,
        epoch: int,
        batch: int,
        elapsed: float,
        total_loss: np.float32,
        total_losses: dict[str, np.float32],
        class_counts: ClassCounts,
        global_step: int,
        batches_total: int,
        evaluation_kind: str,
    ) -> None:
        """Write validation losses and class shares from rank 0."""
        training_engine: TrainingEngine | None = getattr(self, "_training_engine", None)
        if training_engine is not None:
            identity = training_engine.identity(
                epoch=max(1, epoch),
                batch=max(1, batch),
                num_batches=max(1, batches_total),
                accumulation_steps=self.accumulation_steps,
            )
            training_engine.emit(
                ValidationCompleted(
                    access=training_engine.access,
                    identity=identity,
                    total_loss=float(total_loss),
                    target_losses={
                        name: float(value) for name, value in total_losses.items()
                    },
                    evaluation_kind=evaluation_kind,
                )
            )
        if self.rank == 0:
            learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]

            class_distributions: dict[str, list[dict[str, Any]]] = {}
            class_share_summaries: list[str] = []
            for categorical_column in self.class_share_log_columns:
                counts = class_counts[categorical_column].to(torch.int64)
                total = counts.sum()
                total_count = int(total.item())

                if total_count == 0:
                    class_distributions[categorical_column] = []
                    self.logger.warning(
                        "No valid predictions available for class-share column "
                        f"{categorical_column!r}."
                    )
                    continue

                share_dtype = (
                    torch.float32 if counts.device.type == "mps" else torch.float64
                )
                shares = counts.to(share_dtype) / total
                distribution: list[dict[str, Any]] = []
                display_shares = []
                for class_id in range(counts.numel()):
                    count = int(counts[class_id].item())
                    if count == 0:
                        continue
                    global_class_id = self.target_decoder_ids[categorical_column][
                        class_id
                    ]
                    class_label = self.index_maps[categorical_column][global_class_id]
                    share = float(shares[class_id].item())
                    distribution.append(
                        {
                            "class_id": global_class_id,
                            "class_label": class_label,
                            "count": count,
                            "total_count": total_count,
                            "share": share,
                        }
                    )
                    display_shares.append(f"{class_label}: {share:5.5f}")
                class_distributions[categorical_column] = distribution
                class_share_summaries.append(
                    f"{categorical_column} (n={total_count}): "
                    + " | ".join(display_shares)
                )

            if self.metric_writers is None:
                raise RuntimeError(
                    "Rank 0 structured metric writers are not initialized."
                )
            self.metric_writers.write_validation(
                run_id=self.run_id,
                session_id=self.session_id,
                evaluation_kind=evaluation_kind,
                epoch=epoch,
                batch=batch,
                batches_total=batches_total,
                global_step=global_step,
                total_loss=total_loss,
                target_losses=total_losses,
                baseline_loss=self.baseline_loss,
                baseline_target_losses=self.baseline_losses,
                class_distributions=class_distributions,
                learning_rate=learning_rate,
                elapsed_seconds=elapsed,
            )

            metric_logger = self.logger.bind(log_channel="metric")
            metric_logger.info("-" * 89)
            metric_logger.info(
                f"Validation | Epoch: {epoch:3d} | Batch: {batch} | "
                f"Loss: {format_number(total_loss)} | "
                f"Baseline Loss: {format_number(self.baseline_loss)} | "
                f"Time: {elapsed:5.2f}s | LR {format_number(learning_rate)}"
            )

            if len(total_losses) > 1:
                loss_strs = [
                    f"{key}_loss: {format_number(value)}"
                    for key, value in total_losses.items()
                ]
                metric_logger.info(" - " + ", ".join(loss_strs))

            for summary in class_share_summaries:
                metric_logger.info(summary)

            metric_logger.info("-" * 89)


@beartype
def load_inference_model(
    model_type: str,
    model_path: str,
    training_config_path: Optional[str],
    args_config: dict[str, Any],
    device: str,
    infer_with_dropout: bool,
) -> torch.nn.Module:
    """Load a PT checkpoint as a generative or embedding inference module."""
    model_state = torch.load(
        model_path, map_location=torch.device(device), weights_only=False
    )
    if not isinstance(model_state, dict) or "model_state_dict" not in model_state:
        raise ValueError(f"Unsupported PyTorch model artifact: {model_path}")

    selected_dataset = args_config.get("dataset")
    selected_interface = args_config.get("model_interface")
    configured_training = None
    if training_config_path is not None:
        configured_training = load_train_config(
            training_config_path,
            {},
            False,
        )
        if selected_dataset is None:
            if len(configured_training.dataset_training_spec) == 1:
                selected_dataset = next(iter(configured_training.dataset_training_spec))
            elif selected_interface is None:
                raise ValueError(
                    "A dataset or model_interface selection is required for "
                    "multi-dataset configuration-driven PT inference"
                )
        if selected_dataset is not None:
            if selected_dataset not in configured_training.dataset_training_spec:
                raise ValueError(f"Unknown inference dataset {selected_dataset!r}")
            mapped_interface = configured_training.dataset_training_spec[
                selected_dataset
            ].model_interface
            if (
                selected_interface is not None
                and selected_interface != mapped_interface
            ):
                raise ValueError(
                    f"Dataset {selected_dataset!r} maps to interface "
                    f"{mapped_interface!r}, not {selected_interface!r}"
                )
            selected_interface = mapped_interface

    embedded_model_config = model_state.get("model_config")
    if embedded_model_config is not None:
        training_config, selected_interface = resolved_config_from_model_config(
            embedded_model_config,
            device=device,
            interface_name=selected_interface,
        )
    else:
        # Exact-resume checkpoints are not inference bundles, but accepting their
        # resolved config remains useful for local diagnosis.
        embedded_training = model_state.get("training_config")
        if embedded_training is not None:
            training_config = TrainModel.model_validate(embedded_training)
        elif configured_training is not None:
            training_config = configured_training
        else:
            raise ValueError(
                "PyTorch artifact has no model_config; provide "
                "training_config_path for a run checkpoint."
            )

    training_config.training_spec.torch_compile = "none"
    training_config.device = device

    with torch.no_grad():
        if model_type == "generative":
            model = TransformerModel(training_config)
        elif model_type == "embedding":
            model_inner = TransformerModel(training_config)
            model = TransformerEmbeddingModel(model_inner)
        else:
            raise ValueError(f"Unknown PT model type: {model_type!r}")

        route_model = (
            model.transformer_model
            if isinstance(model, TransformerEmbeddingModel)
            else model
        )
        if selected_dataset is not None and selected_dataset in getattr(
            training_config, "dataset_training_spec", {}
        ):
            route_model.activate_dataset(selected_dataset)
        elif selected_interface is not None and getattr(
            route_model, "_composable", False
        ):
            matching_dataset = next(
                (
                    name
                    for name, dataset in training_config.dataset_training_spec.items()
                    if dataset.model_interface == selected_interface
                ),
                None,
            )
            if matching_dataset is None:
                raise ValueError(
                    f"No execution route for model interface {selected_interface!r}"
                )
            route_model.activate_dataset(matching_dataset)

        model.logger.info(f"Loading model weights from {model_path}")
        canonical_state = {
            name.replace("_orig_mod.", ""): value
            for name, value in model_state["model_state_dict"].items()
        }
        if isinstance(model, TransformerEmbeddingModel) and not any(
            name.startswith("transformer_model.") for name in canonical_state
        ):
            canonical_state = {
                f"transformer_model.{name}": value
                for name, value in canonical_state.items()
            }
        model.load_state_dict(canonical_state)

        model.eval()

        if infer_with_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        model.to(device)

    return model


@beartype
def infer_with_embedding_model(
    model: nn.Module,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> np.ndarray:
    """Run batched embedding inference and concatenate CPU outputs."""
    outs0 = []

    categorical_cols = set(model.transformer_model.categorical_columns)

    with torch.no_grad():
        for batch_idx, x_sub in enumerate(x):
            layer_types = (
                model.transformer_model.hparams.training_spec.layer_type_dtypes or {}
            )
            dtype_str = layer_types.get("linear", "float32")
            ref_dtype = get_torch_dtype(dtype_str)
            data_gpu = {}
            for col, x_ in x_sub.items():
                if col in categorical_cols:
                    data_gpu[col] = torch.from_numpy(x_).to(device, dtype=torch.int64)
                else:
                    data_gpu[col] = torch.from_numpy(x_).to(
                        device, dtype=column_data_types.get(col, ref_dtype)
                    )
            metadata_gpu = (
                {
                    col: torch.from_numpy(x_).to(device)
                    for col, x_ in metadata[batch_idx].items()
                }
                if metadata
                else {}
            )

            output_gpu = model.forward(data_gpu, metadata=metadata_gpu)
            output_cpu = output_gpu.cpu().detach().float().numpy()
            output_cpu = output_cpu.transpose(1, 0, 2).reshape(
                output_cpu.shape[0] * output_cpu.shape[1], output_cpu.shape[2]
            )
            outs0.append(output_cpu)
            if device == "cuda":
                torch.cuda.empty_cache()

    outs = np.concatenate(outs0, axis=0)
    return outs


@beartype
def infer_with_generative_model(
    model: nn.Module,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> dict[str, np.ndarray]:
    """Run batched generative inference and trim CPU outputs."""
    outs0 = []

    categorical_cols = set(model.categorical_columns)

    with torch.no_grad():
        for batch_idx, x_sub in enumerate(x):
            layer_types = model.hparams.training_spec.layer_type_dtypes or {}
            dtype_str = layer_types.get("linear", "float32")
            ref_dtype = get_torch_dtype(dtype_str)
            data_gpu = {}
            for col, x_ in x_sub.items():
                if col in categorical_cols:
                    data_gpu[col] = torch.from_numpy(x_).to(device, dtype=torch.int64)
                else:
                    data_gpu[col] = torch.from_numpy(x_).to(
                        device, dtype=column_data_types.get(col, ref_dtype)
                    )
            metadata_gpu = (
                {
                    col: torch.from_numpy(x_).to(device)
                    for col, x_ in metadata[batch_idx].items()
                }
                if metadata
                else {}
            )

            output_gpu = model.forward(data_gpu, metadata=metadata_gpu)
            output_cpu = {k: v.cpu().detach() for k, v in output_gpu.items()}
            outs0.append(output_cpu)
            if device == "cuda":
                torch.cuda.empty_cache()

    outs = {
        target_column: np.concatenate(
            [
                o[target_column]
                .float()
                .numpy()
                .transpose(1, 0, 2)
                .reshape(
                    o[target_column].shape[0] * o[target_column].shape[1],
                    o[target_column].shape[2],
                )
                for o in outs0
            ],
            axis=0,
        )[:size, :]
        for target_column in target_columns
    }

    return outs
