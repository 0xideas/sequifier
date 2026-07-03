import contextlib
import copy
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import asdict

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
import time  # noqa: E402
import uuid  # noqa: E402
import warnings  # noqa: E402
from typing import Any, Optional, Union, cast  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from beartype import beartype  # noqa: E402
from packaging import version  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.amp.grad_scaler import GradScaler  # noqa: E402
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
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
CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = {2}
EMBEDDING_INDEX_DTYPES = (torch.int32, torch.int64)
NARROW_EMBEDDING_INDEX_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.uint16,
)
WIDE_UNSIGNED_EMBEDDING_INDEX_DTYPES = (torch.uint32, torch.uint64)

from sequifier.config.train_config import TrainModel, load_train_config  # noqa: E402
from sequifier.distributed.env import setup_distributed_env  # noqa: E402
from sequifier.helpers import (  # noqa: E402
    conditional_beartype,
    configure_determinism,
    configure_logger,
    construct_index_maps,
    get_torch_dtype,
    normalize_path,
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
from sequifier.model.frontends import (  # noqa: E402
    build_feature_frontend,
    get_feature_embedding_dims,
)
from sequifier.model.layers import RMSNorm, SequifierEncoderLayer  # noqa: E402
from sequifier.objectives import create_objective  # noqa: E402
from sequifier.optimizers.optimizers import get_optimizer_class  # noqa: E402


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
    config: TrainModel, local_rank: int
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


@beartype
def train_worker(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    from_folder: bool,
    global_rank: int,
    torch_compile: str,
):
    """Run one local distributed-training worker."""
    logger = configure_logger(config.project_root, config.model_name, global_rank)

    if config.training_spec.distributed:
        if config.training_spec.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        setup_distributed_env(
            global_rank, local_rank, world_size, config.training_spec.backend
        )

    if from_folder:
        if config.read_format == "pt":
            if config.training_spec.load_full_data_to_ram:
                train_dataset = SequifierDatasetFromFolderPt(
                    config.training_data_path, config
                )
                valid_dataset = SequifierDatasetFromFolderPt(
                    config.validation_data_path, config
                )
            else:
                train_dataset = SequifierDatasetFromFolderPtLazy(
                    config.training_data_path, config
                )
                valid_dataset = SequifierDatasetFromFolderPtLazy(
                    config.validation_data_path, config
                )
        elif config.read_format == "parquet":
            if config.training_spec.load_full_data_to_ram:
                train_dataset = SequifierDatasetFromFolderParquet(
                    config.training_data_path, config
                )
                valid_dataset = SequifierDatasetFromFolderParquet(
                    config.validation_data_path, config
                )
            else:
                train_dataset = SequifierDatasetFromFolderParquetLazy(
                    config.training_data_path, config
                )
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
        train_dataset = SequifierDatasetFromFile(config.training_data_path, config)
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
        pin_memory=config.training_spec.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
        generator=train_loader_generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        sampler=None,
        num_workers=config.training_spec.num_workers,
        pin_memory=config.training_spec.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
        generator=valid_loader_generator,
    )

    model = TransformerModel(config, rank=global_rank, local_rank=local_rank)
    model._data_loader_generators = {
        "train": train_loader_generator,
        "valid": valid_loader_generator,
    }
    base_model = model

    latest_model_path = model._get_latest_model_name()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    checkpoint = None

    # Initialize Optimizer
    if not config.training_spec.distributed:
        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        if config.training_spec.continue_training and latest_model_path:
            checkpoint = torch.load(
                latest_model_path, map_location="cpu", weights_only=False
            )
            model._validate_checkpoint_compatibility(checkpoint, len(train_loader))
            model.load_state_dict(checkpoint["model_state_dict"])
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
            )
        else:
            model.start_epoch = 1
            model.start_batch = 0
            logger.info(
                f"[INFO] Initializing new model with {format_number(pytorch_total_params)} parameters."
            )

        if config.training_spec.device.startswith("cuda"):
            if torch_compile == "outer":
                model = torch.compile(model)
            elif torch_compile == "inner":
                for i in range(len(model.layers)):
                    model.layers[i] = torch.compile(model.layers[i])

        if checkpoint is not None:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()

        model.train_model(train_loader, valid_loader, ddp_model=None)
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
        for layer in model.layers:
            fully_shard(layer, **fsdp_kwargs)

        fully_shard(model, **fsdp_kwargs)
        dist.barrier()

        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        resume_signal = [
            config.training_spec.continue_training and latest_model_path is not None
            if global_rank == 0
            else None
        ]
        dist.broadcast_object_list(resume_signal, src=0)
        did_resume = cast(bool, resume_signal[0])

        if did_resume:
            if global_rank == 0:
                if latest_model_path is None:
                    raise RuntimeError("Rank 0 selected resume without a checkpoint.")
                checkpoint = torch.load(
                    latest_model_path, map_location="cpu", weights_only=False
                )
                full_msd = checkpoint["model_state_dict"]
                full_osd = checkpoint["optimizer_state_dict"]
                start_epoch, start_batch = _checkpoint_start_position(
                    checkpoint, len(train_loader)
                )
                resume_state = {
                    "scaler_state_dict": checkpoint.get("scaler_state_dict"),
                    "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
                    "n_epochs_no_improvement": checkpoint.get(
                        "n_epochs_no_improvement", 0
                    ),
                    "has_best_model_state_dict": checkpoint.get("best_model_state_dict")
                    is not None,
                    "rng_state": checkpoint.get("rng_state"),
                    "data_loader_generator_states": checkpoint.get(
                        "data_loader_generator_states"
                    ),
                    "checkpoint_metadata": checkpoint.get("checkpoint_metadata"),
                }

                meta = [
                    start_epoch,
                    start_batch,
                    checkpoint["scheduler_state_dict"],
                    full_msd,
                    full_osd,
                    resume_state,
                ]
            else:
                meta = [None, None, None, None, None, None]

            # Broadcast the checkpoint data to all ranks simultaneously
            dist.broadcast_object_list(meta, src=0)

            # Unpack on all ranks. The placeholder Nones are replaced by broadcast.
            (
                start_epoch_obj,
                start_batch_obj,
                sched_state_obj,
                full_msd_obj,
                full_osd_obj,
                resume_state_obj,
            ) = meta
            model.start_epoch = cast(int, start_epoch_obj)
            model.start_batch = cast(int, start_batch_obj)
            sched_state = cast(Optional[dict[str, Any]], sched_state_obj)
            full_msd = cast(dict[str, Tensor], full_msd_obj)
            full_osd = cast(dict[str, Any], full_osd_obj)
            resume_state = cast(dict[str, Any], resume_state_obj)
            model._validate_checkpoint_compatibility(
                {"checkpoint_metadata": resume_state.get("checkpoint_metadata")},
                len(train_loader),
            )

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)

            set_model_state_dict(
                base_model,
                model_state_dict=full_msd,
                options=options,
            )

            set_optimizer_state_dict(
                base_model,
                base_model.optimizer,
                optim_state_dict=full_osd,
                options=options,
            )

            if sched_state is not None:
                base_model.scheduler.load_state_dict(sched_state)
            best_model_state_dict = None
            if resume_state.get("has_best_model_state_dict"):
                if global_rank == 0 and checkpoint is not None:
                    best_model_state_dict = checkpoint.get("best_model_state_dict")
                else:
                    best_model_state_dict = {}
            model._apply_checkpoint_training_state(
                resume_state.get("scaler_state_dict"),
                resume_state.get("best_val_loss", float("inf")),
                resume_state.get("n_epochs_no_improvement", 0),
                best_model_state_dict,
                resume_state.get("rng_state"),
                resume_state.get("data_loader_generator_states"),
            )

        else:
            model.start_epoch = 1
            model.start_batch = 0
            logger.info(
                f"[INFO] Initializing new model with {format_number(pytorch_total_params)} parameters."
            )

        if config.training_spec.device.startswith("cuda"):
            if torch_compile == "inner":
                for i in range(len(model.layers)):
                    model.layers[i] = torch.compile(model.layers[i])

        if config.training_spec.device.startswith("cuda"):
            dummy_data, dummy_metadata = create_dummy_data_and_metadata(
                config, local_rank
            )
            with torch.no_grad():
                _ = model(dummy_data, dummy_metadata, False)

            dist.barrier()

        if did_resume:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()

        model.train_model(train_loader, valid_loader, ddp_model=base_model)
        cleanup()
    elif config.training_spec.data_parallelism == "DDP":  # DDP
        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        if config.training_spec.continue_training and latest_model_path:
            checkpoint = torch.load(
                latest_model_path, map_location="cpu", weights_only=False
            )
            base_model._validate_checkpoint_compatibility(checkpoint, len(train_loader))
            base_model.load_state_dict(checkpoint["model_state_dict"])
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
            )
        else:
            model.start_epoch = 1
            model.start_batch = 0
            logger.info(
                f"[INFO] Initializing new model with {format_number(pytorch_total_params)} parameters."
            )

        if config.training_spec.device.startswith("cuda"):
            if torch_compile == "outer":
                model = torch.compile(model)

        device_ids = (
            [local_rank] if config.training_spec.device.startswith("cuda") else None
        )
        ddp_model = DDP(model, device_ids=device_ids, find_unused_parameters=False)

        if config.training_spec.device.startswith("cuda"):
            dummy_data, dummy_metadata = create_dummy_data_and_metadata(
                config, local_rank
            )

            if config.training_spec.layer_autocast:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    _ = ddp_model(dummy_data, dummy_metadata, False)
            else:
                with torch.no_grad():
                    _ = ddp_model(dummy_data, dummy_metadata, False)

            dist.barrier()
        if checkpoint is not None:
            base_model._restore_rng_state()
            base_model._restore_data_loader_generator_states()
        model.train_model(train_loader, valid_loader, ddp_model=ddp_model)
        cleanup()
    else:
        raise ValueError("For data_parallelism, only 'FSDP' and 'DDP' are supported")


@beartype
def _mp_train_worker_wrapper(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    from_folder: bool,
    torch_compile: str,
):
    train_worker(
        local_rank,
        world_size,
        config,
        from_folder,
        global_rank=local_rank,
        torch_compile=torch_compile,
    )


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    """Load train config and launch local or distributed training."""
    config_path = args.config_path or "configs/train.yaml"
    config = load_train_config(config_path, args_config, args.skip_metadata)

    torch.set_float32_matmul_precision(config.training_spec.float32_matmul_precision)

    world_size = config.training_spec.world_size
    from_folder = os.path.isdir(
        normalize_path(config.training_data_path, config.project_root)
    )

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
        train_worker(0, 1, config, from_folder, 0, config.training_spec.torch_compile)


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


class TransformerModel(nn.Module):
    """Sequifier transformer plus train/eval/export routines."""

    @beartype
    def __init__(
        self, hparams: Any, rank: Optional[int] = None, local_rank: Optional[int] = None
    ):
        """Build model modules and training state from config."""
        super().__init__()
        self.project_root = hparams.project_root
        self.model_type = "Transformer"

        self.rank = rank

        self.model_name = hparams.model_name or uuid.uuid4().hex[:8]

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
        self.export_generative_model = hparams.export_generative_model
        self.export_onnx = hparams.export_onnx
        self.export_pt = hparams.export_pt
        self.export_with_dropout = hparams.export_with_dropout
        self.early_stopping_epochs = hparams.training_spec.early_stopping_epochs
        self.hparams = hparams
        self.objective = create_objective(hparams)
        self.dim_model = self.hparams.model_spec.dim_model
        self.initial_embedding_dim = self.hparams.model_spec.initial_embedding_dim
        self.joint_embedding_dim = hparams.model_spec.joint_embedding_dim

        self.use_rope = hparams.model_spec.positional_encoding == "rope"
        if hparams.model_spec.feature_embedding_dims is not None:
            self.feature_embedding_dims = hparams.model_spec.feature_embedding_dims
        elif hparams.model_spec.ingestion_layer_spec.type == "direct_embed":
            self.feature_embedding_dims = get_feature_embedding_dims(
                self.initial_embedding_dim, self.categorical_columns, self.real_columns
            )
        else:
            self.feature_embedding_dims = {}

        self.frontend = build_feature_frontend(
            hparams=hparams,
            direct_real_dtype_provider=self._frontend_direct_real_dtype,
            device_max_concat_length=hparams.training_spec.device_max_concat_length,
        )

        self.layers = nn.ModuleList(
            [
                SequifierEncoderLayer(
                    hparams.model_spec,
                    self.dim_model,
                    hparams.model_spec.n_head,
                    hparams.model_spec.dim_feedforward,
                    hparams.training_spec.dropout,
                    hparams.window_view.context_length,
                )
                for _ in range(hparams.model_spec.num_layers)
            ]
        )

        if hparams.model_spec.norm_first:
            NormClass = (
                RMSNorm
                if hparams.model_spec.normalization == "rmsnorm"
                else nn.LayerNorm
            )
            self.final_norm = NormClass(self.dim_model)
        else:
            self.final_norm = nn.Identity()

        self.prediction_length = hparams.model_spec.prediction_length

        self.decoder = ModuleDict()
        self.softmax = ModuleDict()
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                self.decoder[target_column] = nn.Linear(
                    self.dim_model,
                    self.n_classes[target_column],
                )
                self.softmax[target_column] = nn.LogSoftmax(dim=-1)
            elif target_column_type == "real":
                self.decoder[target_column] = nn.Linear(self.dim_model, 1)
            else:
                raise ValueError(
                    f"Target column type {target_column_type} not in ['categorical', 'real']"
                )

        self.device = hparams.training_spec.device
        self.device_max_concat_length = hparams.training_spec.device_max_concat_length

        if hparams.training_spec.device.startswith("cuda"):
            if local_rank is not None:
                self.device = f"cuda:{local_rank}"
            elif self.rank is not None:  # Backwards compatibility
                self.device = f"cuda:{self.rank}"
            else:
                self.device = hparams.training_spec.device
        else:
            self.device = hparams.training_spec.device

        self.criterion = self._init_criterion(hparams=hparams)
        self.batch_size = hparams.training_spec.batch_size
        self.accumulation_steps = hparams.training_spec.accumulation_steps

        self.register_buffer(
            "src_mask",
            self.objective.build_attention_mask_policy(self.context_length),
            persistent=False,
        )

        self._init_weights()

        self.scheduler_step_on = hparams.training_spec.scheduler_step_on

        self.save_interval_epochs = hparams.training_spec.save_interval_epochs
        self.save_latest_interval_minutes = (
            hparams.training_spec.save_latest_interval_minutes
        )
        self.save_interval_minutes = hparams.training_spec.save_interval_minutes
        self.save_interval_batches = hparams.training_spec.save_interval_batches
        self.save_interval_val_loss = hparams.training_spec.save_interval_val_loss
        self.continue_training = hparams.training_spec.continue_training

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

        self._apply_layer_dtypes()

        self.to(self.device)

    @property
    def encoder(self) -> ModuleDict:
        return getattr(self.frontend, "encoder", ModuleDict())

    @property
    def pos_encoder(self):
        return getattr(self.frontend, "pos_encoder", None)

    @property
    def real_columns_direct(self) -> list[str]:
        return getattr(self.frontend, "real_columns_direct", [])

    @property
    def joint_embedding_layer(self):
        return getattr(self.frontend, "joint_embedding_layer", None)

    def _frontend_direct_real_dtype(self) -> torch.dtype:
        return self.layers[0].ff.get_first_layer_dtype()

    @beartype
    def initialize_optimizer(self, params: Any = None) -> None:
        """Create optimizer and scheduler from training config."""
        if params is None:
            params = self.parameters()

        opt_kwargs = dict(self.hparams.training_spec.optimizer)
        self.optimizer = self._get_optimizer(
            params=params, **self._filter_key(opt_kwargs, "name")
        )

        sched_kwargs = dict(self.hparams.training_spec.scheduler)
        self.scheduler = self._get_scheduler(**self._filter_key(sched_kwargs, "name"))
        self.scheduler_step_on = self.hparams.training_spec.scheduler_step_on

    @beartype
    def _apply_layer_dtypes(self) -> None:
        """Cast configured layer classes to requested dtypes."""
        layer_config = self.hparams.training_spec.layer_type_dtypes

        if not layer_config:
            return

        self.logger.info(f"[INFO] Applying custom layer dtypes: {layer_config}")

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_decoder = any(module is m for m in self.decoder.values())
                if is_decoder and "decoder" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["decoder"]))
                elif "linear" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["linear"]))

            elif isinstance(module, nn.Embedding) and "embedding" in layer_config:
                target_dtype = get_torch_dtype(layer_config["embedding"])
                module.to(dtype=target_dtype)

            elif isinstance(module, (nn.LayerNorm, RMSNorm)) and "norm" in layer_config:
                target_dtype = get_torch_dtype(layer_config["norm"])
                module.to(dtype=target_dtype)

        if "linear" in layer_config:
            target_dtype = get_torch_dtype(layer_config["linear"])
            for criterion in self.criterion.values():
                if hasattr(criterion, "weight") and criterion.weight is not None:
                    criterion.weight.data = criterion.weight.data.to(dtype=target_dtype)

    @beartype
    def _init_criterion(self, hparams: Any) -> ModuleDict:
        """Build unreduced per-target loss modules."""
        criterion = ModuleDict()
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
                criterion_kwargs["weight"] = Tensor(
                    hparams.training_spec.class_weights[target_column]
                )

            criterion_kwargs["reduction"] = "none"

            criterion[target_column] = criterion_class(**criterion_kwargs)
        return criterion

    @beartype
    def _get_feature_embedding_dims(
        self,
        embedding_size: int,
        categorical_columns: list[str],
        real_columns: list[str],
    ) -> dict[str, int]:
        """Allocate embedding dimensions across homogeneous input columns."""
        return get_feature_embedding_dims(
            embedding_size, categorical_columns, real_columns
        )

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        """Return a causal attention mask."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    @staticmethod
    def _filter_key(dict_: dict[str, Any], key: str) -> dict[str, Any]:
        """Return a copy without key."""
        return {k: v for k, v in dict_.items() if k != key}

    @beartype
    def _init_weights(self) -> None:
        """Initialize trainable weights with the model default."""
        init_std = 0.02
        self.frontend.initialize_weights()

        for target_column in self.target_columns:
            self.decoder[target_column].bias.data.zero_()
            self.decoder[target_column].weight.data.normal_(mean=0.0, std=init_std)

    @conditional_beartype
    def _recursive_concat(self, srcs: list[Tensor]):
        """Concatenate tensors in chunks to avoid device concat limits."""
        if len(srcs) <= self.device_max_concat_length:
            return torch.cat(srcs, 2)
        else:
            srcs_inner = []
            for start in range(0, len(srcs), self.device_max_concat_length):
                src = self._recursive_concat(
                    srcs[start : start + self.device_max_concat_length]
                )
                srcs_inner.append(src)
            return self._recursive_concat(srcs_inner)

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
    def _zero_padding_positions(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        """Zero padded query positions after attention/FFN layers."""
        return x * valid_mask[:, :, None].to(dtype=x.dtype)

    @conditional_beartype
    def forward_inner(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        """Encode inputs into contextual hidden states."""
        src2 = self.frontend(src, metadata)

        valid_mask = metadata["attention_valid_mask"].bool()  # type: ignore
        if valid_mask.shape != src2.shape[:2]:
            raise ValueError(
                f"Invalid attention mask shape: got {tuple(valid_mask.shape)}, "
                f"expected {tuple(src2.shape[:2])} = (batch_size, context_length). "
                "Check attention_valid_mask / leftPadLength construction."
            )
        src2 = self._zero_padding_positions(src2, valid_mask)

        mask = self._build_attention_mask(valid_mask, dtype=src2.dtype)

        for layer in self.layers:
            src2 = layer(src2, src_mask=mask)
            src2 = self._zero_padding_positions(src2, valid_mask)

        src2 = self.final_norm(src2)
        src2 = self._zero_padding_positions(src2, valid_mask)

        return src2.transpose(0, 1)

    @conditional_beartype
    def forward_embed(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> Tensor:
        """Return final-step embeddings."""
        return self.forward_inner(src, metadata)[-self.prediction_length :, :, :]

    @conditional_beartype
    def forward_train(
        self, src: dict[str, Tensor], metadata: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Return raw decoded outputs for all target columns."""
        output = self.forward_inner(src, metadata)
        output = {
            target_column: self.decode(target_column, output)
            for target_column in self.target_columns
        }

        return output

    @conditional_beartype
    def decode(self, target_column: str, output: Tensor) -> Tensor:
        """Project hidden states through one target decoder."""

        target_dtype = self.decoder[target_column].weight.dtype
        decoded = self.decoder[target_column](output.to(target_dtype)).to(torch.float32)

        return decoded

    @conditional_beartype
    def apply_softmax(self, target_column: str, output: Tensor) -> Tensor:
        """Apply LogSoftmax only for categorical targets."""
        if target_column in self.real_columns:
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
                k.replace("_orig_mod.", ""): v.cpu().clone()
                for k, v in self.state_dict().items()
            }

    @beartype
    def _check_and_terminate(self):
        """Exit 143 when rank 0 broadcasts an Optuna prune sentinel."""
        if os.getenv("SEQUIFIER_HYPERPARAMETER_SEARCH_RUN") is not None:
            should_prune = 0
            if self.rank == 0:
                prune_file = os.path.join(
                    self.project_root, "logs", f"sequifier-{self.model_name}.prune"
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
                        "[INFO] Pruning signal received from Optuna orchestrator. Tearing down cooperatively."
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
            "distributed": training_spec.distributed,
            "data_parallelism": training_spec.data_parallelism,
            "world_size": (
                dist.get_world_size(group=self._data_parallel_process_group())
                if self._distributed_is_initialized()
                else training_spec.world_size
            ),
            "training_objective": training_spec.training_objective,
            "seed": self.hparams.seed,
            "dropout": training_spec.dropout,
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
            "column_types": self.hparams.column_types,
            "categorical_columns": self.categorical_columns,
            "real_columns": self.real_columns,
            "input_columns": self.input_columns,
            "target_columns": self.target_columns,
            "target_column_types": self.target_column_types,
            "n_classes": self.n_classes,
            "id_maps": self.hparams.id_maps,
            "special_token_ids": self.hparams.special_token_ids,
            "feature_layout": (
                self.hparams.feature_layout.model_dump(mode="json")
                if self.hparams.feature_layout is not None
                else None
            ),
            "model_spec": self.hparams.model_spec.model_dump(mode="json"),
        }
        provenance = {
            "training_data_path": normalize_path(
                self.hparams.training_data_path, self.project_root
            ),
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
                "[WARNING] Checkpoint has no compatibility metadata; "
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

        current_metadata = self._checkpoint_compatibility_metadata(num_batches)
        current_settings = current_metadata["resume_settings"]
        mismatches = []
        for key, current_value in current_settings.items():
            saved_value = saved_settings.get(key)
            if saved_value != current_value:
                mismatches.append(
                    f"{key}: checkpoint={saved_value!r}, current={current_value!r}"
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
        if saved_fingerprint != current_fingerprint:
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
            "[WARNING] Checkpoint has no RNG state for this rank; "
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
                "[WARNING] Checkpoint DataLoader generator state is not a dictionary; "
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
    ) -> None:
        """Restore non-model training state from a checkpoint payload."""
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)
        elif self.scaler.is_enabled():
            self.logger.warning(
                "[WARNING] Checkpoint has no GradScaler state; "
                "resuming with a freshly initialized scaler."
            )

        self._resume_best_val_loss = float(best_val_loss)
        self._resume_n_epochs_no_improvement = int(n_epochs_no_improvement)
        self._resume_best_model_state_dict = best_model_state_dict
        self._resume_rng_state = self._select_rng_state_for_rank(rng_states)
        self._resume_data_loader_generator_states = data_loader_generator_states

    @beartype
    def _restore_rng_state(self) -> None:
        """Apply the checkpoint RNG state after compile/warm-up work is finished."""
        rng_state = self._resume_rng_state
        if rng_state is None:
            self.logger.warning(
                "[WARNING] Checkpoint has no RNG state; stochastic training will "
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
        self.logger.info(f"--- Starting Training for model: {self.model_name} ---")

        best_val_loss: float = float(self._resume_best_val_loss)
        n_epochs_no_improvement = self._resume_n_epochs_no_improvement
        last_epoch = self.start_epoch - 1
        best_model_state = self._resume_best_model_state_dict

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
                    0, 0, elapsed, total_loss, total_losses, class_counts, 0
                )
            for epoch in range(self.start_epoch, self.hparams.training_spec.epochs + 1):
                if (
                    self.early_stopping_epochs is None
                    or n_epochs_no_improvement < self.early_stopping_epochs
                ) and (
                    epoch == self.start_epoch
                    or epoch > self.start_epoch
                    and not np.isnan(total_loss)  # type: ignore # noqa: F821
                ):
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

                    total_loss, total_losses, class_counts = self._evaluate(
                        valid_loader, ddp_model
                    )
                    elapsed = time.time() - epoch_start_time

                    total_expected_batches = epoch * len(train_loader)
                    self._log_epoch_results(
                        epoch,
                        len(train_loader),
                        elapsed,
                        total_loss,
                        total_losses,
                        class_counts,
                        total_expected_batches,
                    )

                    if total_loss < best_val_loss:
                        best_val_loss = float(total_loss)
                        best_model_state = self._get_full_state_dict(ddp_model)
                        n_epochs_no_improvement = 0
                    else:
                        n_epochs_no_improvement += 1

                    if self.scheduler_step_on == "epoch":
                        if (
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
            self.logger.info("\n" + "=" * 89)
            self.logger.info("[WARNING] Training interrupted by user (Ctrl+C).")

            if self.hparams.training_spec.distributed:
                dist.barrier()

            answer_list = ["n"]

            if self.rank == 0:
                try:
                    answer = (
                        input(
                            "Do you want to export the 'best' and 'last' models? (y/n): "
                        )
                        .lower()
                        .strip()
                    )
                    if answer == "y":
                        answer_list[0] = "y"
                except EOFError:  # Handle non-interactive environments
                    answer_list[0] = "n"

            if self.hparams.training_spec.distributed:
                dist.broadcast_object_list(answer_list, src=0)

            if answer_list[0] == "y":
                if self.rank == 0:
                    self.logger.info("[INFO] User opted to export models.")

                if last_epoch is not None and best_model_state is not None:
                    if self.rank == 0:
                        self.logger.info(
                            f"[INFO] Exporting 'last' model from epoch {last_epoch}..."
                        )

                    # FSDP state extraction is collective; only rank 0 writes the result.
                    last_model_state = self._get_full_state_dict(ddp_model)

                    if self.rank == 0:
                        self._export(last_model_state, "last", last_epoch)

                        self.logger.info(
                            "[INFO] Exporting 'best' model (based on best val loss)..."
                        )
                        self._export(best_model_state, "best", last_epoch)
                        self.logger.info("[INFO] Models exported.")
                else:
                    if self.rank == 0:
                        self.logger.info(
                            "[INFO] Could not export model as no epoch ran."
                        )
            else:
                if self.rank == 0:
                    self.logger.info("[INFO] User opted *not* to export. Exiting.")

        if self.hparams.training_spec.distributed:
            dist.barrier()

        last_model_state = self._get_full_state_dict(ddp_model)

        if best_model_state is None:
            if self.rank == 0:
                self.logger.info(
                    "[INFO] No validation improvement... Saving last model as 'best'."
                )
            best_model_state = last_model_state

        if self.rank == 0:
            self._export(last_model_state, "last", last_epoch, clean=True)  # type: ignore
            self._export(best_model_state, "best", last_epoch, clean=True)  # type: ignore
            self.logger.info("--- Training Complete ---")

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

                # Only use standard torch.autocast if FSDP MixedPrecision is NOT handling it natively
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
                    output = model_to_call(data, metadata=metadata, return_logits=True)
                    (
                        loss,
                        backward_components,
                        local_loss_sums,
                        local_token_count,
                    ) = self._calculate_training_loss(output, targets, metadata)

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
                self.scaler.scale(backward_loss).backward()
                self._accumulate_loss_components(
                    train_loss_sums,
                    train_token_count,
                    local_loss_sums,
                    local_token_count,
                )

                optimizer_step_due = (
                    self.accumulation_steps is None
                    or (batch_count + 1) % self.accumulation_steps == 0
                    or (batch_count + 1) == num_batches
                )
                optimizer_step_performed = False

                if optimizer_step_due:
                    self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    optimizer_step_performed = True

                if optimizer_step_due:
                    if not optimizer_step_performed:
                        self.optimizer.zero_grad()

                batches_aggregated += 1
                if (batch_count + 1) % self.log_interval == 0:
                    avg_train_loss, _ = self._finalize_loss_components(
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
                        self.logger.info(
                            f"[INFO] Epoch {epoch:3d} | Batch {(batch_count+1):5d}/{num_batches:5d} | Loss: {format_number(avg_train_loss.detach().cpu().item())} | LR: {format_number(learning_rate)} | S/Batch {format_number(s_per_batch)}"
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
                    if (
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
                                    0,
                                    batch_count + 1,
                                    elapsed_since_batch_save,
                                    val_loss,
                                    val_losses,
                                    class_counts,
                                    current_global_step,
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

        mask = valid_mask.bool().T.contiguous().reshape(-1)
        token_count = mask.sum(dtype=torch.int64)

        loss_sums = {}
        for target_column in target_names:
            target_column_type = self.target_column_types[target_column]
            if target_column_type == "categorical":
                output_tensor = (
                    output[target_column]
                    .float()
                    .reshape(-1, self.n_classes[target_column])
                )
            elif target_column_type == "real":
                output_tensor = (
                    output[target_column].to(dtype=torch.float32).reshape(-1)
                )
            else:
                raise ValueError(
                    f"Target column type {target_column_type} not in ['categorical', 'real']"
                )

            target_tensor = self._loss_target_tensor(target_column, targets)

            if self.target_column_types[target_column] == "real":
                target_tensor = target_tensor.to(dtype=output_tensor.dtype)

            raw_loss = self.criterion[target_column](output_tensor, target_tensor)
            if raw_loss.numel() != mask.numel():
                raise RuntimeError(
                    "Loss/mask size mismatch for target column "
                    f"{target_column!r}: loss has {raw_loss.numel()} elements "
                    f"but mask has {mask.numel()}."
                )
            current_mask = mask.to(dtype=raw_loss.dtype)

            loss_sums[target_column] = (raw_loss * current_mask).sum()

        return loss_sums, token_count

    @beartype
    def _loss_target_tensor(
        self, target_column: str, targets: dict[str, Tensor]
    ) -> Tensor:
        """Return flattened targets for the configured training objective."""
        target_values = self.objective.target_values_for_loss(target_column, targets)
        target_tensor = target_values.T.contiguous().reshape(-1)
        if self.target_column_types[target_column] == "categorical":
            target_tensor = _class_index_tensor(target_tensor)
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
        return float(self.loss_weights[target_column])

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
            target_dtype = self.decoder[col].weight.dtype
            return (
                one_hot(_class_index_tensor(val), self.n_classes[col])
                .reshape(-1, self.n_classes[col])
                .to(dtype=target_dtype)
            )
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

        for col in class_count_columns:
            missing_class_ids = [
                class_id
                for class_id in range(self.n_classes[col])
                if class_id not in self.index_maps[col]
            ]
            if missing_class_ids:
                raise ValueError(
                    f"Class-share column {col!r} is missing index-map entries "
                    f"for class IDs {missing_class_ids}."
                )

        local_class_counts: ClassCounts = {
            col: torch.zeros(
                self.n_classes[col],
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
                        valid_mask,
                        self.n_classes,
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
    ) -> None:
        """Export configured model variants from rank 0."""
        if self.rank != 0:
            return

        # Instantiate a clean, decoupled CPU model for the export phase
        if clean:
            export_hparams = copy.deepcopy(self.hparams)
            export_hparams.training_spec.torch_compile = "none"
        else:
            export_hparams = self.hparams

        export_model = TransformerModel(export_hparams)
        export_model.load_state_dict(state_dict)
        export_model.eval()

        os.makedirs(os.path.join(self.project_root, "models"), exist_ok=True)

        if self.export_generative_model:
            self._export_model(export_model, suffix, epoch)
        if self.export_embedding_model:
            model2 = TransformerEmbeddingModel(export_model)
            self._export_model(model2, f"{suffix}-embedding", epoch)

    @beartype
    def _export_model(
        self,
        model: Union["TransformerModel", "TransformerEmbeddingModel"],
        suffix: str,
        epoch: int,
    ) -> None:
        """Write one model as ONNX and/or PT."""
        os.makedirs(os.path.join(self.project_root, "models"), exist_ok=True)

        if self.export_onnx:
            is_different_type = any(
                p.dtype in [torch.float16, torch.bfloat16, torch.float64]
                for p in model.parameters()
            )
            model_to_export = model

            if is_different_type:
                self.logger.info(
                    "[INFO] Casting model to float32 for ONNX export compatibility..."
                )
                # Safe to deepcopy since `model` is already a pure CPU, unwrapped PyTorch module here.
                model_to_export = model._copy_model().float()

            export_device = next(model_to_export.parameters()).device

            x_cat = {
                col: torch.randint(
                    0,
                    self.n_classes[col],
                    (self.inference_batch_size, self.context_length),
                ).to(export_device, non_blocking=True)
                for col in self.categorical_columns
            }

            dtype_real = torch.float32 if is_different_type else None
            x_real = {
                col: torch.rand(self.inference_batch_size, self.context_length).to(
                    export_device, non_blocking=True, dtype=dtype_real
                )
                for col in self.real_columns
            }

            input_dict = {**x_cat, **x_real}
            attention_valid_mask = torch.ones(
                self.inference_batch_size,
                self.context_length,
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
            export_path = os.path.join(
                self.project_root,
                "models",
                f"sequifier-{self.model_name}-{suffix}-{epoch}.onnx",
            )
            training_mode = (
                torch._C._onnx.TrainingMode.TRAINING
                if self.export_with_dropout
                else torch._C._onnx.TrainingMode.EVAL
            )
            constant_folding = self.export_with_dropout == False  # noqa: E712

            try:
                torch._logging.set_logs(onnx=logging.ERROR)
                logging.getLogger("torch.onnx").setLevel(logging.ERROR)
            except (ImportError, AttributeError):
                torch.onnx.disable_log()  # Fallback for older PyTorch versions
            with warnings.catch_warnings(), open(
                os.devnull, "w"
            ) as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(
                fnull
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
                    do_constant_folding=constant_folding,
                    input_names=input_names,
                    output_names=output_names,
                    training=training_mode,
                )

        if self.export_pt:
            export_path = os.path.join(
                self.project_root,
                "models",
                f"sequifier-{self.model_name}-{suffix}-{epoch}.pt",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "export_with_dropout": self.export_with_dropout,
                },
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
        model_to_extract = ddp_model if ddp_model is not None else self

        if self.hparams.training_spec.data_parallelism == "FSDP":
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)

            # Get model state dict
            raw_model_state = get_model_state_dict(model_to_extract, options=options)
            model_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in raw_model_state.items()
            }

            # Get optimizer state dict
            optim_state_dict = get_optimizer_state_dict(
                model_to_extract, self.optimizer, options=options
            )

        else:
            model_state_dict = self.state_dict()
            model_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in self.state_dict().items()
            }
            optim_state_dict = copy.deepcopy(self.optimizer.state_dict())

        rng_state = self._collect_rng_states_for_checkpoint()
        data_loader_generator_states = self._get_data_loader_generator_states()

        if self.rank != 0:
            return

        os.makedirs(os.path.join(self.project_root, "checkpoints"), exist_ok=True)

        file_name = f"{self.model_name}-{suffix}.pt"

        output_path = os.path.join(
            self.project_root,
            "checkpoints",
            file_name,
        )

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
            "best_val_loss": float(best_val_loss),
            "n_epochs_no_improvement": int(n_epochs_no_improvement),
            "best_model_state_dict": best_model_state_dict,
            "loss": val_loss,
        }

        temp_path = os.path.join(
            self.project_root,
            "checkpoints",
            f".{file_name}.{uuid.uuid4().hex}.tmp",
        )
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, output_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise
        self.logger.info(f"[INFO] Saved checkpoint to {output_path}")

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
        self.logger = configure_logger(self.project_root, self.model_name, self.rank)

    @beartype
    def _get_latest_model_name(self) -> Optional[str]:
        """Return the newest checkpoint path for this model name."""
        checkpoint_path = os.path.join(
            self.project_root, "checkpoints", f"{glob.escape(self.model_name)}-*.pt"
        )
        checkpoint_name_re = re.compile(
            rf"^{re.escape(self.model_name)}-(?:latest|epoch-\d+(?:-batch-\d+)?)\.pt$"
        )

        files = glob.glob(checkpoint_path)
        files = [
            file
            for file in files
            if checkpoint_name_re.fullmatch(os.path.split(file)[1])
        ]
        if files:
            return max(files, key=os.path.getmtime)
        else:
            return None

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
    ) -> None:
        """Log validation metrics and class shares from rank 0."""
        if self.rank == 0:
            learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]

            log_string = f"[INFO] Validation | Epoch: {epoch:3d} | Batch: {batch} | Loss: {format_number(total_loss)} | Baseline Loss: {format_number(self.baseline_loss)} | Time: {elapsed:5.2f}s | LR {format_number(learning_rate)}"

            self.logger.info("-" * 89)
            self.logger.info(log_string)

            metrics_file = os.path.join(
                self.project_root, "logs", f"sequifier-{self.model_name}-metrics.jsonl"
            )
            with open(metrics_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch,
                            "global_step": global_step,
                            "val_loss": float(total_loss),
                            "elapsed": elapsed,
                        }
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())

            if len(total_losses) > 1:
                loss_strs = [
                    f"{key}_loss: {format_number(value)}"
                    for key, value in total_losses.items()
                ]
                self.logger.info("[INFO]  - " + ", ".join(loss_strs))

            for categorical_column in self.class_share_log_columns:
                counts = class_counts[categorical_column].to(torch.int64)
                total = counts.sum()

                if total.item() == 0:
                    self.logger.warning(
                        "[WARNING] No valid predictions available for "
                        f"class-share column {categorical_column!r}."
                    )
                    continue

                share_dtype = (
                    torch.float32 if counts.device.type == "mps" else torch.float64
                )
                shares = counts.to(share_dtype) / total

                value_shares = " | ".join(
                    f"{self.index_maps[categorical_column][class_id]}: "
                    f"{shares[class_id].item():5.5f}"
                    for class_id in range(counts.numel())
                    if counts[class_id].item() > 0
                )

                self.logger.info(
                    f"[INFO] {categorical_column} (n={total.item()}): {value_shares}"
                )

            self.logger.info("-" * 89)


@beartype
def load_inference_model(
    model_type: str,
    model_path: str,
    training_config_path: str,
    args_config: dict[str, Any],
    device: str,
    infer_with_dropout: bool,
) -> torch.nn.Module:
    """Load a PT checkpoint as a generative or embedding inference module."""
    skip_metadata = args_config.get("skip_metadata", False)
    args_config_subset = {
        k: v for k, v in args_config.items() if k not in ["model_path", "data_path"]
    }
    training_config = load_train_config(
        training_config_path, args_config_subset, skip_metadata
    )

    training_config.training_spec.torch_compile = "none"

    with torch.no_grad():
        model = TransformerModel(training_config)
        if model_type == "generative":
            model = TransformerModel(training_config)
        elif model_type == "embedding":
            model_inner = TransformerModel(training_config)
            model = TransformerEmbeddingModel(model_inner)
        else:
            assert False, "impossible"

        model.logger.info(f"[INFO] Loading model weights from {model_path}")
        model_state = torch.load(
            model_path, map_location=torch.device(device), weights_only=False
        )
        model.load_state_dict(model_state["model_state_dict"])

        model.eval()

        if infer_with_dropout:
            if not model_state["export_with_dropout"]:
                warnings.warn(
                    "Model was exported with 'export_with_dropout'==False. By setting 'infer_with_dropout' to True, you are overriding this configuration"
                )
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        if not device.startswith("mps"):
            model = torch.compile(model).to(device)
        else:
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
    column_types: dict[str, torch.dtype],
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
                        device, dtype=column_types.get(col, ref_dtype)
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
    column_types: dict[str, torch.dtype],
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
                        device, dtype=column_types.get(col, ref_dtype)
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
