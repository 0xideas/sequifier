import copy
import glob
import math
import os
import time
import uuid
import warnings
from datetime import timedelta
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from beartype import beartype
from packaging import version
from torch import Tensor, nn
from torch.amp import GradScaler
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
else:
    from torch.distributed._composable.fsdp import (
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )

from torch.distributed.device_mesh import init_device_mesh
from torch.nn import ModuleDict
from torch.nn.functional import one_hot
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

torch._dynamo.config.suppress_errors = True

from sequifier.config.train_config import TrainModel, load_train_config  # noqa: E402
from sequifier.helpers import (  # noqa: E402
    conditional_beartype,
    configure_determinism,
    configure_logger,
    construct_index_maps,
    get_torch_dtype,
)
from sequifier.io.sequifier_dataset_from_file import (  # noqa: E402
    SequifierDatasetFromFile,
)
from sequifier.io.sequifier_dataset_from_folder import (  # noqa: E402
    SequifierDatasetFromFolder,
)
from sequifier.io.sequifier_dataset_from_folder_lazy import (  # noqa: E402
    SequifierDatasetFromFolderLazy,
)
from sequifier.model.layers import RMSNorm, SequifierEncoderLayer  # noqa: E402
from sequifier.optimizers.optimizers import get_optimizer_class  # noqa: E402


@beartype
def setup(rank: int, local_rank: int, world_size: int, backend: str = "nccl"):
    """Sets up the distributed training environment.

    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        backend: The distributed backend to use.
    """
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if not dist.is_initialized():
        timeout_sec = int(os.environ.get("NCCL_TIMEOUT", 1800))

        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )

        dist.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
            device_id=device_id,
        )


def cleanup():
    """Cleans up the distributed training environment."""
    dist.destroy_process_group()


@beartype
def create_dummy_data(config: TrainModel, local_rank: int) -> dict[str, Tensor]:
    dummy_data = {}
    for col in config.input_columns:
        dtype = torch.int64 if col in config.categorical_columns else torch.float32
        dummy_data[col] = torch.ones(
            (config.training_spec.batch_size, config.seq_length),
            dtype=dtype,
            device=local_rank,
        )

    return dummy_data


@beartype
def train_worker(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    from_folder: bool,
    global_rank: int,
    torch_compile: str,
):
    """The worker function for distributed training.

    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        config: The training configuration.
        from_folder: Whether to load data from a folder (e.g., preprocessed .pt files)
                     or a single file (e.g., .parquet).
        global_rank: The global rank
    """
    logger = configure_logger(config.project_root, config.model_name, global_rank)

    if config.training_spec.distributed:
        if config.training_spec.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        setup(global_rank, local_rank, world_size, config.training_spec.backend)

    # 1. Create Datasets and DataLoaders with DistributedSampler
    if from_folder:
        if config.training_spec.load_full_data_to_ram:
            train_dataset = SequifierDatasetFromFolder(
                config.training_data_path, config
            )
            valid_dataset = SequifierDatasetFromFolder(
                config.validation_data_path, config
            )
        else:
            train_dataset = SequifierDatasetFromFolderLazy(
                config.training_data_path, config
            )
            valid_dataset = SequifierDatasetFromFolderLazy(
                config.validation_data_path, config
            )
    else:
        if config.training_spec.distributed:
            raise ValueError(
                "Distributed training is not supported with single-file datasets."
            )
        train_dataset = SequifierDatasetFromFile(config.training_data_path, config)
        valid_dataset = SequifierDatasetFromFile(config.validation_data_path, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Batching is handled natively by the IterableDataset
        sampler=None,  # Sharding is handled natively by the IterableDataset
        num_workers=config.training_spec.num_workers,
        pin_memory=config.training_spec.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        sampler=None,
        num_workers=config.training_spec.num_workers,
        pin_memory=config.training_spec.device not in ["mps", "cpu"],
        prefetch_factor=4 if config.training_spec.num_workers > 0 else None,
        persistent_workers=(config.training_spec.num_workers > 0),
    )

    configure_determinism(config.seed, config.training_spec.enforce_determinism)

    model = TransformerModel(config, rank=global_rank, local_rank=local_rank)
    base_model = model

    latest_model_path = model._get_latest_model_name()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    checkpoint = None

    is_fsdp = config.training_spec.fsdp

    # Initialize Optimizer
    if not config.training_spec.distributed:
        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        if config.training_spec.continue_training and latest_model_path:
            checkpoint = torch.load(
                latest_model_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint["batch"] + 1 >= len(train_loader):
                base_model.start_epoch = checkpoint["epoch"] + 1
                base_model.start_batch = 0
            else:
                base_model.start_epoch = checkpoint["epoch"]
                base_model.start_batch = checkpoint["batch"] + 1
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

        model.train_model(train_loader, valid_loader, ddp_model=None)
    elif is_fsdp:
        mesh = init_device_mesh(
            "cuda", (world_size,)
        )  # 1D mesh for standard ZeRO-3 full sharding

        mp_policy = None
        if config.training_spec.layer_autocast:
            amp_dtype = get_torch_dtype(
                config.training_spec.layer_type_dtypes.get("linear", "bfloat16")
                if config.training_spec.layer_type_dtypes
                else "float32"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=amp_dtype,
                reduce_dtype=amp_dtype,
                output_dtype=amp_dtype,
            )

        offload_policy = (
            OffloadPolicy() if config.training_spec.fsdp_cpu_offload else None
        )
        for layer in model.layers:
            fully_shard(
                layer, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy
            )
        fully_shard(
            model, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy
        )
        dist.barrier()

        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        if config.training_spec.continue_training and latest_model_path:
            if global_rank == 0:
                checkpoint = torch.load(
                    latest_model_path, map_location="cpu", weights_only=False
                )
                full_msd = checkpoint["model_state_dict"]
                full_osd = checkpoint["optimizer_state_dict"]

                if checkpoint["batch"] + 1 >= len(train_loader):
                    start_epoch = checkpoint["epoch"] + 1
                    start_batch = 0
                else:
                    start_epoch = checkpoint["epoch"]
                    start_batch = checkpoint["batch"] + 1

                meta = [
                    start_epoch,
                    start_batch,
                    checkpoint["scheduler_state_dict"],
                    full_msd,
                    full_osd,
                ]
            else:
                meta = [None, None, None, None, None]

            # Broadcast the checkpoint data to all ranks simultaneously
            dist.broadcast_object_list(meta, src=0)

            # Unpack on all ranks
            model.start_epoch, model.start_batch, sched_state, full_msd, full_osd = meta  # type: ignore

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

        if config.training_spec.device.startswith("cuda"):
            dummy_data = create_dummy_data(config, local_rank)
            with torch.no_grad():
                _ = model(dummy_data, False)

            dist.barrier()

        model.train_model(train_loader, valid_loader, ddp_model=base_model)
        cleanup()
    else:  # DDP
        if config.training_spec.continue_training and latest_model_path:
            checkpoint = torch.load(
                latest_model_path, map_location="cpu", weights_only=False
            )
            base_model.load_state_dict(checkpoint["model_state_dict"])
            base_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            base_model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint["batch"] + 1 >= len(train_loader):
                base_model.start_epoch = checkpoint["epoch"] + 1
                base_model.start_batch = 0
            else:
                base_model.start_epoch = checkpoint["epoch"]
                base_model.start_batch = checkpoint["batch"] + 1
        else:
            model.start_epoch = 1
            model.start_batch = 0
            logger.info(
                f"[INFO] Initializing new model with {format_number(pytorch_total_params)} parameters."
            )

        params_to_optimize = model.parameters()
        model.initialize_optimizer(params=params_to_optimize)

        device_ids = (
            [local_rank] if config.training_spec.device.startswith("cuda") else None
        )
        ddp_model = DDP(model, device_ids=device_ids, find_unused_parameters=False)

        if config.training_spec.device.startswith("cuda"):
            if torch_compile == "outer":
                ddp_model = torch.compile(ddp_model)
            elif torch_compile == "inner":
                for i in range(len(model.layers)):
                    ddp_model.module.layers[i] = torch.compile(
                        ddp_model.module.layers[i]
                    )

        if config.training_spec.device.startswith("cuda"):
            dummy_data = create_dummy_data(config, local_rank)

            if config.training_spec.layer_autocast:
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    _ = ddp_model(dummy_data, False)
            else:
                with torch.no_grad():
                    _ = ddp_model(dummy_data, False)

            dist.barrier()
        model.train_model(train_loader, valid_loader, ddp_model=ddp_model)
        cleanup()


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
    """The main training function.

    Args:
        args: The command-line arguments.
        args_config: The configuration dictionary.
    """
    config_path = args.config_path or "configs/train.yaml"
    config = load_train_config(config_path, args_config, args.skip_metadata)

    torch.set_float32_matmul_precision(config.training_spec.float32_matmul_precision)

    world_size = config.training_spec.world_size
    from_folder = config.read_format == "pt"

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
    else:
        train_worker(0, 1, config, from_folder, 0, config.training_spec.torch_compile)


@beartype
def format_number(number: Union[int, float, np.float32]) -> str:
    """Format a number for display.

    Args:
        number: The number to format.

    Returns:
        A formatted string representation of the number.
    """
    if np.isnan(number):
        return "NaN"
    elif number == 0:
        order_of_magnitude = 0
    else:
        order_of_magnitude = math.floor(math.log(np.abs(number), 10))

    number_adjusted = number * (10 ** (-order_of_magnitude))
    return f"{number_adjusted:5.2f}e{order_of_magnitude}"


class TransformerEmbeddingModel(nn.Module):
    """A wrapper around the TransformerModel to expose the embedding functionality."""

    def __init__(self, transformer_model: "TransformerModel"):
        """Initializes the TransformerEmbeddingModel.

        Args:
            transformer_model: The TransformerModel to wrap.
        """
        super().__init__()
        self.transformer_model = transformer_model
        self.logger = self.transformer_model.logger

    @beartype
    def _copy_model(self):
        """Copies the model.

        This creates a deep copy of the model, typically for saving the
        "best model". It temporarily removes the `log_file` attribute
        before copying to avoid errors, then re-initializes it.

        Returns:
            A deep copy of the current TransformerModel instance.
        """
        logger_ref = self.transformer_model.logger
        del self.transformer_model.logger
        del self.logger
        model_copy = copy.deepcopy(self)
        model_copy.transformer_model._initialize_log_file()
        self.transformer_model.logger = logger_ref
        self.logger = self.transformer_model.logger
        return model_copy

    @conditional_beartype
    def forward(self, src: dict[str, Tensor]):
        """Forward pass for the embedding model.

        Args:
            src: The input data.

        Returns:
            The embedded output.
        """
        return self.transformer_model.forward_embed(src)


class TransformerModel(nn.Module):
    """The main Transformer model for the sequifier.

    This class implements the Transformer model, including the training and
    evaluation loops, as well as the export functionality.
    """

    @beartype
    def __init__(
        self, hparams: Any, rank: Optional[int] = None, local_rank: Optional[int] = None
    ):
        """Initializes the TransformerModel.

        Based on the hyperparameters, this initializes:
        - Embeddings for categorical and real features (self.encoder)
        - Positional encoders (self.pos_encoder)
        - The main TransformerEncoder (self.transformer_encoder)
        - Output decoders for each target column (self.decoder)
        - Loss functions (self.criterion)
        - Optimizer (self.optimizer) and scheduler (self.scheduler)

        Args:
            hparams: The hyperparameters for the model (e.g., from TrainModel config).
            rank: The rank of the current process (for distributed training).
        """
        super().__init__()
        self.project_root = hparams.project_root
        self.model_type = "Transformer"

        self.rank = rank

        self.model_name = hparams.model_name or uuid.uuid4().hex[:8]

        self._initialize_log_file()

        self.logger.info(f"--- Starting Training for model: {self.model_name} ---")

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
        self.seq_length = hparams.seq_length
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
        self.drop = nn.Dropout(hparams.training_spec.dropout)
        self.encoder = ModuleDict()
        self.dim_model = self.hparams.model_spec.dim_model
        self.initial_embedding_dim = self.hparams.model_spec.initial_embedding_dim
        self.joint_embedding_dim = hparams.model_spec.joint_embedding_dim

        if self.joint_embedding_dim is not None:
            self.joint_embedding_layer = nn.Linear(
                self.initial_embedding_dim, self.joint_embedding_dim
            )
        else:
            self.joint_embedding_layer = None

        self.use_rope = hparams.model_spec.positional_encoding == "rope"
        if hparams.model_spec.feature_embedding_dims is not None:
            self.feature_embedding_dims = hparams.model_spec.feature_embedding_dims
        else:
            self.feature_embedding_dims = self._get_feature_embedding_dims(
                self.initial_embedding_dim, self.categorical_columns, self.real_columns
            )

        self.real_columns_with_embedding = []
        self.real_columns_direct = []
        for col in self.real_columns:
            if self.feature_embedding_dims[col] > 1:
                self.encoder[col] = nn.Linear(1, self.feature_embedding_dims[col])
                self.real_columns_with_embedding.append(col)
            else:
                if self.feature_embedding_dims[col] != 1:
                    raise ValueError(
                        f"Real column {col} without embedding must have feature_embedding_dims=1"
                    )
                self.real_columns_direct.append(col)

        for col, n_classes in self.n_classes.items():
            if col in self.categorical_columns:
                self.encoder[col] = nn.Embedding(
                    n_classes, self.feature_embedding_dims[col]
                )

        if not self.use_rope:
            self.pos_encoder = ModuleDict()
            for col in self.real_columns:
                self.pos_encoder[col] = nn.Embedding(
                    self.seq_length, self.feature_embedding_dims[col]
                )
            for col, n_classes in self.n_classes.items():
                if col in self.categorical_columns:
                    self.pos_encoder[col] = nn.Embedding(
                        self.seq_length, self.feature_embedding_dims[col]
                    )
        else:
            self.pos_encoder = None

        self.layers = nn.ModuleList(
            [
                SequifierEncoderLayer(
                    hparams.model_spec,
                    self.dim_model,
                    hparams.model_spec.n_head,
                    hparams.model_spec.dim_feedforward,
                    hparams.training_spec.dropout,
                    hparams.seq_length,
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
            self._generate_square_subsequent_mask(self.seq_length),
            persistent=False,  # Optional: prevents the mask from being saved in your checkpoints
        )

        self._init_weights()

        self.scheduler_step_on = hparams.training_spec.scheduler_step_on

        self.save_interval_epochs = hparams.training_spec.save_interval_epochs
        self.save_latest_interval_minutes = (
            hparams.training_spec.save_latest_interval_minutes
        )
        self.save_batch_interval_minutes = (
            hparams.training_spec.save_batch_interval_minutes
        )
        self.save_batch_interval_minutes_val_loss = (
            hparams.training_spec.save_batch_interval_minutes_val_loss
        )
        self.continue_training = hparams.training_spec.continue_training

        use_scaler = False
        if hparams.training_spec.layer_type_dtypes:
            if "float16" in hparams.training_spec.layer_type_dtypes.values():
                use_scaler = True

        self.scaler = GradScaler(device=self.device.split(":")[0], enabled=use_scaler)

        self._apply_layer_dtypes()

        self.to(self.device)

    @beartype
    def initialize_optimizer(self, params: Any = None) -> None:
        """Initializes the optimizer and scheduler."""
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
        """Casts specific layer types to configured dtypes (e.g., bfloat16, float8)."""
        layer_config = self.hparams.training_spec.layer_type_dtypes

        if not layer_config:
            return

        self.logger.info(f"[INFO] Applying custom layer dtypes: {layer_config}")

        # Iterate over all sub-modules and cast based on type
        for name, module in self.named_modules():
            # Linear Layers
            if isinstance(module, nn.Linear):
                is_decoder = any(module is m for m in self.decoder.values())
                if is_decoder and "decoder" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["decoder"]))
                elif "linear" in layer_config:
                    module.to(dtype=get_torch_dtype(layer_config["linear"]))

            # Embeddings
            elif isinstance(module, nn.Embedding) and "embedding" in layer_config:
                target_dtype = get_torch_dtype(layer_config["embedding"])
                module.to(dtype=target_dtype)

            # Normalization (RMSNorm, LayerNorm)
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
        """Initializes the criterion (loss function) for each target column.

        Args:
            hparams: The hyperparameters for the model, used to find criterion names
                and class weights.

        Returns:
            A dictionary mapping target column names to their loss function instances.
        """
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
        """Calculates the embedding dimension for each column.

        This attempts to distribute the total `embedding_size` across all
        input columns.

        Args:
            embedding_size: The total embedding dimension (initial_embedding_dim).
            categorical_columns: List of categorical column names.
            real_columns: List of real-valued column names.

        Returns:
            A dictionary mapping column names to their calculated embedding dimension.
        """
        if not (len(categorical_columns) + len(real_columns)) > 0:
            raise ValueError("No columns found")

        if len(categorical_columns) == 0 and len(real_columns) > 0:
            if embedding_size < len(real_columns):
                raise ValueError(
                    f"initial_embedding_dim ({embedding_size}) is smaller than the number of real input columns ({len(real_columns)}). "
                    "Cannot allocate at least 1 dimension per column."
                )

            feature_embedding_dims = {col: 1 for col in real_columns}
            column_index = dict(enumerate(real_columns))

            remaining_dims = embedding_size - len(real_columns)
            for i in range(remaining_dims):
                j = i % len(real_columns)
                feature_embedding_dims[column_index[j]] += 1

            if sum(feature_embedding_dims.values()) != embedding_size:
                raise ValueError(
                    f"Auto-calculated embedding dimensions ({sum(feature_embedding_dims.values())}) do not sum to initial_embedding_dim ({embedding_size})."
                )
        elif len(real_columns) == 0 and len(categorical_columns) > 0:
            if embedding_size < len(categorical_columns):
                raise ValueError(
                    f"initial_embedding_dim ({embedding_size}) is smaller than the number of categorical columns ({len(categorical_columns)}). "
                    "Resulting embedding dimension would be 0."
                )

            if (embedding_size % len(categorical_columns)) != 0:
                raise ValueError(
                    f"initial_embedding_dim ({embedding_size}) must be divisible by n_categorical ({len(categorical_columns)})"
                )
            dim_model_comp = embedding_size // len(categorical_columns)
            feature_embedding_dims = {
                col: dim_model_comp for col in categorical_columns
            }
        else:
            raise ValueError(
                "If both real and categorical variables are present, feature_embedding_dims config value must be set"
            )

        return feature_embedding_dims

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag.

        This is used as a mask to prevent attention to future tokens in the
        transformer.

        Args:
            sz: The size of the square mask (sequence length).

        Returns:
            A square tensor of shape (sz, sz) with -inf in the upper triangle.
        """
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    @staticmethod
    def _filter_key(dict_: dict[str, Any], key: str) -> dict[str, Any]:
        """Filters a key from a dictionary.

        Args:
            dict_: The dictionary to filter.
            key: The key to remove.

        Returns:
            A new dictionary without the specified key.
        """
        return {k: v for k, v in dict_.items() if k != key}

    @beartype
    def _init_weights(self) -> None:
        """Initializes the weights of the model."""
        init_std = 0.02
        for col in self.categorical_columns:
            self.encoder[col].weight.data.normal_(mean=0.0, std=init_std)

        for target_column in self.target_columns:
            self.decoder[target_column].bias.data.zero_()
            self.decoder[target_column].weight.data.normal_(mean=0.0, std=init_std)

        if self.pos_encoder is not None:
            for col_name in self.pos_encoder:
                self.pos_encoder[col_name].weight.data.normal_(mean=0.0, std=init_std)

        if self.joint_embedding_layer is not None:
            self.joint_embedding_layer.weight.data.normal_(mean=0.0, std=init_std)
            if self.joint_embedding_layer.bias is not None:
                self.joint_embedding_layer.bias.data.zero_()

    @conditional_beartype
    def _recursive_concat(self, srcs: list[Tensor]):
        """Recursively concatenates a list of tensors.

        This is used to avoid device-specific limits on the number of tensors
        that can be concatenated at once by breaking the operation into
        smaller, recursive chunks.

        Args:
            srcs: A list of tensors to concatenate along dimension 2.

        Returns:
            A single tensor resulting from the recursive concatenation.
        """
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
    def forward_inner(self, src: dict[str, Tensor]) -> Tensor:
        """The inner forward pass of the model.

        This handles embedding lookup, positional encoding, and passing the
        combined tensor through the transformer encoder.

        Args:
            src: A dictionary mapping column names to input tensors
                 (batch_size, seq_length).

        Returns:
            The raw output tensor from the TransformerEncoder
            (seq_length, batch_size, dim_model).
        """
        srcs = []
        for col in self.categorical_columns:
            src_t = self.encoder[col](src[col].T) * math.sqrt(
                self.initial_embedding_dim
            )

            if not self.use_rope:
                pos = (
                    torch.arange(
                        0, self.seq_length, dtype=torch.long, device=src_t.device
                    )
                    .repeat(src_t.shape[1], 1)
                    .T
                )
                src_p = self.pos_encoder[col](pos)  # type: ignore

                src_c = self.drop(src_t + src_p)
            else:
                src_c = self.drop(src_t)

            srcs.append(src_c)

        for col in self.real_columns:
            if col in self.real_columns_direct:
                target_dtype = self.layers[0].ff.get_first_layer_dtype()
                src_t = src[col].T.unsqueeze(2).to(dtype=target_dtype) * math.sqrt(
                    self.initial_embedding_dim
                )
            else:
                assert col in self.real_columns_with_embedding
                layer = self.encoder[col]
                inp = src[col].T[:, :, None].to(dtype=layer.weight.dtype)
                src_t = layer(inp) * math.sqrt(self.initial_embedding_dim)

            if not self.use_rope:
                pos = (
                    torch.arange(
                        0, self.seq_length, dtype=torch.long, device=src_t.device
                    )
                    .repeat(src_t.shape[1], 1)
                    .T
                )
                src_p = self.pos_encoder[col](pos)  # type: ignore
                src_c = self.drop(src_t + src_p)
            else:
                src_c = self.drop(src_t)

            srcs.append(src_c)

        src2 = self._recursive_concat(srcs)
        src2 = src2.transpose(0, 1)

        if self.joint_embedding_layer is not None:
            src2 = self.joint_embedding_layer(src2)

        mask = self.src_mask.to(dtype=src2.dtype)
        for layer in self.layers:
            src2 = layer(src2, src_mask=mask)

        src2 = self.final_norm(src2)

        return src2.transpose(0, 1)

    @conditional_beartype
    def forward_embed(self, src: dict[str, Tensor]) -> Tensor:
        """Forward pass for the embedding model.

        This returns only the embedding from the *last* token in the sequence.

        Args:
            src: A dictionary mapping column names to input tensors
                 (batch_size, seq_length).

        Returns:
            The embedding tensor for the last token
            (batch_size, dim_model).
        """
        return self.forward_inner(src)[-self.prediction_length :, :, :]

    @conditional_beartype
    def forward_train(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass for training.

        This runs the inner forward pass and then applies the appropriate
        decoder for each target column.

        Args:
            src: A dictionary mapping column names to input tensors
                 (batch_size, seq_length).

        Returns:
            A dictionary mapping target column names to their raw output
            (logit) tensors (seq_length, batch_size, n_classes/1).
        """
        output = self.forward_inner(src)
        output = {
            target_column: self.decode(target_column, output)
            for target_column in self.target_columns
        }

        return output

    @conditional_beartype
    def decode(self, target_column: str, output: Tensor) -> Tensor:
        """Decodes the output of the transformer encoder.

        Applies the appropriate final linear layer for a given target column.

        Args:
            target_column: The name of the target column to decode.
            output: The raw output tensor from the TransformerEncoder
                    (seq_length, batch_size, dim_model).

        Returns:
            The decoded output (logits or real value) for the target column
            (seq_length, batch_size, n_classes/1).
        """

        target_dtype = self.decoder[target_column].weight.dtype
        decoded = self.decoder[target_column](output.to(target_dtype)).to(torch.float32)

        return decoded

    @conditional_beartype
    def apply_softmax(self, target_column: str, output: Tensor) -> Tensor:
        """Applies softmax to the output of the decoder.

        If the target is real, it returns the output unchanged.
        If the target is categorical, it applies LogSoftmax.

        Args:
            target_column: The name of the target column.
            output: The decoded output tensor (logits or real value).

        Returns:
            The output tensor, with LogSoftmax applied if categorical.
        """
        if target_column in self.real_columns:
            return output
        else:
            return self.softmax[target_column](output.float())

    @conditional_beartype
    def forward(
        self, src: dict[str, Tensor], return_logits: Union[bool, Tensor] = False
    ) -> dict[str, Tensor]:
        """The main forward pass of the model.

        This is typically used for inference/evaluation, returning the
        probabilities/values for the *last* token in the sequence.

        Args:
            src: A dictionary mapping column names to input tensors
                 (batch_size, seq_length).
            return_logits: Return logits

        Returns:
            A dictionary mapping target column names to their final
            output (LogSoftmax probabilities or real values) for the
            last token (batch_size, n_classes/1).
        """
        output = self.forward_train(src)
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
        if self.hparams.training_spec.fsdp:
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
    def train_model(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        ddp_model: Optional[nn.Module] = None,
    ) -> None:
        """Trains the model.

        This method contains the main training loop, including epoch iteration,
        validation, early stopping logic, and model saving/exporting.

        Args:
            train_loader: DataLoader for the training dataset.
            valid_loader: DataLoader for the validation dataset.
            ddp_model: ddp model
        """
        best_val_loss = float("inf")
        n_epochs_no_improvement = 0
        last_epoch = self.start_epoch - 1
        best_model_state = None

        try:
            self.last_latest_save_time = time.time()
            self.last_batch_save_time = time.time()

            if self.start_epoch == 1:
                total_loss, total_losses, output = self._evaluate(
                    valid_loader, ddp_model
                )
                elapsed = 0.0

                self._log_epoch_results(0, 0, elapsed, total_loss, total_losses, output)
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

                    self._train_epoch(train_loader, valid_loader, epoch, ddp_model)

                    total_loss, total_losses, output = self._evaluate(
                        valid_loader, ddp_model
                    )
                    elapsed = time.time() - epoch_start_time

                    self._log_epoch_results(
                        epoch,
                        len(train_loader),
                        elapsed,
                        total_loss,
                        total_losses,
                        output,
                    )

                    if total_loss < best_val_loss:
                        best_val_loss = total_loss
                        best_model_state = self._get_full_state_dict(ddp_model)
                        n_epochs_no_improvement = 0
                    else:
                        n_epochs_no_improvement += 1

                    if self.scheduler_step_on == "epoch":
                        self.scheduler.step()

                    if epoch % self.save_interval_epochs == 0:
                        self._save(
                            epoch,
                            len(train_loader) - 1,
                            total_loss,
                            ddp_model=ddp_model,
                            suffix=f"epoch-{epoch}",
                        )

                    last_epoch = epoch
        except KeyboardInterrupt:
            self.logger.info("\n" + "=" * 89)
            self.logger.info("[WARNING] Training interrupted by user (Ctrl+C).")

            if self.hparams.training_spec.distributed:
                dist.barrier()

            # 1. Use a list to hold the answer so it can be broadcasted across ranks
            answer_list = ["n"]

            # 2. Only Rank 0 prompts the user
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

            # 3. Broadcast the decision to all GPUs so they stay in sync
            if self.hparams.training_spec.distributed:
                dist.broadcast_object_list(answer_list, src=0)

            # 4. If the decision is 'y', ALL ranks must participate in state dict extraction
            if answer_list[0] == "y":
                if self.rank == 0:
                    self.logger.info("[INFO] User opted to export models.")

                if last_epoch is not None and best_model_state is not None:
                    if self.rank == 0:
                        self.logger.info(
                            f"[INFO] Exporting 'last' model from epoch {last_epoch}..."
                        )

                    # ALL RANKS MUST EXECUTE THIS to prevent FSDP all_gather deadlocks
                    last_model_state = self._get_full_state_dict(ddp_model)

                    # ONLY Rank 0 executes the file I/O
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

        # 2. Restrict the export saving to Rank 0 inside the _export method (which you already do)
        # or guard the I/O specifically:
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
    ) -> None:
        """Trains the model for one epoch.

        Iterates through the training DataLoader, computes loss, performs
        backpropagation, and updates model parameters. The DataLoader is expected
        to yield tuples of (sequences_dict, targets_dict, sequence_ids, subsequence_ids, start_positions).
        The IDs and positions are currently unused in this training loop.

        Args:
            train_loader: DataLoader for the training dataset.
            epoch: The current epoch number (used for logging).
        """
        total_loss = 0.0
        batches_aggregated = 0

        start_time = time.time()
        num_batches = len(train_loader)
        start_batch = self.start_batch
        self.start_batch = 0

        model_to_call = ddp_model if ddp_model is not None else self

        model_to_call.train()

        is_fsdp = self.hparams.training_spec.fsdp

        for batch_count, (data, targets, _, _, _) in enumerate(train_loader):
            if batch_count >= start_batch:
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

                # Only use standard torch.autocast if FSDP MixedPrecision is NOT handling it natively
                if self.hparams.training_spec.layer_autocast and not is_fsdp:
                    amp_dtype = get_torch_dtype(
                        self.hparams.training_spec.layer_type_dtypes.get(
                            "linear", "bfloat16"
                        )
                        if self.hparams.training_spec.layer_type_dtypes
                        else "float32"
                    )
                    with torch.autocast(
                        device_type=self.device.split(":")[0], dtype=amp_dtype
                    ):
                        output = model_to_call(data, True)
                        loss, losses = self._calculate_loss(output, targets)
                else:
                    output = model_to_call(data, True)
                    loss, losses = self._calculate_loss(output, targets)

                self.scaler.scale(loss).backward()

                if (
                    self.accumulation_steps is None
                    or (batch_count + 1) % self.accumulation_steps == 0
                    or (batch_count + 1) == num_batches
                ):
                    self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                batches_aggregated += 1
                if (batch_count + 1) % self.log_interval == 0 and self.rank == 0:
                    learning_rate = self.scheduler.get_last_lr()[0]
                    s_per_batch = (time.time() - start_time) / max(
                        1, batches_aggregated
                    )
                    avg_train_loss = total_loss / max(1, batches_aggregated)
                    self.logger.info(
                        f"[INFO] Epoch {epoch:3d} | Batch {(batch_count+1):5d}/{num_batches:5d} | Loss: {format_number(avg_train_loss)} | LR: {format_number(learning_rate)} | S/Batch {format_number(s_per_batch)}"
                    )
                    total_loss = 0.0
                    batches_aggregated = 0
                    self.start_batch = 0
                    start_time = time.time()

                del data, targets, output, loss, losses

                if self.scheduler_step_on == "batch":
                    self.scheduler.step()

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

                if not self.hparams.training_spec.distributed or self.rank == 0:
                    if self.save_latest_interval_minutes is not None and (
                        current_time - self.last_latest_save_time
                    ) >= (self.save_latest_interval_minutes * 60):
                        current_time = time.time()
                        should_save_latest[0] = 1
                        self.last_latest_save_time = current_time

                    if self.save_batch_interval_minutes is not None and (
                        current_time - self.last_batch_save_time
                    ) >= (self.save_batch_interval_minutes * 60):
                        should_save_batch[0] = 1
                        self.last_batch_save_time = current_time

                if self.hparams.training_spec.distributed:
                    dist.broadcast(should_save_latest, src=0)
                    dist.broadcast(should_save_batch, src=0)
                    dist.barrier()

                if should_save_batch.item() == 1:
                    if self.save_batch_interval_minutes_val_loss:
                        val_loss, val_losses, output = self._evaluate(
                            valid_loader, ddp_model
                        )

                        if not self.hparams.training_spec.distributed or self.rank == 0:
                            self._log_epoch_results(
                                0,
                                batch_count + 1,
                                (current_time - self.last_batch_save_time),
                                val_loss,
                                val_losses,
                                output,
                            )
                            val_loss_batch[0] = val_loss
                    else:
                        val_loss_batch[0] = np.float32(np.nan)

                if self.hparams.training_spec.distributed:
                    dist.broadcast(val_loss_batch, src=0)

                if should_save_latest.item() == 1:
                    self._save(
                        epoch,
                        batch_count,
                        np.float32(np.nan),
                        ddp_model,
                        suffix="latest",
                    )
                    if self.rank != 0:
                        self.last_latest_save_time = time.time()

                val_loss = np.float32(val_loss_batch.item())
                if val_loss != 0:
                    self._save(
                        epoch,
                        batch_count,
                        val_loss,  # type: ignore
                        ddp_model,
                        suffix=f"epoch-{epoch}-batch-{batch_count + 1}",
                    )
                    if self.rank != 0:
                        self.last_batch_save_time = time.time()

    @beartype
    def _calculate_loss(
        self, output: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Calculates the loss for the given output and targets.

        Compares the model's output (from `forward_train`) with the target
        values, applying the appropriate criterion for each target column
        and combining them using loss weights.

        Args:
            output: A dictionary of output tensors from the model
                    (seq_length, batch_size, n_classes/1).
            targets: A dictionary of target tensors
                     (batch_size, seq_length).

        Returns:
            A tuple containing:
            - The total combined (weighted) loss as a single Tensor.
            - A dictionary of individual (unweighted) loss Tensors for each
              target column.
        """
        mask_col = next(
            (
                col
                for col in targets.keys()
                if self.target_column_types[col] == "categorical"
            ),
            list(targets.keys())[0],
        )

        if self.target_column_types[mask_col] == "real":
            seq_mask_2d = (targets[mask_col] != 0.0).long().cumsum(dim=1) > 0
        else:
            seq_mask_2d = targets[mask_col] != 0

        mask = seq_mask_2d.T.contiguous().reshape(-1)

        losses = {}
        for target_column in targets.keys():
            target_column_type = self.target_column_types[target_column]
            if target_column_type == "categorical":
                output[target_column] = (
                    output[target_column]
                    .float()
                    .reshape(-1, self.n_classes[target_column])
                )
            elif target_column_type == "real":
                output[target_column] = (
                    output[target_column].to(dtype=torch.float32).reshape(-1)
                )

            target_tensor = targets[target_column].T.contiguous().reshape(-1)

            if self.target_column_types[target_column] == "real":
                target_tensor = target_tensor.to(dtype=output[target_column].dtype)

            raw_loss = self.criterion[target_column](
                output[target_column], target_tensor
            )

            current_mask = mask.to(dtype=raw_loss.dtype)

            losses[target_column] = (raw_loss * current_mask).sum() / (
                current_mask.sum() + 1e-9
            )

        loss = None
        for target_column in targets.keys():
            losses[target_column] = losses[target_column] * (
                self.loss_weights[target_column]
                if self.loss_weights is not None
                else 1.0
            )
            if loss is None:
                loss = losses[target_column].clone()
            else:
                loss += losses[target_column]

        if loss is None:
            raise RuntimeError(
                "Loss calculation failed; no loss tensors were generated."
            )

        return loss, losses

    @beartype
    def _copy_model(self):
        """Copies the model.

        This creates a deep copy of the model, typically for saving the
        "best model". It temporarily removes the `log_file` attribute
        before copying to avoid errors, then re-initializes it.

        Returns:
            A deep copy of the current TransformerModel instance.
        """
        logger_ref = self.logger
        del self.logger
        model_copy = copy.deepcopy(self)
        model_copy._initialize_log_file()
        self.logger = logger_ref
        return model_copy

    @beartype
    def _transform_val(self, col: str, val: Tensor) -> Tensor:
        """ "Transforms input data to match the format of model output.

        This is used *only* for calculating the baseline loss, where
        the input (e.g., categorical indices) needs to be one-hot encoded
        to be comparable to the model's (logit) output.

        Args:
            col: The name of the column being transformed.
            val: The input tensor (categorical indices).

        Returns:
            A tensor transformed to be compatible with the loss function
            (e.g., one-hot encoded).
        """
        if self.target_column_types[col] == "categorical":
            target_dtype = self.decoder[col].weight.dtype
            return (
                one_hot(val, self.n_classes[col])
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
    ) -> tuple[np.float32, dict[str, np.float32], dict[str, Tensor]]:
        """Evaluates the model on the validation set.

        Iterates through the validation data, calculates the total loss,
        and aggregates results across all processes if in distributed mode.
        Also calculates a one-time baseline loss on the first call.
        The DataLoader is expected to yield tuples of
        (sequences_dict, targets_dict, sequence_ids, subsequence_ids, start_positions).
        The IDs and positions are currently unused during evaluation.

        Args:
            valid_loader: DataLoader for the validation dataset.
            ddp_model: DDP model

        Returns:
            A tuple containing:
            - The total aggregated validation loss (float).
            - A dictionary of aggregated losses for each target column (dict[str, float]).
            - The output tensor dictionary from the last batch (used for class share logging).
        """

        total_loss_collect = []
        # Initialize a dict to hold lists of losses for each target
        total_losses_collect = {col: [] for col in self.target_columns}
        output = {}  # for type checking

        model_to_call = ddp_model if ddp_model is not None else self

        model_to_call.eval()

        is_fsdp = self.hparams.training_spec.fsdp

        with torch.no_grad():
            for data, targets, _, _, _ in valid_loader:
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

                if self.hparams.training_spec.layer_autocast and not is_fsdp:
                    amp_dtype = get_torch_dtype(
                        self.hparams.training_spec.layer_type_dtypes.get(
                            "linear", "bfloat16"
                        )
                        if self.hparams.training_spec.layer_type_dtypes
                        else "float32"
                    )
                    with torch.autocast(
                        device_type=self.device.split(":")[0], dtype=amp_dtype
                    ):
                        output = model_to_call(data, True)
                        loss, losses = self._calculate_loss(output, targets)
                else:
                    output = model_to_call(data, True)
                    loss, losses = self._calculate_loss(output, targets)

                total_loss_collect.append(loss.item())
                for col, loss in losses.items():
                    total_losses_collect[col].append(loss.item())

                # Free up GPU memory
                del data, targets, loss, losses
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if len(total_loss_collect) > 0:
            total_loss_local = np.mean(total_loss_collect)
            total_losses_local = {
                col: np.mean(loss_list)
                for col, loss_list in total_losses_collect.items()
            }
        else:
            # Handle empty validation set case
            total_loss_local = 0.0
            total_losses_local = {col: 0.0 for col in self.target_columns}

        # 2. Aggregate losses across all GPUs if in distributed mode
        if self.hparams.training_spec.distributed:
            # Put local losses into tensors for reduction
            total_loss_tensor = torch.tensor(
                total_loss_local, device=self.device, dtype=torch.float32
            )

            # Ensure consistent order for the losses tensor
            loss_keys = sorted(total_losses_local.keys())
            losses_values = [total_losses_local[k] for k in loss_keys]
            losses_tensor = torch.tensor(
                losses_values, device=self.device, dtype=torch.float32
            )

            # Sum losses from all processes. The result is broadcast back to all processes.
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)

            world_size = dist.get_world_size()
            total_loss_tensor /= world_size
            losses_tensor /= world_size

            # Update local variables with the aggregated global results
            total_loss_global = total_loss_tensor.cpu().numpy()
            losses_global_values = losses_tensor.cpu().numpy()
            total_losses_global = dict(zip(loss_keys, losses_global_values))
        else:
            # If not distributed, local losses are the global losses
            total_loss_global = total_loss_local
            total_losses_global = total_losses_local

        # 3. Handle one-time baseline loss calculation (must also be synchronized)
        if not hasattr(self, "baseline_loss"):
            baseline_loss_local_collect = []
            baseline_losses_local_collect = {col: [] for col in self.target_columns}

            # Iterate over the sharded validation loader
            for data, targets, _, _, _ in valid_loader:
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

                pseudo_output = {}
                targets_for_baseline = {}
                for col in self.target_columns:
                    if col in data:
                        pseudo_output[col] = self._transform_val(
                            col, data[col].transpose(0, 1)
                        )
                        targets_for_baseline[col] = targets[col]

                if len(pseudo_output) > 0:
                    loss, losses = self._calculate_loss(
                        pseudo_output, targets_for_baseline
                    )
                    baseline_loss_local_collect.append(loss.item())
                    for col, loss_ in losses.items():
                        baseline_losses_local_collect[col].append(loss_.item())

            # Sum the losses for the local shard
            if len(baseline_loss_local_collect):
                baseline_loss_local = np.mean(baseline_loss_local_collect)
                baseline_losses_local = {
                    col: np.mean(loss_list)
                    for col, loss_list in baseline_losses_local_collect.items()
                }
            else:
                baseline_loss_local = -1.0
                baseline_losses_local = {col: -1.0 for col in self.target_columns}

            # Broadcast the baseline values from the main process to all others
            if self.hparams.training_spec.distributed:
                total_loss_tensor = torch.tensor(
                    baseline_loss_local, device=self.device, dtype=torch.float32
                )
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                loss_keys = sorted(baseline_losses_local.keys())
                losses_values = [baseline_losses_local[k] for k in loss_keys]
                losses_tensor = torch.tensor(
                    losses_values, device=self.device, dtype=torch.float32
                )
                dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)

                world_size = dist.get_world_size()
                total_loss_tensor /= world_size
                losses_tensor /= world_size

                self.baseline_loss = total_loss_tensor.item()
                self.baseline_losses = dict(zip(loss_keys, losses_tensor.cpu().numpy()))
            else:
                # If not distributed, local is global
                self.baseline_loss = baseline_loss_local
                self.baseline_losses = baseline_losses_local

        model_to_call.train()
        torch.clear_autocast_cache()

        return (
            np.float32(total_loss_global),
            {k: np.float32(v) for k, v in total_losses_global.items()},
            output,
        )

    @beartype
    def _export(
        self,
        state_dict: dict[str, Tensor],
        suffix: str,
        epoch: int,
        clean: bool = False,
    ) -> None:
        """Exports the model.

        This is a wrapper function that handles exporting the model (and
        optionally the embedding-only model) on rank 0 only.

        Args:
            state_dict: The state dict of the model instance to export (e.g., best model or last model).
            suffix: A string suffix to append to the model filename (e.g., "best", "last").
            epoch: The current epoch number, included in the filename.
        """
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
        """Exports the model to ONNX and/or PyTorch format.

        Saves the model weights as a .pt file and/or exports the model
        graph and weights as an .onnx file based on the config.

        Args:
            model: The model instance (TransformerModel or TransformerEmbeddingModel).
            suffix: A string suffix for the filename (e.g., "best", "last-embedding").
            epoch: The current epoch number, included in the filename.
        """
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
                    0, self.n_classes[col], (self.inference_batch_size, self.seq_length)
                ).to(export_device, non_blocking=True)
                for col in self.categorical_columns
            }

            dtype_real = torch.float32 if is_different_type else None
            x_real = {
                col: torch.rand(self.inference_batch_size, self.seq_length).to(
                    export_device, non_blocking=True, dtype=dtype_real
                )
                for col in self.real_columns
            }

            x = {"src": {**x_cat, **x_real}}

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

            torch.onnx.export(
                model_to_export,
                x,  # model input (or a tuple for multiple inputs)
                export_path,  # where to save the model
                export_params=True,  # store the trained parameter weights
                opset_version=14,  # the ONNX version
                do_constant_folding=constant_folding,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
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
    ) -> None:
        """Saves the model checkpoint.

        Saves the model state, optimizer state, and epoch number to a .pt
        file in the checkpoints directory. Only runs on rank 0.

        Args:
            val_loss: The validation loss at the current epoch.
            ddp_model: DDP model
            suffix: Checkpoint file suffix.
        """
        model_to_extract = ddp_model if ddp_model is not None else self
        is_fsdp = self.hparams.training_spec.fsdp

        if is_fsdp:
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
            optim_state_dict = self.optimizer.state_dict()

        if self.rank != 0:
            return

        os.makedirs(os.path.join(self.project_root, "checkpoints"), exist_ok=True)

        file_name = f"{self.model_name}-{suffix}.pt"

        output_path = os.path.join(
            self.project_root,
            "checkpoints",
            file_name,
        )

        torch.save(
            {
                "epoch": epoch,
                "batch": batch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optim_state_dict,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": val_loss,
            },
            output_path,
        )
        self.logger.info(f"[INFO] Saved checkpoint to {output_path}")

    @beartype
    def _get_optimizer(self, params: Any, **kwargs):
        """Gets the optimizer.

        Initializes the optimizer specified in the hyperparameters.

        Args:
            params: params
            **kwargs: Additional arguments to pass to the optimizer constructor
                      (e.g., weight_decay).

        Returns:
            An initialized torch.optim.Optimizer instance.
        """
        optimizer_class = get_optimizer_class(self.hparams.training_spec.optimizer.name)
        return optimizer_class(
            params, lr=self.hparams.training_spec.learning_rate, **kwargs
        )

    @beartype
    def _get_scheduler(self, **kwargs):
        """Gets the scheduler.

        Initializes the learning rate scheduler specified in the hyperparameters.

        Args:
            **kwargs: Additional arguments to pass to the scheduler constructor
                      (e.g., step_size).

        Returns:
            An initialized torch.optim.lr_scheduler._LRScheduler instance.
        """
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
        """Initializes the log file."""
        # Replaces old LogFile class instantiation
        self.logger = configure_logger(self.project_root, self.model_name, self.rank)

    @beartype
    def _get_latest_model_name(self) -> Optional[str]:
        """Gets the name of the latest model checkpoint.

        Scans the checkpoints directory for files matching the current
        `model_name` and returns the path to the most recently modified one.

        Returns:
            The file path (str) to the latest checkpoint, or None if no
            checkpoint is found.
        """
        checkpoint_path = os.path.join(self.project_root, "checkpoints", "*")

        files = glob.glob(checkpoint_path)
        files = [
            file for file in files if os.path.split(file)[1].startswith(self.model_name)
        ]
        if files:
            return max(files, key=os.path.getctime)
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
        output: dict[str, Tensor],
    ) -> None:
        """Logs the results of an epoch.

        Writes validation loss, individual losses, learning rate, and
        class share statistics (if configured) to the log file.
        Only runs on rank 0.

        Args:
            epoch: Current epoch number.
            elapsed: Time taken for the epoch (in seconds).
            total_loss: The total aggregated validation loss.
            total_losses: A dictionary of aggregated losses for each target.
            output: The output tensor dictionary from the last validation batch,
                    used for class share logging.
            batch: Current batch number.
        """
        if self.rank == 0:
            learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]

            log_string = f"[INFO] Validation | Epoch: {epoch:3d} | Batch: {batch} | Loss: {format_number(total_loss)} | Baseline Loss: {format_number(self.baseline_loss)} | Time: {elapsed:5.2f}s | LR {format_number(learning_rate)}"

            self.logger.info("-" * 89)
            self.logger.info(log_string)

            if len(total_losses) > 1:
                loss_strs = [
                    f"{key}_loss: {format_number(value)}"
                    for key, value in total_losses.items()
                ]
                self.logger.info("[INFO]  - " + ", ".join(loss_strs))

            for categorical_column in self.class_share_log_columns:
                output_values = (
                    output[categorical_column].argmax(1).cpu().detach().numpy()
                )
                output_counts_df = (
                    pl.Series("values", output_values).value_counts().sort("values")
                )
                output_counts = output_counts_df.get_column("count")

                output_counts = output_counts / output_counts.sum()
                value_shares = " | ".join(
                    [
                        f"{self.index_maps[categorical_column][row['values']]}: {row['count']:5.5f}"
                        for row in output_counts_df.iter_rows(named=True)
                    ]
                )
                self.logger.info(f"[INFO] {categorical_column}: {value_shares}")

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
    """Loads a trained model for inference.

    Args:
        model_type: "generative" or "embedding".
        model_path: Path to the saved .pt model file.
        training_config_path: Path to the .yaml config file used for training.
        args_config: A dictionary of override configurations.
        device: The device to load the model onto (e.g., "cuda", "cpu").
        infer_with_dropout: Whether to force dropout layers to be active
                          during inference.

    Returns:
        The loaded and compiled torch.nn.Module (TransformerModel or
        TransformerEmbeddingModel) in evaluation mode.
    """
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
) -> np.ndarray:
    """Performs inference with an embedding model.

    Args:
        model: The loaded TransformerEmbeddingModel.
        x: A list of input data dictionaries (batched).
        device: The device to run inference on.
        size: The total number of samples (unused in this function).
        target_columns: List of target column names (unused in this function).

    Returns:
        A NumPy array containing the concatenated embeddings from all batches.
    """
    outs0 = []

    categorical_cols = set(model.transformer_model.categorical_columns)

    with torch.no_grad():
        for x_sub in x:
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
                    data_gpu[col] = torch.from_numpy(x_).to(device, dtype=ref_dtype)

            output_gpu = model.forward(data_gpu)
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
) -> dict[str, np.ndarray]:
    """Performs inference with a generative model.

    Args:
        model: The loaded TransformerModel.
        x: A list of input data dictionaries (batched).
        device: The device to run inference on.
        size: The total number of samples to trim the final output to.
        target_columns: List of target column names to extract from the output.

    Returns:
        A dictionary mapping target column names to their concatenated
        output NumPy arrays, trimmed to `size`.
    """
    outs0 = []

    categorical_cols = set(model.categorical_columns)

    with torch.no_grad():
        for x_sub in x:
            layer_types = model.hparams.training_spec.layer_type_dtypes or {}
            dtype_str = layer_types.get("linear", "float32")
            ref_dtype = get_torch_dtype(dtype_str)
            data_gpu = {}
            for col, x_ in x_sub.items():
                if col in categorical_cols:
                    data_gpu[col] = torch.from_numpy(x_).to(device, dtype=torch.int64)
                else:
                    data_gpu[col] = torch.from_numpy(x_).to(device, dtype=ref_dtype)

            output_gpu = model.forward(data_gpu)
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
