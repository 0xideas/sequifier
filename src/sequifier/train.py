import copy
import glob
import math
import os
import time
import uuid
import warnings
from typing import Any, Optional, Union

import numpy as np
import polars as pl
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from beartype import beartype
from torch import Tensor, nn
from torch.nn import ModuleDict, TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import one_hot
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

torch._dynamo.config.suppress_errors = True
from sequifier.config.train_config import TrainModel, load_train_config  # noqa: E402
from sequifier.helpers import LogFile  # noqa: E402
from sequifier.helpers import construct_index_maps  # noqa: E402
from sequifier.io.sequifier_dataset_from_file import (  # noqa: E402
    SequifierDatasetFromFile,
)
from sequifier.io.sequifier_dataset_from_folder import (  # noqa: E402
    SequifierDatasetFromFolder,
)
from sequifier.io.sequifier_dataset_from_folder_lazy import (  # noqa: E402
    SequifierDatasetFromFolderLazy,
)
from sequifier.optimizers.optimizers import get_optimizer_class  # noqa: E402
from sequifier.samplers.distributed_grouped_random_sampler import (  # noqa: E402
    DistributedGroupedRandomSampler,
)


@beartype
def setup(rank: int, world_size: int, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@beartype
def train_worker(rank: int, world_size: int, config: TrainModel, from_folder: bool):
    if config.training_spec.distributed:
        setup(rank, world_size, config.training_spec.backend)

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
        assert config.training_spec.distributed == False  # noqa: E712
        train_dataset = SequifierDatasetFromFile(config.training_data_path, config)
        valid_dataset = SequifierDatasetFromFile(config.validation_data_path, config)

    if from_folder:
        if config.training_spec.distributed:
            # 2. Use the new distributed sampler for the multi-GPU case
            train_sampler = DistributedGroupedRandomSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )
            valid_sampler = DistributedGroupedRandomSampler(
                valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            # Use the simple grouped sampler for the single-GPU case
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = None
    else:
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            if config.training_spec.distributed
            else None
        )
        valid_sampler = (
            DistributedSampler(
                valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if config.training_spec.distributed
            else None
        )

    if from_folder:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training_spec.batch_size,
            sampler=train_sampler,
            shuffle=False,  # Shuffle only if not using sampler
            num_workers=config.training_spec.num_workers,  # Use multiple workers for data loading
            pin_memory=config.training_spec.device not in ["mps", "cpu"],
        )

        # For validation, it's often fine to just run it on the main process
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.training_spec.batch_size,
            sampler=valid_sampler,
            shuffle=False,
        )
    elif not from_folder:
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            sampler=None,
            num_workers=config.training_spec.num_workers,
            pin_memory=False,
            persistent_workers=(config.training_spec.num_workers > 0),
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=None, sampler=None, shuffle=False
        )
    else:
        assert False, "not possible"

    # 2. Instantiate and wrap the model
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = TransformerModel(config, rank)

    if config.training_spec.distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    model = torch.compile(model)

    # 3. Start training
    # When using DDP, the original model is accessed via the .module attribute
    original_model = model.module if config.training_spec.distributed else model
    original_model.train_model(train_loader, valid_loader, train_sampler, valid_sampler)

    if config.training_spec.distributed:
        cleanup()


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    config_path = args.config_path or "configs/train.yaml"
    config = load_train_config(config_path, args_config, args.on_unprocessed)
    print(f"--- Starting Training for model: {config.model_name} ---")

    world_size = config.training_spec.world_size

    from_folder = config.read_format == "pt"
    if config.training_spec.distributed:
        mp.spawn(
            train_worker,
            args=(world_size, config, from_folder),
            nprocs=world_size,
            join=True,
        )
    else:
        # Fallback to single-GPU/CPU training
        train_worker(0, world_size, config, from_folder)


@beartype
def format_number(number: Union[int, float, np.float32]) -> str:
    """
    Format a number for display.

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
        order_of_magnitude = math.floor(math.log(number, 10))

    number_adjusted = number * (10 ** (-order_of_magnitude))
    return f"{number_adjusted:5.2f}e{order_of_magnitude}"


class TransformerEmbeddingModel(nn.Module):
    def __init__(self, transformer_model: "TransformerModel"):
        super().__init__()
        self.transformer_model = transformer_model

    def forward(self, src: dict[str, Tensor]):
        return self.transformer_model.forward_embed(src)


class TransformerModel(nn.Module):
    @beartype
    def __init__(self, hparams: Any, rank: Optional[int] = None):
        super().__init__()
        self.project_path = hparams.project_path
        self.model_type = "Transformer"
        self.model_name = hparams.model_name or uuid.uuid4().hex[:8]

        self.rank = rank

        self.selected_columns = hparams.selected_columns
        self.categorical_columns = [
            col
            for col in hparams.categorical_columns
            if self.selected_columns is None or col in self.selected_columns
        ]
        self.real_columns = [
            col
            for col in hparams.real_columns
            if self.selected_columns is None or col in self.selected_columns
        ]

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
        self.pos_encoder = ModuleDict()
        self.embedding_size = max(
            self.hparams.model_spec.d_model, self.hparams.model_spec.nhead
        )
        if hparams.model_spec.d_model_by_column is not None:
            self.d_model_by_column = hparams.model_spec.d_model_by_column
        else:
            self.d_model_by_column = self._get_d_model_by_column(
                self.embedding_size, self.categorical_columns, self.real_columns
            )

        self.real_columns_with_embedding = []
        self.real_columns_direct = []
        for col in self.real_columns:
            if self.d_model_by_column[col] > 1:
                self.encoder[col] = nn.Linear(1, self.d_model_by_column[col])
                self.real_columns_with_embedding.append(col)
            else:
                assert self.d_model_by_column[col] == 1
                self.real_columns_direct.append(col)
            self.pos_encoder[col] = nn.Embedding(
                self.seq_length, self.d_model_by_column[col]
            )
        for col, n_classes in self.n_classes.items():
            if col in self.categorical_columns:
                self.encoder[col] = nn.Embedding(n_classes, self.d_model_by_column[col])
                self.pos_encoder[col] = nn.Embedding(
                    self.seq_length, self.d_model_by_column[col]
                )

        encoder_layers = TransformerEncoderLayer(
            self.embedding_size,
            hparams.model_spec.nhead,
            hparams.model_spec.d_hid,
            hparams.training_spec.dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, hparams.model_spec.nlayers, enable_nested_tensor=False
        )

        self.decoder = ModuleDict()
        self.softmax = ModuleDict()
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                self.decoder[target_column] = nn.Linear(
                    self.embedding_size,
                    self.n_classes[target_column],
                )
                self.softmax[target_column] = nn.LogSoftmax(dim=-1)
            elif target_column_type == "real":
                self.decoder[target_column] = nn.Linear(self.embedding_size, 1)
            else:
                raise ValueError(
                    f"Target column type {target_column_type} not in ['categorical', 'real']"
                )

        self.device = hparams.training_spec.device
        self.device_max_concat_length = hparams.training_spec.device_max_concat_length

        if hparams.training_spec.device == "cuda" and self.rank is not None:
            self.device = f"cuda:{self.rank}"
        else:
            self.device = hparams.training_spec.device

        self.to(self.device)

        self.criterion = self._init_criterion(hparams=hparams)
        self.batch_size = hparams.training_spec.batch_size
        self.accumulation_steps = hparams.training_spec.accumulation_steps

        self.src_mask = self._generate_square_subsequent_mask(self.seq_length).to(
            self.device
        )

        self._init_weights()
        self.optimizer = self._get_optimizer(
            **self._filter_key(hparams.training_spec.optimizer, "name")
        )
        self.scheduler = self._get_scheduler(
            **self._filter_key(hparams.training_spec.scheduler, "name")
        )

        self.iter_save = hparams.training_spec.iter_save
        self.continue_training = hparams.training_spec.continue_training
        load_string = self._load_weights_conditional()
        self._initialize_log_file()
        self.log_file.write(load_string)

    @beartype
    def _init_criterion(self, hparams: Any) -> dict[str, Any]:
        criterion = {}
        for target_column in self.target_columns:
            criterion_class = eval(
                f"torch.nn.{hparams.training_spec.criterion[target_column]}"
            )
            criterion_kwargs = {}
            if (
                hparams.training_spec.class_weights is not None
                and target_column in hparams.training_spec.class_weights
            ):
                criterion_kwargs["weight"] = Tensor(
                    hparams.training_spec.class_weights[target_column]
                ).to(self.device)
            criterion[target_column] = criterion_class(**criterion_kwargs)
        return criterion

    @beartype
    def _get_d_model_by_column(
        self,
        embedding_size: int,
        categorical_columns: list[str],
        real_columns: list[str],
    ) -> dict[str, int]:
        print(f"{len(categorical_columns) = } {len(real_columns) = }")
        assert (len(categorical_columns) + len(real_columns)) > 0, "No columns found"
        if len(categorical_columns) == 0 and len(real_columns) > 0:
            d_model_by_column = {col: 1 for col in real_columns}
            column_index = dict(enumerate(real_columns))
            for i in range(embedding_size):
                if sum(d_model_by_column.values()) % embedding_size != 0:
                    j = i % len(real_columns)
                    d_model_by_column[column_index[j]] += 1
            assert sum(d_model_by_column.values()) % embedding_size == 0
        elif len(real_columns) == 0 and len(categorical_columns) > 0:
            assert (
                (embedding_size % len(categorical_columns)) == 0
            ), f"If only categorical variables are included, d_model must be a multiple of the number of categorical variables ({embedding_size = } % {len(categorical_columns) = }) != 0"
            d_model_comp = embedding_size // len(categorical_columns)
            d_model_by_column = {col: d_model_comp for col in categorical_columns}
        else:
            raise UserWarning(
                "If both real and categorical variables are present, d_model_by_column config value must be set"
            )

        return d_model_by_column

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    @staticmethod
    def _filter_key(dict_: dict[str, Any], key: str) -> dict[str, Any]:
        return {k: v for k, v in dict_.items() if k != key}

    @beartype
    def _init_weights(self) -> None:
        init_std = 0.02
        for col in self.categorical_columns:
            self.encoder[col].weight.data.normal_(mean=0.0, std=init_std)

        for target_column in self.target_columns:
            self.decoder[target_column].bias.data.zero_()
            self.decoder[target_column].weight.data.normal_(mean=0.0, std=init_std)

        for col_name in self.pos_encoder:
            self.pos_encoder[col_name].weight.data.normal_(mean=0.0, std=init_std)

    @beartype
    def _recursive_concat(self, srcs: list[Tensor]):
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

    @beartype
    def forward_embed(self, src: dict[str, Tensor]) -> Tensor:
        srcs = []
        for col in self.categorical_columns:
            src_t = self.encoder[col](src[col].T) * math.sqrt(self.embedding_size)
            pos = (
                torch.arange(0, self.seq_length, dtype=torch.long, device=self.device)
                .repeat(src_t.shape[1], 1)
                .T
            )
            src_p = self.pos_encoder[col](pos)

            src_c = self.drop(src_t + src_p)

            srcs.append(src_c)

        for col in self.real_columns:
            if col in self.real_columns_direct:
                src_t = src[col].T.unsqueeze(2).repeat(1, 1, 1) * math.sqrt(
                    self.embedding_size
                )
            else:
                assert col in self.real_columns_with_embedding
                src_t = self.encoder[col](src[col].T[:, :, None]) * math.sqrt(
                    self.embedding_size
                )

            pos = (
                torch.arange(0, self.seq_length, dtype=torch.long, device=self.device)
                .repeat(src_t.shape[1], 1)
                .T
            )

            src_p = self.pos_encoder[col](pos)

            src_c = self.drop(src_t + src_p)

            srcs.append(src_c)

        src2 = self._recursive_concat(srcs)

        output = self.transformer_encoder(src2, self.src_mask)

        return output

    @beartype
    def forward_train(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.forward_embed(src)
        output = {
            target_column: self.decode(target_column, output)
            for target_column in self.target_columns
        }

        return output

    @beartype
    def decode(self, target_column: str, output: Tensor) -> Tensor:
        decoded = self.decoder[target_column](output)
        return decoded

    @beartype
    def apply_softmax(self, target_column: str, output: Tensor) -> Tensor:
        if target_column in self.real_columns:
            return output
        else:
            return self.softmax[target_column](output)

    @beartype
    def forward(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.forward_train(src)
        return {
            target_column: self.apply_softmax(target_column, out[-1, :, :])
            for target_column, out in output.items()
        }

    @beartype
    def train_model(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_sampler: Optional[
            Union[RandomSampler, DistributedSampler, DistributedGroupedRandomSampler]
        ],
        valid_sampler: Optional[
            Union[RandomSampler, DistributedSampler, DistributedGroupedRandomSampler]
        ],
    ) -> None:
        best_val_loss = float("inf")
        n_epochs_no_improvement = 0

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

                if train_sampler and not isinstance(train_sampler, RandomSampler):
                    train_sampler.set_epoch(epoch)
                self._train_epoch(train_loader, epoch)

                if valid_sampler and not isinstance(valid_sampler, RandomSampler):
                    valid_sampler.set_epoch(epoch)

                total_loss, total_losses, output = self._evaluate(valid_loader)
                elapsed = time.time() - epoch_start_time

                self._log_epoch_results(
                    epoch, elapsed, total_loss, total_losses, output
                )

                if total_loss < best_val_loss:
                    best_val_loss = total_loss
                    best_model = self._copy_model()
                    n_epochs_no_improvement = 0
                else:
                    n_epochs_no_improvement += 1

                self.scheduler.step()
                if epoch % self.iter_save == 0:
                    self._save(epoch, total_loss)

                last_epoch = epoch

        self._export(self, "last", last_epoch)  # type: ignore
        self._export(best_model, "best", last_epoch)  # type: ignore
        self.log_file.write("--- Training Complete ---")
        self.log_file.close()

    @beartype
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> None:
        self.train()

        total_loss = 0.0
        start_time = time.time()
        num_batches = len(train_loader)

        for batch_count, (data, targets) in enumerate(train_loader):
            data = {
                k: v.to(self.device, non_blocking=True)
                for k, v in data.items()
                if k in self.selected_columns
            }
            targets = {
                k: v.to(self.device, non_blocking=True)
                for k, v in targets.items()
                if k in self.target_column_types
            }
            output = self.forward_train(data)

            loss, losses = self._calculate_loss(output, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            if (
                self.accumulation_steps is None
                or (batch_count + 1) % self.accumulation_steps == 0
                or (batch_count + 1) == num_batches
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_count + 1) % self.log_interval == 0 and self.rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                s_per_batch = (time.time() - start_time) / self.log_interval
                self.log_file.write(
                    f"[INFO] Epoch {epoch:3d} | Batch {(batch_count+1):5d}/{num_batches:5d} | Loss: {format_number(total_loss)} | LR: {format_number(lr)} | S/Batch {format_number(s_per_batch)}"
                )
                total_loss = 0.0
                start_time = time.time()

            del data, targets, output, loss, losses

    @beartype
    def _calculate_loss(
        self, output: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses = {}
        for target_column, target_column_type in self.target_column_types.items():
            if target_column_type == "categorical":
                output[target_column] = output[target_column].reshape(
                    -1, self.n_classes[target_column]
                )
            elif target_column_type == "real":
                output[target_column] = output[target_column].reshape(-1)

            losses[target_column] = self.criterion[target_column](
                output[target_column], targets[target_column].T.contiguous().reshape(-1)
            )
        loss = None
        for target_column in self.target_columns:
            losses[target_column] = losses[target_column] * (
                self.loss_weights[target_column]
                if self.loss_weights is not None
                else 1.0
            )
            if loss is None:
                loss = losses[target_column].clone()
            else:
                loss += losses[target_column]

        assert loss is not None

        return loss, losses

    @beartype
    def _copy_model(self):
        log_file = self.log_file
        del self.log_file
        model_copy = copy.deepcopy(self)
        self.log_file = log_file
        return model_copy

    @beartype
    def _transform_val(self, col: str, val: Tensor) -> Tensor:
        if self.target_column_types[col] == "categorical":
            return (
                one_hot(val, self.n_classes[col])
                .reshape(-1, self.n_classes[col])
                .float()
            )
        else:
            assert self.target_column_types[col] == "real"
            return val

    @beartype
    def _evaluate(
        self, valid_loader: DataLoader
    ) -> tuple[np.float32, dict[str, np.float32], dict[str, Tensor]]:
        self.eval()  # Turn on evaluation mode

        total_loss_collect = []
        # Initialize a dict to hold lists of losses for each target
        total_losses_collect = {col: [] for col in self.target_columns}
        output = {}  # for type checking
        with torch.no_grad():
            for i, (data, targets) in enumerate(valid_loader):
                # Move data to the current process's assigned GPU
                data = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in data.items()
                    if k in self.selected_columns
                }
                targets = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in targets.items()
                    if k in self.target_column_types
                }

                output = self.forward_train(data)
                loss, losses = self._calculate_loss(output, targets)

                total_loss_collect.append(loss.item())
                for col, loss in losses.items():
                    total_losses_collect[col].append(loss.item())

                # Free up GPU memory
                del data, targets, loss, losses
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # 1. Sum the losses calculated on this GPU process
        total_loss_local = np.sum(total_loss_collect)
        total_losses_local = {
            col: np.sum(loss_list) for col, loss_list in total_losses_collect.items()
        }

        # 2. Aggregate losses across all GPUs if in distributed mode
        if self.hparams.training_spec.distributed:
            # Put local losses into tensors for reduction
            total_loss_tensor = torch.tensor(total_loss_local, device=self.device)

            # Ensure consistent order for the losses tensor
            loss_keys = sorted(total_losses_local.keys())
            losses_values = [total_losses_local[k] for k in loss_keys]
            losses_tensor = torch.tensor(losses_values, device=self.device)

            # Sum losses from all processes. The result is broadcast back to all processes.
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)

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
            for data, targets in valid_loader:
                data = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in data.items()
                    if k in self.selected_columns
                }
                targets = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in targets.items()
                    if k in self.target_column_types
                }
                # Replicate original logic of using input as pseudo-output
                pseudo_output = {
                    col: self._transform_val(col, data[col]) for col in targets.keys()
                }

                loss, losses = self._calculate_loss(pseudo_output, targets)

                baseline_loss_local_collect.append(loss.item())
                for col, loss_ in losses.items():
                    baseline_losses_local_collect[col].append(loss_.item())

            # Sum the losses for the local shard
            baseline_loss_local = np.sum(baseline_loss_local_collect)
            baseline_losses_local = {
                col: np.sum(loss_list)
                for col, loss_list in baseline_losses_local_collect.items()
            }

            # Broadcast the baseline values from the main process to all others
            if self.hparams.training_spec.distributed:
                total_loss_tensor = torch.tensor(
                    baseline_loss_local, device=self.device
                )
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                self.baseline_loss = total_loss_tensor.item()

                loss_keys = sorted(baseline_losses_local.keys())
                losses_values = [baseline_losses_local[k] for k in loss_keys]
                losses_tensor = torch.tensor(losses_values, device=self.device)
                dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)

                self.baseline_losses = dict(zip(loss_keys, losses_tensor.cpu().numpy()))
            else:
                # If not distributed, local is global
                self.baseline_loss = baseline_loss_local
                self.baseline_losses = baseline_losses_local

        return (
            np.float32(total_loss_global),
            {k: np.float32(v) for k, v in total_losses_global.items()},
            output,
        )

    @beartype
    def _get_batch(
        self,
        X: dict[str, Tensor],
        y: dict[str, Tensor],
        batch_start: int,
        batch_size: int,
        to_device: bool,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if to_device:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :].to(
                        self.device, non_blocking=True
                    )
                    for col in X.keys()
                },
                {
                    target_column: y[target_column][
                        batch_start : batch_start + batch_size, :
                    ].to(self.device, non_blocking=True)
                    for target_column in y.keys()
                },
            )
        else:
            return (
                {
                    col: X[col][batch_start : batch_start + batch_size, :]
                    for col in X.keys()
                },
                {
                    target_column: y[target_column][
                        batch_start : batch_start + batch_size, :
                    ]
                    for target_column in y.keys()
                },
            )

    @beartype
    def _export(self, model: "TransformerModel", suffix: str, epoch: int) -> None:
        if self.rank != 0:
            return

        self.eval()

        os.makedirs(os.path.join(self.project_path, "models"), exist_ok=True)

        if self.export_generative_model:
            self._export_model(model, suffix, epoch)
        if self.export_embedding_model:
            model2 = TransformerEmbeddingModel(model)
            suffix = f"{suffix}-embedding"
            self._export_model(model2, suffix, epoch)

    def _export_model(
        self,
        model: Union["TransformerModel", "TransformerEmbeddingModel"],
        suffix: str,
        epoch: int,
    ) -> None:
        if self.export_onnx:
            x_cat = {
                col: torch.randint(
                    0,
                    self.n_classes[col],
                    (self.inference_batch_size, self.seq_length),
                ).to(self.device, non_blocking=True)
                for col in self.categorical_columns
            }
            x_real = {
                col: torch.rand(self.inference_batch_size, self.seq_length).to(
                    self.device, non_blocking=True
                )
                for col in self.real_columns
            }

            x = {"src": {**x_cat, **x_real}}

            # Export the model
            export_path = os.path.join(
                self.project_path,
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
                model,  # model being run
                x,  # model input (or a tuple for multiple inputs)
                export_path,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=14,  # the ONNX version to export the model to
                do_constant_folding=constant_folding,  # whether to execute constant folding for optimization
                input_names=["input"],  # the model's input names
                output_names=["output"],  # the model's output names
                dynamic_axes={
                    "input": {0: "batch_size"},  # variable length axes
                    "output": {0: "batch_size"},
                },
                training=training_mode,
            )
        if self.export_pt:
            export_path = os.path.join(
                self.project_path,
                "models",
                f"sequifier-{self.model_name}-{suffix}-{epoch}.pt",
            )
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "export_with_dropout": self.export_with_dropout,
                },
                export_path,
            )

    @beartype
    def _save(self, epoch: int, val_loss: np.float32) -> None:
        if self.rank != 0:
            return
        os.makedirs(os.path.join(self.project_path, "checkpoints"), exist_ok=True)

        output_path = os.path.join(
            self.project_path,
            "checkpoints",
            f"{self.model_name}-epoch-{epoch}.pt",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": val_loss,
            },
            output_path,
        )
        if self.rank == 0:
            self.log_file.write(f"[INFO] Saved model to {output_path}")

    @beartype
    def _get_optimizer(self, **kwargs):
        optimizer_class = get_optimizer_class(self.hparams.training_spec.optimizer.name)
        return optimizer_class(
            self.parameters(), lr=self.hparams.training_spec.lr, **kwargs
        )

    @beartype
    def _get_scheduler(self, **kwargs):
        scheduler_class = eval(
            f"torch.optim.lr_scheduler.{self.hparams.training_spec.scheduler.name}"
        )
        return scheduler_class(self.optimizer, **kwargs)

    @beartype
    def _initialize_log_file(self):
        os.makedirs(os.path.join(self.project_path, "logs"), exist_ok=True)
        open_mode = "w" if self.start_epoch == 1 else "a"
        path = os.path.join(
            self.project_path, "logs", f"sequifier-{self.model_name}-[NUMBER].txt"
        )
        if self.rank is not None:
            path = path.replace("[NUMBER]", f"rank{self.rank}-[NUMBER]")
        self.log_file = LogFile(path, open_mode, self.rank)

    @beartype
    def _load_weights_conditional(self) -> str:
        latest_model_path = self._get_latest_model_name()
        pytorch_total_params = sum(p.numel() for p in self.parameters())

        if latest_model_path is not None and self.continue_training:
            checkpoint = torch.load(
                latest_model_path,
                map_location=torch.device(self.device),
                weights_only=False,
            )
            self.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if k == "step":
                            # Keep the 'step' tensor on the CPU.
                            state[k] = v.cpu()
                        else:
                            # Move all other state tensors to the model's device.
                            state[k] = v.to(self.device)

            return f"[INFO] Resuming training from checkpoint '{latest_model_path}'. Total params: {format_number(pytorch_total_params)}"
        else:
            self.start_epoch = 1
            return f"[INFO] Initializing new model with {format_number(pytorch_total_params)} parameters."

    @beartype
    def _get_latest_model_name(self) -> Optional[str]:
        checkpoint_path = os.path.join(self.project_path, "checkpoints", "*")

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
        elapsed: float,
        total_loss: np.float32,
        total_losses: dict[str, np.float32],
        output: dict[str, Tensor],
    ) -> None:
        if self.rank == 0:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

            self.log_file.write("-" * 89)
            self.log_file.write(
                f"[INFO] Validation | Epoch: {epoch:3d} | Loss: {format_number(total_loss)} | Baseline Loss: {format_number(self.baseline_loss)} | Time: {elapsed:5.2f}s | LR {format_number(lr)}"
            )

            if len(total_losses) > 1:
                loss_strs = [
                    f"{key}_loss: {format_number(value)}"
                    for key, value in total_losses.items()
                ]
                self.log_file.write("[INFO]  - " + ", ".join(loss_strs))

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
                self.log_file.write(f"[INFO] {categorical_column}: {value_shares}")

            self.log_file.write("-" * 89)


@beartype
def load_inference_model(
    model_path: str,
    training_config_path: str,
    args_config: dict[str, Any],
    device: str,
    infer_with_dropout: bool,
) -> torch.nn.Module:
    training_config = load_train_config(
        training_config_path, args_config, args_config["on_unprocessed"]
    )

    with torch.no_grad():
        model = TransformerModel(training_config)
        model.log_file.write(f"[INFO] Loading model weights from {model_path}")
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

        model = torch.compile(model).to(device)

    return model


@beartype
def infer_with_model(
    model: torch._dynamo.eval_frame.OptimizedModule,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
) -> dict[str, np.ndarray]:
    outs0 = []
    with torch.no_grad():
        for x_sub in x:
            data_gpu = {
                col: torch.from_numpy(x_).to(device) for col, x_ in x_sub.items()
            }
            output_gpu = model.forward(data_gpu)
            output_cpu = {k: v.cpu().detach() for k, v in output_gpu.items()}
            outs0.append(output_cpu)
            if device == "cuda":
                torch.cuda.empty_cache()

    outs = {
        target_column: np.concatenate(
            [o[target_column].numpy() for o in outs0],
            axis=0,
        )[:size, :]
        for target_column in target_columns
    }

    return outs
