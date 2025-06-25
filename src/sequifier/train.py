import copy
import glob
import math
import os
import time
import uuid
import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch._dynamo
from beartype import beartype
from torch import Tensor, nn
from torch.nn import ModuleDict, TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import one_hot

torch._dynamo.config.suppress_errors = True
from sequifier.config.train_config import load_train_config  # noqa: E402
from sequifier.helpers import PANDAS_TO_TORCH_TYPES  # noqa: E402
from sequifier.helpers import LogFile  # noqa: E402
from sequifier.helpers import construct_index_maps  # noqa: E402
from sequifier.helpers import normalize_path  # noqa: E402
from sequifier.helpers import read_data  # noqa: E402
from sequifier.helpers import numpy_to_pytorch, subset_to_selected_columns  # noqa: E402
from sequifier.optimizers.optimizers import get_optimizer_class  # noqa: E402


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    """
    Train the model using the provided configuration.

    Args:
        args: Command line arguments.
        args_config: Configuration dictionary.
    """
    config_path = args.config_path or "configs/train.yaml"
    config = load_train_config(config_path, args_config, args.on_unprocessed)

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    data_train = read_data(
        normalize_path(config.training_data_path, config.project_path),
        config.read_format,
    )
    if config.selected_columns is not None:
        data_train = subset_to_selected_columns(data_train, config.selected_columns)

    X_train, y_train = numpy_to_pytorch(
        data_train,
        column_types,
        config.selected_columns,
        config.target_columns,
        config.seq_length,
        config.training_spec.device,
        to_device=False,
    )
    del data_train

    data_valid = read_data(
        normalize_path(config.validation_data_path, config.project_path),
        config.read_format,
    )
    if config.selected_columns is not None:
        data_valid = subset_to_selected_columns(data_valid, config.selected_columns)

    X_valid, y_valid = numpy_to_pytorch(
        data_valid,
        column_types,
        config.selected_columns,
        config.target_columns,
        config.seq_length,
        config.training_spec.device,
        to_device=False,
    )
    del data_valid

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = torch.compile(TransformerModel(config).to(config.training_spec.device))

    model.train_model(X_train, y_train, X_valid, y_valid)


@beartype
def format_number(number: Union[int, float, np.float32]) -> str:
    """
    Format a number for display.

    Args:
        number: The number to format.

    Returns:
        A formatted string representation of the number.
    """
    if pd.isnull(number):
        return "NaN"
    elif number == 0:
        order_of_magnitude = 0
    else:
        order_of_magnitude = math.floor(math.log(number, 10))

    number_adjusted = number * (10 ** (-order_of_magnitude))
    return f"{number_adjusted:5.2f}e{order_of_magnitude}"


class TransformerModel(nn.Module):
    @beartype
    def __init__(self, hparams: Any):
        super().__init__()
        self.project_path = hparams.project_path
        self.model_type = "Transformer"
        self.model_name = hparams.model_name or uuid.uuid4().hex[:8]

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
    def forward_train(self, src: dict[str, Tensor]) -> dict[str, Tensor]:
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

        src2 = torch.cat(srcs, 2)

        output = self.transformer_encoder(src2, self.src_mask)
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
        X_train: dict[str, Tensor],
        y_train: dict[str, Tensor],
        X_valid: dict[str, Tensor],
        y_valid: dict[str, Tensor],
    ) -> None:
        best_val_loss = float("inf")
        n_epochs_no_improvement = 0

        for epoch in range(self.start_epoch, self.hparams.training_spec.epochs + 1):
            if (
                self.early_stopping_epochs is None
                or n_epochs_no_improvement < self.early_stopping_epochs
                or (epoch > self.start_epoch and not np.isnan(total_loss))  # type: ignore # noqa: F821
            ):
                epoch_start_time = time.time()
                self._train_epoch(X_train, y_train, epoch)
                total_loss, total_losses, output = self._evaluate(X_valid, y_valid)
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
        self.log_file.write("Training transformer complete")
        self.log_file.close()

    @beartype
    def _train_epoch(
        self, X_train: dict[str, Tensor], y_train: dict[str, Tensor], epoch: int
    ) -> None:
        self.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()

        num_batches = math.ceil(
            X_train[self.target_columns[0]].shape[0] / self.batch_size
        )  # any column will do
        batch_order = list(
            np.random.choice(
                np.arange(num_batches), size=num_batches, replace=False
            ).flatten()
        )
        for batch_count, batch in enumerate(batch_order):
            batch_start = int(batch * self.batch_size)

            data, targets = self._get_batch(
                X_train, y_train, batch_start, self.batch_size, to_device=True
            )
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
            if (batch_count + 1) % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                s_per_batch = (time.time() - start_time) / self.log_interval
                self.log_file.write(
                    f"| epoch {epoch:3d} | {(batch_count+1):5d}/{num_batches:5d} batches | "
                    f"lr {format_number(lr)} | s/batch {format_number(s_per_batch)} | "
                    f"loss {format_number(total_loss)}"
                )
                total_loss = 0.0
                start_time = time.time()

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
        self, X_valid: dict[str, Tensor], y_valid: dict[str, Tensor]
    ) -> tuple[np.float32, dict[str, np.float32], dict[str, Tensor]]:
        self.eval()  # turn on evaluation mode

        with torch.no_grad():
            num_batches = math.ceil(
                X_valid[self.target_columns[0]].shape[0] / self.batch_size
            )  # any column will do
            total_loss_collect, total_losses_collect = [], []
            for batch_start in range(0, num_batches * self.batch_size, self.batch_size):
                data, targets = self._get_batch(
                    X_valid,
                    y_valid,
                    batch_start,
                    batch_start + self.batch_size,
                    to_device=True,
                )
                output = self.forward_train(data)
                total_loss_iter, total_losses_iter = self._calculate_loss(
                    output, targets
                )
                total_loss_collect.append(total_loss_iter.cpu())
                total_losses_collect.append(total_losses_iter)

                torch.cuda.empty_cache()

        total_loss = np.sum(total_loss_collect)
        total_losses = {
            target_column: np.sum(
                [
                    total_losses_i[target_column].cpu()
                    for total_losses_i in total_losses_collect
                ]
            )
            for target_column in total_losses_iter.keys()  # type: ignore
        }
        if not hasattr(self, "baseline_loss"):
            data, targets = self._get_batch(
                X_valid, y_valid, 0, list(X_valid.values())[0].shape[0], to_device=False
            )
            self.baseline_loss, self.baseline_losses = self._calculate_loss(
                {
                    col: self._transform_val(col, data[col]) for col in targets.keys()
                },  # this variant is chosen because the same batch might have several "sequenceId" sequences
                {col: val for col, val in targets.items()},
            )
            self.baseline_loss = self.baseline_loss.item()
            self.baseline_losses = {
                target_column: target_loss.item()
                for target_column, target_loss in self.baseline_losses.items()
            }

        return total_loss, total_losses, output  # type: ignore

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
                        self.device
                    )
                    for col in X.keys()
                },
                {
                    target_column: y[target_column][
                        batch_start : batch_start + batch_size, :
                    ].to(self.device)
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
        self.eval()

        os.makedirs(os.path.join(self.project_path, "models"), exist_ok=True)
        if self.export_onnx:
            x_cat = {
                col: torch.randint(
                    0,
                    self.n_classes[col],
                    (self.inference_batch_size, self.seq_length),
                ).to(self.device)
                for col in self.categorical_columns
            }
            x_real = {
                col: torch.rand(self.inference_batch_size, self.seq_length).to(
                    self.device
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
        self.log_file.write(f"Saved model to {output_path}")

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
        self.log_file = LogFile(
            os.path.join(
                self.project_path, "logs", f"sequifier-{self.model_name}-[NUMBER].txt"
            ),
            open_mode,
        )

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
            return f"Loading model weights from {latest_model_path}. Total params: {format_number(pytorch_total_params)}"
        else:
            self.start_epoch = 1
            return f"Initializing new model with {format_number(pytorch_total_params)} params"

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
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

        self.log_file.write("-" * 89)
        self.log_file.write(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | lr: {lr} | "
            f"valid loss {format_number(total_loss)} | baseline loss {format_number(self.baseline_loss)}"
        )

        if len(total_losses) > 1:
            self.log_file.write(
                ", ".join(
                    [
                        f"'{target_column} loss': {format_number(tloss)}"
                        for target_column, tloss in total_losses.items()
                    ]
                ),
                level=2,
            )
            self.log_file.write(
                ", ".join(
                    [
                        f"'{target_column} baseline loss': {format_number(bloss)}"
                        for target_column, bloss in self.baseline_losses.items()
                    ]
                ),
                level=2,
            )

        for categorical_column in self.class_share_log_columns:
            output_values = output[categorical_column].argmax(1).cpu().detach().numpy()
            output_counts = pd.Series(output_values).value_counts().sort_index()
            output_counts = output_counts / output_counts.sum()
            value_shares = " | ".join(
                [
                    f"{self.index_maps[categorical_column][value]}: {share:5.5f}"
                    for value, share in output_counts.to_dict().items()
                ]
            )
            self.log_file.write(f"{categorical_column}: {value_shares}")

        self.log_file.write("-" * 89)


@beartype
def load_inference_model(
    model_path: str,
    training_config_path: str,
    args_config: dict[str, Any],
    device: str,
    infer_with_dropout: bool,
) -> torch._dynamo.eval_frame.OptimizedModule:
    training_config = load_train_config(
        training_config_path, args_config, args_config["on_unprocessed"]
    )

    with torch.no_grad():
        model = TransformerModel(training_config)
        model.log_file.write(f"Loading model weights from {model_path}")
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
    outs0 = [
        model.forward(
            {col: torch.from_numpy(x_).to(device) for col, x_ in x_sub.items()}
        )
        for x_sub in x
    ]
    outs = {
        target_column: np.concatenate(
            [o[target_column].cpu().detach().numpy() for o in outs0],
            axis=0,
        )[:size, :]
        for target_column in target_columns
    }

    return outs
