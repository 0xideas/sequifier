import copy
import glob
import math
import os
import re
import time
import uuid
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import ModuleDict, TransformerEncoder, TransformerEncoderLayer

from sequifier.config.train_config import load_transformer_config
from sequifier.helpers import PANDAS_TO_TORCH_TYPES, LogFile, numpy_to_pytorch


class TransformerModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.project_path = hparams.project_path
        self.target_column = hparams.target_column
        self.target_column_type = hparams.target_column_type
        self.model_name = (
            hparams.model_name
            if hparams.model_name is not None
            else uuid.uuid4().hex[:8]
        )
        self.hparams = hparams
        self.real_columns_repetitions = self.get_real_columns_repetitions(
            hparams.real_columns, hparams.model_spec.nhead
        )
        self.model_type = "Transformer"
        self.log_interval = hparams.log_interval
        self.encoder = ModuleDict()
        self.pos_encoder = ModuleDict()
        for col, n_classes in hparams.n_classes.items():
            if col in hparams.categorical_columns:
                self.encoder[col] = nn.Embedding(n_classes, hparams.model_spec.d_model)
                self.pos_encoder[col] = PositionalEncoding(
                    hparams.model_spec.d_model, hparams.training_spec.dropout
                )

        embedding_size = (
            hparams.model_spec.d_model * len(hparams.categorical_columns)
        ) + int(np.sum(list(self.real_columns_repetitions.values())))

        encoder_layers = TransformerEncoderLayer(
            embedding_size,
            hparams.model_spec.nhead,
            hparams.model_spec.d_hid,
            hparams.training_spec.dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, hparams.model_spec.nlayers, enable_nested_tensor=False
        )

        if self.target_column_type == "categorical":
            self.decoder = nn.Linear(
                embedding_size * hparams.seq_length,
                hparams.n_classes[self.target_column],
            )
        elif self.target_column_type == "real":
            self.decoder = nn.Linear(embedding_size * hparams.seq_length, 1)
        else:
            raise Exception(
                f"{self.target_column_type = } not in ['categorical', 'real']"
            )

        self.criterion = eval(f"torch.nn.{hparams.training_spec.criterion}()")
        self.batch_size = hparams.training_spec.batch_size
        self.device = hparams.training_spec.device

        self.src_mask = self.generate_square_subsequent_mask(
            self.hparams.seq_length
        ).to(self.device)

        self.init_weights()
        self.optimizer = self.get_optimizer(
            **self.filter_key(hparams.training_spec.optimizer, "name")
        )
        self.scheduler = self.get_scheduler(
            **self.filter_key(hparams.training_spec.scheduler, "name")
        )

        self.iter_save = hparams.training_spec.iter_save
        self.continue_training = hparams.training_spec.continue_training
        self.load_weights_conditional()

        os.makedirs(os.path.join(self.project_path, "logs"), exist_ok=True)
        open_mode = "w" if self.start_epoch == 1 else "a"
        self.log_file = LogFile(
            os.path.join(self.project_path, "logs", f"sequifier-{self.model_name}.txt"),
            open_mode,
        )

    def get_real_columns_repetitions(self, real_columns, nhead):
        real_columns_repetitions = {col: 1 for col in real_columns}
        column_index = dict(enumerate(real_columns))
        for i in range(nhead * len(real_columns)):
            if np.sum(list(real_columns_repetitions.values())) % nhead != 0:
                j = i % len(real_columns)
                real_columns_repetitions[column_index[j]] += 1
        assert np.sum(list(real_columns_repetitions.values())) % nhead == 0

        return real_columns_repetitions

    def filter_key(self, dict_, key):
        return {k: v for k, v in dict_.items() if k != key}

    def get_optimizer(self, **kwargs):
        optimizer_class = eval(
            f"torch.optim.{self.hparams.training_spec.optimizer.name}"
        )
        return optimizer_class(
            self.parameters(), lr=self.hparams.training_spec.lr, **kwargs
        )

    def get_scheduler(self, **kwargs):
        scheduler_class = eval(
            f"torch.optim.lr_scheduler.{self.hparams.training_spec.scheduler.name}"
        )
        return scheduler_class(self.optimizer, **kwargs)

    def init_weights(self) -> None:
        initrange = 0.1
        for col in self.hparams.categorical_columns:
            self.encoder[col].weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: dict[str, Tensor]) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [batch_size, n_classes]
        """

        srcs = []
        for col in self.hparams.categorical_columns:
            src_t = self.encoder[col](src[col].T) * math.sqrt(
                self.hparams.model_spec.d_model
            )
            src_t = self.pos_encoder[col](src_t)
            srcs.append(src_t)

        for col in self.hparams.real_columns:
            srcs.append(
                src[col].T.unsqueeze(2).repeat(1, 1, self.real_columns_repetitions[col])
            )

        src = torch.cat(srcs, 2)

        output = self.transformer_encoder(src, self.src_mask)
        transposed = output.transpose(0, 1)
        concatenated = transposed.reshape(
            transposed.size()[0], transposed.size()[1] * transposed.size()[2]
        )
        output = self.decoder(concatenated)
        return output

    def get_batch(
        self,
        X,
        y,
        i,
        batch_size,
    ):
        return (
            {col: X[col][i : i + batch_size, :] for col in X.keys()},
            y[i : i + batch_size],
        )

    def train_epoch(self, X_train, y_train, epoch) -> None:
        self.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()

        num_batches = math.ceil(len(X_train[self.target_column]) / self.batch_size)
        for batch, i in enumerate(
            range(0, X_train[self.target_column].size(0) - 1, self.batch_size)
        ):
            data, targets = self.get_batch(X_train, y_train, i, self.batch_size)
            output = self(data)
            if self.target_column_type == "categorical":
                output = output.view(-1, self.hparams.n_classes[self.target_column])
            elif self.target_column_type == "real":
                output = output.flatten()
            else:
                pass

            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            if batch % self.log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / self.log_interval
                cur_loss_normalized = (
                    1000 * total_loss / (self.log_interval * self.batch_size)
                )
                ppl = math.exp(cur_loss_normalized)
                self.log_file.write(
                    f"| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | "
                    f"lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss_normalized :5.5f} | ppl {ppl:8.2f}"
                )
                total_loss = 0.0
                start_time = time.time()

    def train_model(self, X_train, y_train, X_valid, y_valid):
        best_val_loss = float("inf")
        best_model = None

        for epoch in range(
            self.start_epoch, self.hparams.training_spec.epochs + self.start_epoch
        ):
            epoch_start_time = time.time()
            self.train_epoch(X_train, y_train, epoch)
            val_loss_normalized = 1000 * self.evaluate(X_valid, y_valid)
            val_ppl = math.exp(val_loss_normalized)
            elapsed = time.time() - epoch_start_time
            self.log_file.write("-" * 89)
            self.log_file.write(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {val_loss_normalized:5.5f} | valid ppl {val_ppl:8.2f}"
            )
            self.log_file.write("-" * 89)

            if val_loss_normalized < best_val_loss:
                best_val_loss = val_loss_normalized
                best_model = self.copy_model()

            self.scheduler.step()
            if epoch % self.iter_save == 0:
                self.save(epoch, val_loss_normalized)

        model_name = self.hparams.model_name

        self.export(self, "last")
        self.export(best_model, "best")
        self.log_file.write("Training transformer complete")
        self.log_file.close()

    def copy_model(self):
        log_file = self.log_file
        del self.log_file
        model_copy = copy.deepcopy(self)
        self.log_file = log_file
        return model_copy

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def evaluate(self, X_valid, y_valid) -> float:
        self.eval()  # turn on evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, X_valid[self.target_column].size(0), self.batch_size):
                data, targets = self.get_batch(X_valid, y_valid, i, self.batch_size)
                output = self(data)
                if self.target_column_type == "categorical":
                    output = output.view(-1, self.hparams.n_classes[self.target_column])
                elif self.target_column_type == "real":
                    output = output.flatten()
                else:
                    pass

                total_loss += self.criterion(output, targets).item()

        return total_loss / (X_valid[self.target_column].size(0))

    def export(self, model, suffix):
        self.eval()
        x_cat = {
            col: torch.randint(
                0,
                self.hparams.n_classes[col],
                (self.batch_size, self.hparams.seq_length),
            ).to(self.device)
            for col in self.hparams.categorical_columns
        }
        x_real = {
            col: torch.rand(self.batch_size, self.hparams.seq_length).to(self.device)
            for col in self.hparams.real_columns
        }

        x = {"src": {**x_cat, **x_real}}

        os.makedirs(os.path.join(self.project_path, "models"), exist_ok=True)
        # Export the model
        export_path = os.path.join(
            self.project_path, "models", f"sequifier-{self.model_name}-{suffix}.onnx"
        )
        torch.onnx.export(
            model,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            export_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=14,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )

    def save(self, epoch, val_loss):
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

    def load_weights_conditional(self):

        latest_model_path = self.get_latest_model_name()

        if latest_model_path is not None and self.continue_training:
            self.log_file.write(f"Loading model weights from {latest_model_path}")
            checkpoint = torch.load(latest_model_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = (
                int(re.findall("epoch-([0-9]+)", latest_model_path)[0]) + 1
            )
        else:
            self.start_epoch = 1

    def get_latest_model_name(self):

        checkpoint_path = os.path.join(self.project_path, "checkpoints", "*")

        files = glob.glob(
            checkpoint_path
        )  # * means all if need specific format then *.csv
        files = [
            file for file in files if os.path.split(file)[1].startswith(self.model_name)
        ]
        if len(files):
            return max(files, key=os.path.getctime)
        else:
            return None


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def train(args, args_config):
    config = load_transformer_config(
        args.config_path, args_config, args.on_preprocessed
    )

    column_types = {
        col: PANDAS_TO_TORCH_TYPES[config.column_types[col]]
        for col in config.column_types
    }

    data_train = pd.read_csv(
        config.training_data_path, sep=",", decimal=".", index_col=None
    )
    X_train, y_train = numpy_to_pytorch(
        data_train,
        column_types,
        config.target_column,
        config.seq_length,
        config.training_spec.device,
    )
    # del data_train

    data_valid = pd.read_csv(
        config.validation_data_path, sep=",", decimal=".", index_col=None
    )
    X_valid, y_valid = numpy_to_pytorch(
        data_valid,
        column_types,
        config.target_column,
        config.seq_length,
        config.training_spec.device,
    )
    del data_valid

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = TransformerModel(config).to(config.training_spec.device)

    model.train_model(X_train, y_train, X_valid, y_valid)
