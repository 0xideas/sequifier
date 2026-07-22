from typing import Any, Iterator, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import ModuleDict

from sequifier.model.dtypes import cast_floating_to_module_dtype


def _validate_module_dict_key(key: str, usage: str) -> None:
    if key == "":
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")


class TargetDecoderBranch(nn.Module):
    def __init__(
        self,
        *,
        target_columns: list[str],
        target_column_types: dict[str, str],
        n_classes: dict[str, int],
        input_dim: int,
        hidden_dims: list[int],
        activation_fn: str,
        dropout: float,
        hidden_weight_l2: float = 0.0,
    ):
        super().__init__()
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.hidden_weight_l2 = hidden_weight_l2

        layers: list[nn.Module] = []
        layer_input_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(layer_input_dim, hidden_dim))
            layers.append(self._activation(self.activation_fn))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            layer_input_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

        self.output_layers = ModuleDict()
        for target_column in self.target_columns:
            target_column_type = self.target_column_types[target_column]
            if target_column_type == "categorical":
                output_dim = self.n_classes[target_column]
            elif target_column_type == "real":
                output_dim = 1
            else:
                raise ValueError(
                    f"Target column type {target_column_type} not in "
                    "['categorical', 'real']"
                )
            self.output_layers[target_column] = nn.Linear(layer_input_dim, output_dim)

    @staticmethod
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unknown decoder activation_fn: {name}")

    def _project_hidden(self, x: Tensor) -> Tensor:
        hidden = x
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                hidden = layer(cast_floating_to_module_dtype(hidden, layer))
            else:
                hidden = layer(hidden)
        return hidden

    def decode(self, target_column: str, x: Tensor) -> Tensor:
        hidden = self._project_hidden(x)
        output_layer = cast(nn.Linear, self.output_layers[target_column])
        return output_layer(cast_floating_to_module_dtype(hidden, output_layer)).to(
            torch.float32
        )

    def target_dtype(self, target_column: str) -> torch.dtype:
        return cast(nn.Linear, self.output_layers[target_column]).weight.dtype

    def hidden_weight_parameters(self) -> Iterator[nn.Parameter]:
        """Yield hidden linear kernels, excluding biases and output layers."""
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                yield layer.weight

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = self._project_hidden(x)
        outputs = {}
        for target_column in self.target_columns:
            output_layer = cast(nn.Linear, self.output_layers[target_column])
            outputs[target_column] = output_layer(
                cast_floating_to_module_dtype(hidden, output_layer)
            ).to(torch.float32)
        return outputs


class TargetDecoding(nn.Module):
    def __init__(
        self,
        *,
        branches: dict[str, TargetDecoderBranch],
        target_columns: list[str],
        target_to_branch: dict[str, str],
    ):
        super().__init__()
        for branch_name in branches:
            _validate_module_dict_key(
                branch_name, f"Target decoding branch {branch_name!r}"
            )
        self.branches = ModuleDict(branches)
        self.target_columns = target_columns
        self.target_to_branch = target_to_branch

    def __contains__(self, target_column: object) -> bool:
        return isinstance(target_column, str) and target_column in self.target_to_branch

    def decode(self, target_column: str, x: Tensor) -> Tensor:
        branch = cast(
            TargetDecoderBranch,
            self.branches[self.target_to_branch[target_column]],
        )
        return branch.decode(target_column, x)

    def target_dtype(self, target_column: str) -> torch.dtype:
        branch = cast(
            TargetDecoderBranch,
            self.branches[self.target_to_branch[target_column]],
        )
        return branch.target_dtype(target_column)

    def regularization_loss(self) -> Tensor:
        """Return decoder-scoped L2 for unique hidden linear kernels."""
        loss: Optional[Tensor] = None
        seen_weights: set[int] = set()
        for branch in self.branches.values():
            branch = cast(TargetDecoderBranch, branch)
            if branch.hidden_weight_l2 == 0.0:
                continue
            for weight in branch.hidden_weight_parameters():
                weight_id = id(weight)
                if weight_id in seen_weights:
                    continue
                seen_weights.add(weight_id)
                loss_dtype = (
                    torch.float64 if weight.dtype == torch.float64 else torch.float32
                )
                weight_loss = (
                    weight.to(dtype=loss_dtype).square().sum() * branch.hidden_weight_l2
                )
                loss = weight_loss if loss is None else loss + weight_loss

        if loss is not None:
            return loss

        reference_parameter = next(self.parameters())
        return reference_parameter.new_zeros((), dtype=torch.float32)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        branch_outputs = {
            branch_name: cast(TargetDecoderBranch, branch)(x)
            for branch_name, branch in self.branches.items()
        }
        return {
            target_column: branch_outputs[self.target_to_branch[target_column]][
                target_column
            ]
            for target_column in self.target_columns
        }


def _branch_target_columns(
    branch_config: Any,
    *,
    default_target_columns: Optional[list[str]],
    usage: str,
) -> list[str]:
    target_columns = getattr(branch_config, "target_columns", None)
    if target_columns is not None:
        return target_columns
    if default_target_columns is not None:
        return default_target_columns
    raise ValueError(f"{usage} must configure target_columns")


def _branch_hidden_dims(branch_config: Any) -> list[int]:
    if branch_config.type == "linear":
        return []
    if branch_config.type == "mlp":
        return list(branch_config.hidden_dims)
    raise ValueError(f"Unknown target decoder type: {branch_config.type}")


def build_target_decoding(hparams: Any) -> TargetDecoding:
    model_spec = hparams.model_spec
    decoding_spec = model_spec.decoding_spec
    if decoding_spec is None:
        raise ValueError("decoding_spec must be configured")

    input_dim = model_spec.dim_model * model_spec.decoding_support

    if isinstance(decoding_spec, dict):
        branch_items = list(decoding_spec.items())
        default_target_columns = None
    else:
        branch_items = [("default", decoding_spec)]
        default_target_columns = hparams.target_columns

    branches = {}
    target_to_branch = {}
    for branch_name, branch_config in branch_items:
        target_columns = _branch_target_columns(
            branch_config,
            default_target_columns=default_target_columns,
            usage=f"Target decoding branch {branch_name!r}",
        )
        branches[branch_name] = TargetDecoderBranch(
            target_columns=target_columns,
            target_column_types=hparams.target_column_types,
            n_classes=hparams.n_classes,
            input_dim=input_dim,
            hidden_dims=_branch_hidden_dims(branch_config),
            activation_fn=getattr(branch_config, "activation_fn", "relu"),
            dropout=getattr(branch_config, "dropout", 0.0),
            hidden_weight_l2=getattr(branch_config, "hidden_weight_l2", 0.0),
        )
        for target_column in target_columns:
            target_to_branch[target_column] = branch_name

    return TargetDecoding(
        branches=branches,
        target_columns=hparams.target_columns,
        target_to_branch=target_to_branch,
    )
