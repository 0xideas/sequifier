from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterator, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import ModuleDict

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.tracing import TraceContext


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
        hidden_block_end_indices: list[int] = []
        layer_input_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(layer_input_dim, hidden_dim))
            layers.append(self._activation(self.activation_fn))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            hidden_block_end_indices.append(len(layers) - 1)
            layer_input_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.hidden_block_end_indices = tuple(hidden_block_end_indices)

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
        hidden, _ = self._project_hidden_with_activations(x, ())
        return hidden

    def _project_hidden_with_activations(
        self, x: Tensor, block_indices: tuple[int, ...]
    ) -> tuple[Tensor, dict[int, Tensor]]:
        hidden = x
        activations: dict[int, Tensor] = {}
        selected_indices = set(block_indices)
        block_index_by_end = {
            module_index: block_index
            for block_index, module_index in enumerate(self.hidden_block_end_indices)
            if block_index in selected_indices
        }
        for module_index, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                hidden = layer(cast_floating_to_module_dtype(hidden, layer))
            else:
                hidden = layer(hidden)
            block_index = block_index_by_end.get(module_index)
            if block_index is not None:
                activations[block_index] = hidden
        return hidden, activations

    def project_hidden_with_activations(
        self, x: Tensor, block_indices: tuple[int, ...]
    ) -> dict[int, Tensor]:
        """Return selected post-activation/dropout MLP block outputs."""
        _, activations = self._project_hidden_with_activations(x, block_indices)
        return activations

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

    def forward(
        self,
        x: Tensor,
        *,
        trace: TraceContext | None = None,
        branch_name: str = "default",
    ) -> dict[str, Tensor]:
        hidden = x
        block_index_by_end = {
            module_index: block_index
            for block_index, module_index in enumerate(self.hidden_block_end_indices)
        }
        for module_index, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                hidden = layer(cast_floating_to_module_dtype(hidden, layer))
            else:
                hidden = layer(hidden)
            block_index = block_index_by_end.get(module_index)
            if trace is not None and block_index is not None:
                hidden = trace.emit(
                    f"decoder.branch.{branch_name}.block.{block_index}",
                    hidden,
                    axes=("batch", "time", "channel"),
                    width=hidden.shape[-1],
                )
        outputs = {}
        for target_column in self.target_columns:
            output_layer = cast(nn.Linear, self.output_layers[target_column])
            output = output_layer(
                cast_floating_to_module_dtype(hidden, output_layer)
            ).to(torch.float32)
            if trace is not None:
                output = trace.emit(
                    f"decoder.branch.{branch_name}.logits.{target_column}",
                    output,
                    axes=("batch", "time", "channel"),
                    width=output.shape[-1],
                )
            outputs[target_column] = output
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

    def hidden_block_activations(
        self,
        x: Tensor,
        block_indices_by_branch: dict[str, tuple[int, ...]],
    ) -> dict[tuple[str, int], Tensor]:
        """Return selected logical MLP hidden-block activations by branch."""
        activations: dict[tuple[str, int], Tensor] = {}
        for branch_name, block_indices in block_indices_by_branch.items():
            branch = cast(TargetDecoderBranch, self.branches[branch_name])
            branch_activations = branch.project_hidden_with_activations(
                x, block_indices
            )
            activations.update(
                {
                    (branch_name, block_index): activation
                    for block_index, activation in branch_activations.items()
                }
            )
        return activations

    def forward(
        self, x: Tensor, *, trace: TraceContext | None = None
    ) -> dict[str, Tensor]:
        branch_outputs = {
            branch_name: cast(TargetDecoderBranch, branch)(
                x, trace=trace, branch_name=branch_name
            )
            for branch_name, branch in self.branches.items()
        }
        return {
            target_column: branch_outputs[self.target_to_branch[target_column]][
                target_column
            ]
            for target_column in self.target_columns
        }


@dataclass(frozen=True)
class ResolvedDecoderBranch:
    name: str
    config: Any
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class DecodingPlan:
    """Canonical internal form for single and named decoding specs."""

    branches: tuple[ResolvedDecoderBranch, ...]
    target_to_branch: dict[str, str]


DECODER_HIDDEN_DIMS: dict[str, Callable[[Any], list[int]]] = {
    "linear": lambda config: [],
    "mlp": lambda config: list(config.hidden_dims),
}


def resolve_decoding_plan(hparams: Any) -> DecodingPlan:
    decoding_spec = hparams.model_spec.decoder

    if decoding_spec.type == "composite":
        branch_items = list(decoding_spec.branches.items())
        default_target_columns = None
    else:
        branch_items = [("default", decoding_spec)]
        default_target_columns = hparams.target_columns

    branches = []
    target_to_branch = {}
    for branch_name, branch_config in branch_items:
        if branch_config.type not in DECODER_HIDDEN_DIMS:
            raise ValueError(f"Unknown target decoder type: {branch_config.type}")
        target_columns = branch_config.target_columns
        if target_columns is None:
            if default_target_columns is None:
                raise ValueError(
                    f"Target decoding branch {branch_name!r} must configure "
                    "target_columns."
                )
            target_columns = default_target_columns

        missing_columns = set(target_columns) - set(hparams.target_columns)
        if missing_columns:
            raise ValueError(
                f"Target decoding branch {branch_name!r} references unknown "
                f"target_columns: {sorted(missing_columns)}"
            )
        for target_column in target_columns:
            if target_column in target_to_branch:
                raise ValueError(
                    "Target decoding branches cannot share target columns: "
                    f"{target_column!r} appears in both "
                    f"{target_to_branch[target_column]!r} and {branch_name!r}."
                )
            target_to_branch[target_column] = branch_name
        branches.append(
            ResolvedDecoderBranch(
                name=branch_name,
                config=branch_config,
                target_columns=tuple(target_columns),
            )
        )

    undecoded_columns = set(hparams.target_columns) - set(target_to_branch)
    if undecoded_columns:
        raise ValueError(
            "model_spec.decoder must decode every target column; "
            f"missing {sorted(undecoded_columns)}"
        )
    return DecodingPlan(tuple(branches), target_to_branch)


def build_target_decoding(
    hparams: Any,
    target_n_classes: Optional[dict[str, int]] = None,
) -> TargetDecoding:
    model_spec = hparams.model_spec
    plan = resolve_decoding_plan(hparams)

    decoder_n_classes = (
        hparams.n_classes if target_n_classes is None else target_n_classes
    )
    input_dim = model_spec.backbone.architecture.dim_model * model_spec.decoder.support

    branches = {}
    for branch in plan.branches:
        branch_config = branch.config
        branches[branch.name] = TargetDecoderBranch(
            target_columns=list(branch.target_columns),
            target_column_types=hparams.target_column_types,
            n_classes=decoder_n_classes,
            input_dim=input_dim,
            hidden_dims=DECODER_HIDDEN_DIMS[branch_config.type](branch_config),
            activation_fn=getattr(branch_config, "activation_fn", "relu"),
            dropout=getattr(branch_config, "dropout", 0.0),
            hidden_weight_l2=getattr(branch_config, "hidden_weight_l2", 0.0),
        )

    return TargetDecoding(
        branches=branches,
        target_columns=hparams.target_columns,
        target_to_branch=plan.target_to_branch,
    )
