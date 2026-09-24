from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import ModuleDict

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.encoder_stack import TransformerEncoderStack
from sequifier.model.tracing import TraceContext
from sequifier.typechecking import beartype, conditional_beartype


@beartype
def _validate_module_dict_key(key: str, usage: str) -> None:
    if key == "":
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")


class TargetDecoderBranch(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        target_columns: list[str],
        target_column_types: dict[str, str],
        n_classes: dict[str, int],
        input_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
        hidden_weight_l2: float = 0.0,
    ):
        super().__init__()
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.hidden_weight_l2 = hidden_weight_l2

        layers: list[nn.Module] = []
        hidden_block_end_indices: list[int] = []
        layer_input_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(layer_input_dim, hidden_dim))
            layers.append(self._activation(self.activation))
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
            output_layer = nn.Linear(layer_input_dim, output_dim)
            output_layer._sequifier_decoder_output = True  # type: ignore[attr-defined]
            self.output_layers[target_column] = output_layer

    @staticmethod
    @conditional_beartype
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unknown decoder activation: {name}")

    @conditional_beartype
    def _project_hidden(self, x: Tensor) -> Tensor:
        hidden, _ = self._project_hidden_with_activations(x, ())
        return hidden

    @conditional_beartype
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

    @conditional_beartype
    def project_hidden_with_activations(
        self, x: Tensor, block_indices: tuple[int, ...]
    ) -> dict[int, Tensor]:
        """Return selected post-activation/dropout MLP block outputs."""
        _, activations = self._project_hidden_with_activations(x, block_indices)
        return activations

    @conditional_beartype
    def decode(self, target_column: str, x: Tensor) -> Tensor:
        hidden = self._project_hidden(x)
        output_layer = cast(nn.Linear, self.output_layers[target_column])
        return output_layer(cast_floating_to_module_dtype(hidden, output_layer)).to(
            torch.float32
        )

    @conditional_beartype
    def target_dtype(self, target_column: str) -> torch.dtype:
        return cast(nn.Linear, self.output_layers[target_column]).weight.dtype

    @conditional_beartype
    def hidden_weight_parameters(self) -> Iterator[nn.Parameter]:
        """Yield hidden linear kernels, excluding biases and output layers."""
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                yield layer.weight

    @conditional_beartype
    def forward(
        self,
        x: Tensor,
        *,
        teacher_targets: dict[str, Tensor] | None = None,
        teacher_valid_mask: Tensor | None = None,
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
                    width=self.hidden_dims[block_index],
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
                    width=output_layer.out_features,
                )
            outputs[target_column] = output
        return outputs


class AutoregressiveTransformerDecoderBranch(nn.Module):
    """An ordered, causal decoder over targets at one temporal position."""

    @beartype
    def __init__(
        self,
        *,
        target_columns: list[str],
        target_column_types: dict[str, str],
        target_n_classes: dict[str, int],
        target_global_to_decoder: dict[str, list[int]],
        input_dim: int,
        architecture: Any,
        shared_categorical_target_groups: list[list[str]],
        tie_input_output_embeddings: bool,
    ) -> None:
        super().__init__()
        self.target_columns = target_columns
        self.target_column_types = target_column_types
        self.input_dim = input_dim
        self.hidden_dims: list[int] = []
        self.hidden_weight_l2 = 0.0
        self.width = architecture.dim_model
        sequence_length = len(target_columns) + 1

        self.context_projection = nn.Linear(input_dim, self.width)
        self.bos_embedding = nn.Parameter(torch.empty(self.width))
        nn.init.normal_(self.bos_embedding, mean=0.0, std=self.width**-0.5)
        self.position_embedding = (
            nn.Embedding(sequence_length, self.width)
            if architecture.position_encoding.type == "learned"
            else None
        )
        self.transformer = TransformerEncoderStack(
            architecture, max_context_length=sequence_length
        )

        group_for_target: dict[str, int] = {}
        for group_index, group in enumerate(shared_categorical_target_groups):
            kinds = {target_column_types[target] for target in group}
            if kinds != {"categorical"}:
                raise ValueError(
                    "shared categorical target groups may contain only categorical "
                    f"targets, got {group!r}"
                )
            mappings = [target_global_to_decoder[target] for target in group]
            if any(mapping != mappings[0] for mapping in mappings[1:]):
                raise ValueError(
                    "shared categorical targets must have identical decoder-ID "
                    f"semantics, got {group!r}"
                )
            sizes = {target_n_classes[target] for target in group}
            if len(sizes) != 1:
                raise ValueError(
                    "shared categorical targets must have identical decoder "
                    f"vocabulary sizes, got {group!r}"
                )
            for target in group:
                group_for_target[target] = group_index

        self.value_embeddings = ModuleDict()
        self.output_layers = ModuleDict()
        shared_embeddings: dict[int, nn.Embedding] = {}
        shared_outputs: dict[int, nn.Linear] = {}
        self._lookup_buffer_names: dict[str, str] = {}
        for target_index, target in enumerate(target_columns):
            kind = target_column_types[target]
            group_index = group_for_target.get(target)
            is_final_target = target_index == len(target_columns) - 1
            embedding: nn.Module | None = None
            if kind == "categorical":
                size = target_n_classes[target]
                if not is_final_target or tie_input_output_embeddings:
                    embedding = (
                        shared_embeddings.setdefault(
                            group_index, nn.Embedding(size, self.width)
                        )
                        if group_index is not None
                        else nn.Embedding(size, self.width)
                    )
                elif group_index is not None:
                    # A shared group containing the final target necessarily has
                    # an earlier member whose embedding is used as decoder input.
                    embedding = shared_embeddings[group_index]
                output = (
                    shared_outputs.setdefault(group_index, nn.Linear(self.width, size))
                    if group_index is not None
                    else nn.Linear(self.width, size)
                )
                if tie_input_output_embeddings:
                    if embedding is None:
                        raise RuntimeError(
                            f"Missing tied value embedding for target {target!r}."
                        )
                    output.weight = embedding.weight
                buffer_name = f"global_to_decoder_{target_index}"
                self.register_buffer(
                    buffer_name,
                    torch.tensor(target_global_to_decoder[target], dtype=torch.long),
                )
                self._lookup_buffer_names[target] = buffer_name
            elif kind == "real":
                if group_index is not None:
                    raise ValueError(
                        f"Real target {target!r} cannot use categorical sharing"
                    )
                if not is_final_target:
                    embedding = nn.Linear(1, self.width)
                output = nn.Linear(self.width, 1)
            else:
                raise ValueError(f"Unknown target column type {kind!r}.")
            output._sequifier_decoder_output = True  # type: ignore[attr-defined]
            if embedding is not None:
                self.value_embeddings[target] = embedding
            self.output_layers[target] = output

        causal_mask = torch.full(
            (sequence_length, sequence_length), float("-inf"), dtype=torch.float32
        ).triu(diagonal=1)
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, sequence_length, sequence_length),
            persistent=False,
        )

    def _add_positions(self, sequence: Tensor) -> Tensor:
        if self.position_embedding is None:
            return sequence
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        return sequence + self.position_embedding(positions).to(sequence.dtype)

    def _embed_value(
        self,
        target: str,
        values: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        module = self.value_embeddings[target]
        if self.target_column_types[target] == "real":
            layer = cast(nn.Linear, module)
            values = values.to(dtype=layer.weight.dtype).unsqueeze(-1)
            return layer(values)
        lookup = getattr(self, self._lookup_buffer_names[target])
        global_ids = values.to(torch.long)
        if valid_mask is None:
            effective_valid_mask = torch.ones_like(global_ids, dtype=torch.bool)
        else:
            effective_valid_mask = valid_mask.bool()
            if effective_valid_mask.shape != global_ids.shape:
                raise ValueError(
                    f"Teacher validity mask cannot align to categorical target "
                    f"{target!r}: {tuple(effective_valid_mask.shape)} != "
                    f"{tuple(global_ids.shape)}."
                )
        validate_ids = (
            not torch.onnx.is_in_onnx_export() and not torch.compiler.is_compiling()
        )
        in_bounds = (global_ids >= 0) & (global_ids < lookup.numel())
        if validate_ids and bool((~in_bounds & effective_valid_mask).any()):
            raise ValueError(f"Categorical target {target!r} contains invalid IDs.")
        safe_global_ids = global_ids.masked_fill(~in_bounds | ~effective_valid_mask, 0)
        decoder_ids = lookup[safe_global_ids]
        if validate_ids and bool(((decoder_ids < 0) & effective_valid_mask).any()):
            raise ValueError(
                f"Categorical target {target!r} contains excluded special tokens "
                "at valid teacher-forcing positions."
            )
        decoder_ids = decoder_ids.masked_fill(~effective_valid_mask, 0).clamp_min(0)
        return cast(nn.Embedding, module)(decoder_ids)

    def _project_target(self, target: str, hidden: Tensor) -> Tensor:
        layer = cast(nn.Linear, self.output_layers[target])
        return layer(cast_floating_to_module_dtype(hidden, layer)).to(torch.float32)

    def trace_sites(self, branch_name: str) -> tuple[Any, ...]:
        from sequifier.model.tracing import TraceSite

        slot_axes = ("batch", "time", "slot", "channel")
        sites = [
            TraceSite(f"decoder.branch.{branch_name}.slot_input", slot_axes, self.width)
        ]
        for index in range(len(self.transformer.layers)):
            sites.extend(
                [
                    TraceSite(
                        f"decoder.branch.{branch_name}.layer.{index}.input",
                        slot_axes,
                        self.width,
                    ),
                    TraceSite(
                        f"decoder.branch.{branch_name}.layer.{index}.output",
                        slot_axes,
                        self.width,
                    ),
                ]
            )
        sites.append(
            TraceSite(f"decoder.branch.{branch_name}.final_norm", slot_axes, self.width)
        )
        return tuple(sites)

    def _run_transformer(
        self,
        sequence: Tensor,
        *,
        batch: int,
        time: int,
        trace: TraceContext | None,
        branch_name: str,
    ) -> Tensor:
        sequence = self._add_positions(sequence)
        mask = self.causal_mask[:, :, : sequence.shape[1], : sequence.shape[1]]
        hidden = sequence
        for index, layer in enumerate(self.transformer.layers):
            if trace is not None:
                shaped = hidden.reshape(batch, time, hidden.shape[1], self.width)
                shaped = trace.emit(
                    f"decoder.branch.{branch_name}.layer.{index}.input",
                    shaped,
                    axes=("batch", "time", "slot", "channel"),
                    width=self.width,
                )
                hidden = shaped.reshape(batch * time, hidden.shape[1], self.width)
            hidden = layer(hidden, src_mask=mask)
            if trace is not None:
                shaped = hidden.reshape(batch, time, hidden.shape[1], self.width)
                shaped = trace.emit(
                    f"decoder.branch.{branch_name}.layer.{index}.output",
                    shaped,
                    axes=("batch", "time", "slot", "channel"),
                    width=self.width,
                )
                hidden = shaped.reshape(batch * time, hidden.shape[1], self.width)
        hidden = self.transformer.final_norm(
            cast_floating_to_module_dtype(hidden, self.transformer.final_norm)
        )
        if trace is not None:
            shaped = hidden.reshape(batch, time, hidden.shape[1], self.width)
            shaped = trace.emit(
                f"decoder.branch.{branch_name}.final_norm",
                shaped,
                axes=("batch", "time", "slot", "channel"),
                width=self.width,
            )
            hidden = shaped.reshape(batch * time, hidden.shape[1], self.width)
        return hidden

    def _teacher_forced(
        self,
        context: Tensor,
        teacher_targets: dict[str, Tensor],
        *,
        teacher_valid_mask: Tensor | None,
        trace: TraceContext | None,
        branch_name: str,
    ) -> dict[str, Tensor]:
        batch, time, _ = context.shape
        missing = set(self.target_columns).difference(teacher_targets)
        if missing:
            raise ValueError(
                f"Missing teacher targets for decoder branch: {sorted(missing)!r}."
            )
        projected = self.context_projection(
            cast_floating_to_module_dtype(context, self.context_projection)
        ).reshape(batch * time, 1, self.width)
        shifted = [self.bos_embedding.expand(batch * time, 1, self.width)]
        valid_mask = (
            None if teacher_valid_mask is None else teacher_valid_mask[:, -time:]
        )
        if valid_mask is not None and valid_mask.shape != (batch, time):
            raise ValueError(
                "Teacher validity mask cannot align to decoded shape "
                f"{(batch, time)} from {tuple(valid_mask.shape)}."
            )
        for target in self.target_columns[:-1]:
            values = teacher_targets[target][:, -time:]
            if values.shape[:2] != (batch, time):
                raise ValueError(
                    f"Teacher target {target!r} cannot align to decoded shape "
                    f"{(batch, time)} from {tuple(values.shape)}."
                )
            embedded = self._embed_value(target, values, valid_mask).reshape(
                batch * time, 1, self.width
            )
            shifted.append(embedded)
        sequence = torch.cat([projected, *shifted], dim=1)
        if trace is not None:
            shaped = trace.emit(
                f"decoder.branch.{branch_name}.slot_input",
                sequence.reshape(batch, time, sequence.shape[1], self.width),
                axes=("batch", "time", "slot", "channel"),
                width=self.width,
            )
            sequence = shaped.reshape(batch * time, sequence.shape[1], self.width)
        hidden = self._run_transformer(
            sequence,
            batch=batch,
            time=time,
            trace=trace,
            branch_name=branch_name,
        )[:, 1:]
        outputs = {}
        for index, target in enumerate(self.target_columns):
            output = self._project_target(target, hidden[:, index]).reshape(
                batch, time, -1
            )
            if trace is not None:
                output = trace.emit(
                    f"decoder.branch.{branch_name}.logits.{target}",
                    output,
                    axes=("batch", "time", "channel"),
                    width=output.shape[-1],
                )
            outputs[target] = output
        return outputs

    def _greedy(
        self,
        context: Tensor,
        *,
        trace: TraceContext | None,
        branch_name: str,
    ) -> dict[str, Tensor]:
        batch, time, _ = context.shape
        projected = self.context_projection(
            cast_floating_to_module_dtype(context, self.context_projection)
        ).reshape(batch * time, 1, self.width)
        sequence = torch.cat(
            [projected, self.bos_embedding.expand(batch * time, 1, self.width)], dim=1
        )
        outputs: dict[str, Tensor] = {}
        for index, target in enumerate(self.target_columns):
            if trace is not None:
                shaped = trace.emit(
                    f"decoder.branch.{branch_name}.slot_input",
                    sequence.reshape(batch, time, sequence.shape[1], self.width),
                    axes=("batch", "time", "slot", "channel"),
                    width=self.width,
                )
                sequence = shaped.reshape(batch * time, sequence.shape[1], self.width)
            hidden = self._run_transformer(
                sequence,
                batch=batch,
                time=time,
                trace=trace,
                branch_name=branch_name,
            )[:, -1]
            logits = self._project_target(target, hidden)
            output = logits.reshape(batch, time, -1)
            if trace is not None:
                output = trace.emit(
                    f"decoder.branch.{branch_name}.logits.{target}",
                    output,
                    axes=("batch", "time", "channel"),
                    width=output.shape[-1],
                )
            outputs[target] = output
            if index + 1 < len(self.target_columns):
                generated = (
                    logits.argmax(dim=-1)
                    if self.target_column_types[target] == "categorical"
                    else logits.squeeze(-1)
                )
                if self.target_column_types[target] == "categorical":
                    embedded = cast(nn.Embedding, self.value_embeddings[target])(
                        generated.to(torch.long)
                    )
                else:
                    embedded = self._embed_value(target, generated)
                sequence = torch.cat([sequence, embedded.unsqueeze(1)], dim=1)
        return outputs

    def forward(
        self,
        x: Tensor,
        *,
        teacher_targets: dict[str, Tensor] | None = None,
        teacher_valid_mask: Tensor | None = None,
        trace: TraceContext | None = None,
        branch_name: str = "default",
    ) -> dict[str, Tensor]:
        if teacher_targets is not None:
            return self._teacher_forced(
                x,
                teacher_targets,
                teacher_valid_mask=teacher_valid_mask,
                trace=trace,
                branch_name=branch_name,
            )
        return self._greedy(x, trace=trace, branch_name=branch_name)


class TargetDecoding(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        branches: dict[str, nn.Module],
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

    @conditional_beartype
    def __contains__(self, target_column: object) -> bool:
        return isinstance(target_column, str) and target_column in self.target_to_branch

    @conditional_beartype
    def decode(self, target_column: str, x: Tensor) -> Tensor:
        branch = cast(
            TargetDecoderBranch,
            self.branches[self.target_to_branch[target_column]],
        )
        return branch.decode(target_column, x)

    @conditional_beartype
    def target_dtype(self, target_column: str) -> torch.dtype:
        branch = cast(
            TargetDecoderBranch,
            self.branches[self.target_to_branch[target_column]],
        )
        return branch.target_dtype(target_column)

    @conditional_beartype
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

    @conditional_beartype
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

    @conditional_beartype
    def forward(
        self,
        x: Tensor,
        *,
        teacher_targets: dict[str, Tensor] | None = None,
        teacher_valid_mask: Tensor | None = None,
        trace: TraceContext | None = None,
    ) -> dict[str, Tensor]:
        branch_outputs = {
            branch_name: branch(
                x,
                teacher_targets=teacher_targets,
                teacher_valid_mask=teacher_valid_mask,
                trace=trace,
                branch_name=branch_name,
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


@beartype
def resolve_decoding_plan(hparams: Any) -> DecodingPlan:
    decoding_spec = hparams.model.decoder

    if decoding_spec.type == "composite":
        branch_items = list(decoding_spec.branches.items())
        default_target_columns = None
    else:
        branch_items = [("default", decoding_spec)]
        default_target_columns = hparams.target_columns

    branches = []
    target_to_branch = {}
    for branch_name, branch_config in branch_items:
        if branch_config.type not in {
            *DECODER_HIDDEN_DIMS,
            "autoregressive_transformer",
        }:
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
        if branch_config.type == "autoregressive_transformer":
            for group in branch_config.shared_categorical_target_groups:
                noncategorical = [
                    target
                    for target in group
                    if hparams.target_column_types[target] != "categorical"
                ]
                if noncategorical:
                    raise ValueError(
                        "shared categorical target groups contain real targets: "
                        f"{noncategorical!r}"
                    )
                decoder_ids = [hparams.target_decoder_ids[target] for target in group]
                if any(ids != decoder_ids[0] for ids in decoder_ids[1:]):
                    raise ValueError(
                        "shared categorical targets must have identical decoder-ID "
                        f"semantics, got {group!r}"
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
            "model.decoder must decode every target column; "
            f"missing {sorted(undecoded_columns)}"
        )
    return DecodingPlan(tuple(branches), target_to_branch)


@beartype
def build_target_decoding(
    hparams: Any,
    target_n_classes: Optional[dict[str, int]] = None,
    target_global_to_decoder: Optional[dict[str, list[int]]] = None,
) -> TargetDecoding:
    model = hparams.model
    plan = resolve_decoding_plan(hparams)

    decoder_n_classes = (
        hparams.n_classes if target_n_classes is None else target_n_classes
    )
    input_dim = model.backbone.architecture.dim_model * model.decoder.support

    branches = {}
    for branch in plan.branches:
        branch_config = branch.config
        if branch_config.type == "autoregressive_transformer":
            if target_global_to_decoder is None:
                raise ValueError(
                    "Autoregressive transformer decoding requires categorical "
                    "global-to-decoder mappings."
                )
            branches[branch.name] = AutoregressiveTransformerDecoderBranch(
                target_columns=list(branch.target_columns),
                target_column_types=hparams.target_column_types,
                target_n_classes=decoder_n_classes,
                target_global_to_decoder=target_global_to_decoder,
                input_dim=input_dim,
                architecture=branch_config.architecture,
                shared_categorical_target_groups=(
                    branch_config.shared_categorical_target_groups
                ),
                tie_input_output_embeddings=(branch_config.tie_input_output_embeddings),
            )
        else:
            branches[branch.name] = TargetDecoderBranch(
                target_columns=list(branch.target_columns),
                target_column_types=hparams.target_column_types,
                n_classes=decoder_n_classes,
                input_dim=input_dim,
                hidden_dims=DECODER_HIDDEN_DIMS[branch_config.type](branch_config),
                activation=getattr(branch_config, "activation", "relu"),
                dropout=getattr(branch_config, "dropout", 0.0),
                hidden_weight_l2=getattr(branch_config, "hidden_weight_l2", 0.0),
            )

    return TargetDecoding(
        branches=branches,
        target_columns=hparams.target_columns,
        target_to_branch=plan.target_to_branch,
    )
