"""Objective-aware loss calculation outside the network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from sequifier.io.batch import SequifierBatch
from sequifier.model.network import ComposableTransformerNetwork, ModelOutput
from sequifier.training.runtime import DatasetRuntime


@dataclass(frozen=True)
class PreparedBatch:
    features: dict[str, Tensor]
    targets: dict[str, Tensor]
    metadata: dict[str, Tensor]


@dataclass(frozen=True)
class LossResult:
    backward_loss: Tensor
    target_losses: dict[str, Tensor]
    accounting_sums: dict[str, Tensor]
    accounting_count: Tensor


class LossService:
    def prepare_batch(
        self,
        batch: SequifierBatch,
        dataset: DatasetRuntime,
        device: torch.device,
        *,
        eval_seed: int | None = None,
    ) -> PreparedBatch:
        interface = dataset.config.interface
        features = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.inputs.items()
            if key in interface.input_columns
        }
        targets = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.targets.items()
            if key in interface.target_column_types
        }
        metadata = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.metadata.items()
        }
        features, targets, metadata = dataset.objective.prepare_batch(
            features, targets, metadata, eval_seed=eval_seed
        )
        return PreparedBatch(features, targets, metadata)

    def calculate(
        self,
        output: ModelOutput,
        batch: PreparedBatch,
        dataset: DatasetRuntime,
        network: ComposableTransformerNetwork,
    ) -> LossResult:
        interface = dataset.config.interface
        target_names = list(interface.target_columns)
        missing = set(target_names).difference(batch.targets)
        if missing:
            raise RuntimeError(f"Missing target columns: {sorted(missing)!r}.")
        valid_mask = dataset.objective.build_loss_mask(batch.metadata)
        targets, valid_mask = dataset.objective.transform_targets_for_loss(
            batch.targets, valid_mask
        )
        decoded_length = next(iter(output.logits.values())).shape[1]
        valid_mask = valid_mask[:, -decoded_length:]
        flat_mask = valid_mask.reshape(-1).bool()
        local_count = flat_mask.sum(dtype=torch.int64)
        sums: dict[str, Tensor] = {}
        components: dict[str, Tensor] = {}
        global_count = local_count.detach().clone()
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if world_size > 1:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        denominator = global_count.clamp_min(1)
        total: Tensor | None = None
        for target in target_names:
            kind = interface.target_column_types[target]
            logits = output.logits[target]
            target_values = dataset.objective.target_values_for_loss(target, targets)
            target_values = target_values[:, -decoded_length:].reshape(-1)
            excluded: Tensor | None = None
            if kind == "categorical":
                logits_for_loss = logits.float().reshape(
                    -1, dataset.runtime_metadata.target_n_classes[target]
                )
                global_ids = target_values.to(torch.int64)
                lookup = torch.tensor(
                    dataset.runtime_metadata.target_global_to_decoder[target],
                    device=global_ids.device,
                )
                target_for_loss = lookup[global_ids]
                excluded = target_for_loss < 0
            elif kind == "real":
                logits_for_loss = logits.float().reshape(-1)
                target_for_loss = target_values.to(logits_for_loss.dtype)
            else:
                raise ValueError(f"Unknown target column type {kind!r}.")
            output_count = logits_for_loss.shape[0]
            if (
                output_count != target_for_loss.numel()
                or output_count != flat_mask.numel()
            ):
                raise RuntimeError(
                    f"Loss/mask size mismatch for {target!r}: "
                    f"output={output_count}, target={target_for_loss.numel()}, "
                    f"mask={flat_mask.numel()}."
                )
            if excluded is not None:
                if bool((excluded & flat_mask).any()):
                    raise ValueError(
                        f"Categorical target {target!r} contains excluded special "
                        "tokens at valid loss positions."
                    )
                target_for_loss = target_for_loss.masked_fill(excluded, 0)
            raw = dataset.criteria[target](logits_for_loss, target_for_loss)
            if raw.numel() != flat_mask.numel():
                raise RuntimeError(
                    f"Loss/mask size mismatch for {target!r}: "
                    f"{raw.numel()} != {flat_mask.numel()}."
                )
            sums[target] = raw.reshape(-1).masked_select(flat_mask).sum()
            weight = float((dataset.loss_weights or {}).get(target, 1.0))
            component = (
                sums[target] * weight * world_size / denominator.to(sums[target].dtype)
            )
            components[target] = component
            total = component if total is None else total + component
        if total is None:
            raise RuntimeError("Loss calculation produced no target components.")
        backward_loss: Tensor = total + network.regularization_loss(
            dataset.interface_name
        )
        accounting_dtype = (
            torch.float32 if backward_loss.device.type == "mps" else torch.float64
        )
        return LossResult(
            backward_loss=backward_loss,
            target_losses=components,
            accounting_sums={
                name: value.detach().to(accounting_dtype)
                for name, value in sums.items()
            },
            accounting_count=local_count.detach(),
        )

    def finalize_accounting(
        self,
        sums: dict[str, Tensor],
        count: Tensor,
        dataset: DatasetRuntime,
        *,
        allow_empty: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        targets = list(dataset.config.interface.target_columns)
        packed = torch.stack(
            [sums[target] for target in targets]
            + [count.to(next(iter(sums.values())).dtype)]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced_count = packed[-1]
        if reduced_count.item() == 0:
            if not allow_empty:
                raise RuntimeError("No valid loss tokens found.")
            zeros = {target: packed[0].new_zeros(()) for target in targets}
            return packed[0].new_zeros(()), zeros
        target_losses = {
            target: packed[index]
            / reduced_count
            * float((dataset.loss_weights or {}).get(target, 1.0))
            for index, target in enumerate(targets)
        }
        return sum(target_losses.values(), start=packed[0].new_zeros(())), target_losses
