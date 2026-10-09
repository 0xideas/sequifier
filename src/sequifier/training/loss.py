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
    loss_targets: dict[str, Tensor] | None = None
    loss_valid_mask: Tensor | None = None


@dataclass(frozen=True)
class LossResult:
    backward_loss: Tensor
    target_losses: dict[str, Tensor]
    accounting_sums: dict[str, Tensor]
    accounting_count: Tensor
    component_losses: dict[str, tuple[Tensor, ...]] | None = None


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
        if interface.depth_layouts:
            from sequifier.model.execution_schema import ExecutionSchema

            ExecutionSchema.from_interface(
                interface, interface.window_view.context_length
            ).validate(batch.inputs, batch.metadata)
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
        loss_valid_mask = dataset.objective.build_loss_mask(metadata)
        transformed_targets, loss_valid_mask = (
            dataset.objective.transform_targets_for_loss(targets, loss_valid_mask)
        )
        loss_targets = {
            target: dataset.objective.target_values_for_loss(
                target, transformed_targets
            )
            for target in interface.target_columns
        }
        return PreparedBatch(
            features=features,
            targets=targets,
            metadata=metadata,
            loss_targets=loss_targets,
            loss_valid_mask=loss_valid_mask,
        )

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
        valid_mask = batch.loss_valid_mask
        targets = batch.loss_targets
        if targets is None or valid_mask is None:
            # Compatibility for callers constructing PreparedBatch directly.
            valid_mask = dataset.objective.build_loss_mask(batch.metadata)
            transformed, valid_mask = dataset.objective.transform_targets_for_loss(
                batch.targets, valid_mask
            )
            targets = {
                target: dataset.objective.target_values_for_loss(target, transformed)
                for target in target_names
            }
        decoded_length = next(iter(output.logits.values())).shape[1]
        valid_mask = valid_mask[:, -decoded_length:]
        flat_mask = valid_mask.reshape(-1).bool()
        local_count = flat_mask.sum(dtype=torch.int64)
        sums: dict[str, Tensor] = {}
        components: dict[str, Tensor] = {}
        hash_component_losses: dict[str, tuple[Tensor, ...]] = {}
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
            if (
                target in getattr(interface, "categorical_hashing", {})
                and target not in output.auxiliary_logits
            ):
                raise RuntimeError(
                    f"Hashed target {target!r} is missing auxiliary component logits; "
                    "resolved candidate scores cannot be used for loss."
                )
            logits = output.logits[target]
            target_values = targets[target]
            target_values = target_values[:, -decoded_length:].reshape(-1)
            excluded: Tensor | None = None
            if kind == "categorical":
                global_ids = target_values.to(torch.int64)
                lookup = torch.tensor(
                    dataset.runtime_metadata.target_global_to_decoder[target],
                    device=global_ids.device,
                )
                target_for_loss = lookup[global_ids]
                excluded = target_for_loss < 0
                if target in output.auxiliary_logits:
                    codec = network.resolve_interface(
                        dataset.interface_name
                    ).decoder.hash_codecs[target]
                    if bool((excluded & flat_mask).any()):
                        raise ValueError(
                            f"Categorical target {target!r} contains excluded special "
                            "tokens at valid loss positions."
                        )
                    safe_ids = global_ids.masked_fill(excluded, 0)
                    codes = codec.encode(safe_ids)
                    component_sums = []
                    for index, component_logits in enumerate(
                        output.auxiliary_logits[target]
                    ):
                        flat_logits = component_logits.float().reshape(
                            -1, codec.widths[index]
                        )
                        label = codes[:, index]
                        if flat_logits.shape[0] != flat_mask.numel():
                            raise RuntimeError(
                                f"Loss/mask size mismatch for {target!r}"
                            )
                        raw_component = dataset.criteria[target](flat_logits, label)
                        component_sums.append(
                            raw_component.reshape(-1).masked_select(flat_mask).sum()
                        )
                    sums[target] = sum(component_sums) / len(component_sums)
                    for index, value in enumerate(component_sums):
                        sums[f"@hash/{target}/{index}"] = value
                    weight = float((dataset.loss_weights or {}).get(target, 1.0))
                    scaled = tuple(
                        value
                        * (weight / len(component_sums))
                        * world_size
                        / denominator.to(value.dtype)
                        for value in component_sums
                    )
                    hash_component_losses[target] = scaled
                    components[target] = sum(scaled)
                    if weight > 0:
                        total = (
                            components[target]
                            if total is None
                            else total + components[target]
                        )
                    continue
                logits_for_loss = logits.float().reshape(
                    -1, dataset.runtime_metadata.target_n_classes[target]
                )
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
            if weight == 0.0:
                components[target] = sums[target].detach().new_zeros(())
                continue
            component = (
                sums[target] * weight * world_size / denominator.to(sums[target].dtype)
            )
            components[target] = component
            total = component if total is None else total + component
        if total is None:
            raise RuntimeError("Loss calculation requires a positive-weight target.")
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
            component_losses=hash_component_losses,
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
        reduced_sums, reduced_count = self.reduce_accounting(sums, count, targets)
        return self.finalize_reduced_accounting(
            reduced_sums,
            reduced_count,
            dataset,
            allow_empty=allow_empty,
        )

    def reduce_accounting(
        self,
        sums: dict[str, Tensor],
        count: Tensor,
        targets: list[str],
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Reduce unweighted per-target sums and their shared token count."""
        targets = targets + sorted(set(sums) - set(targets))
        packed = torch.stack(
            [sums[target] for target in targets]
            + [count.to(next(iter(sums.values())).dtype)]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return (
            {target: packed[index] for index, target in enumerate(targets)},
            packed[-1],
        )

    def finalize_reduced_accounting(
        self,
        sums: dict[str, Tensor],
        count: Tensor,
        dataset: DatasetRuntime,
        *,
        allow_empty: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the dataset's current loss weights to reduced accounting sums."""
        logical_targets = list(dataset.config.interface.target_columns)
        targets = logical_targets + sorted(set(sums) - set(logical_targets))
        if count.item() == 0:
            if not allow_empty:
                raise RuntimeError("No valid loss tokens found.")
            zero = next(iter(sums.values())).new_zeros(())
            return zero, {target: zero.clone() for target in targets}
        target_losses = {}
        for target in targets:
            if target in logical_targets:
                weight = float((dataset.loss_weights or {}).get(target, 1.0))
            else:
                logical = target[len("@hash/") :].rsplit("/", 1)[0]
                width_count = len(
                    dataset.config.interface.categorical_hash_contracts[logical][
                        "widths"
                    ]
                )
                weight = (
                    float((dataset.loss_weights or {}).get(logical, 1.0)) / width_count
                )
            target_losses[target] = sums[target] / count * weight
        zero = next(iter(sums.values())).new_zeros(())
        return sum(
            (target_losses[target] for target in logical_targets), start=zero
        ), target_losses
