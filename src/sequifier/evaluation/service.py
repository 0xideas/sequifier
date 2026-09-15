"""Validation execution, baseline accounting, and class distributions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Sequence

import torch
from torch.nn.functional import one_hot

from sequifier.helpers import construct_index_maps
from sequifier.model.network import ModelOutput
from sequifier.training.loss import LossService, PreparedBatch
from sequifier.training.runtime import DatasetRuntimeRegistry, build_source_runtime


@dataclass(frozen=True)
class EvaluationContext:
    run_id: str
    session_id: str
    phase_index: int
    phase_epoch: int
    epoch: int
    training_batch: int
    training_batches_total: int
    global_step: int
    device: torch.device
    rank: int
    world_size: int
    distributed_strategy: Any
    kind: str = "validation"


@dataclass(frozen=True)
class EvaluationSourceResult:
    source: str
    dataset: str
    part: str | None
    total_loss: float
    target_losses: dict[str, float]
    baseline_loss: float
    baseline_target_losses: dict[str, float]
    class_distributions: dict[str, list[dict[str, Any]]]
    count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    sources: dict[str, EvaluationSourceResult] = field(default_factory=dict)

    @property
    def total_losses(self) -> dict[str, float]:
        return {name: result.total_loss for name, result in self.sources.items()}


@dataclass(frozen=True)
class BaselineAccounting:
    """Globally reduced baseline statistics before loss weighting."""

    sums: dict[str, torch.Tensor]
    count: torch.Tensor


class EvaluationService:
    def __init__(self, loss_service: LossService) -> None:
        self.loss_service = loss_service
        self._baselines: dict[tuple[str, str], BaselineAccounting] = {}

    @staticmethod
    def _baseline_definition(dataset: Any) -> str:
        """Identify inputs that affect raw baseline loss statistics."""
        interface = dataset.config.interface
        training = getattr(dataset.objective.config, "global_training", None)

        def model_value(value: Any) -> Any:
            dump = getattr(value, "model_dump", None)
            if callable(dump):
                return dump(mode="json")
            if is_dataclass(value) and not isinstance(value, type):
                return asdict(value)
            return value

        definition = {
            "objective": dataset.objective.name,
            "objective_options": {
                "bert_spec": model_value(getattr(training, "bert_spec", None)),
                "next_occurrence_config": model_value(
                    getattr(training, "next_occurrence_config", None)
                ),
            },
            "window_view": model_value(interface.window_view),
            "prediction_length": getattr(interface.decoder, "prediction_length", None),
            "target_columns": interface.target_columns,
            "target_column_types": interface.target_column_types,
            "target_decoder_ids": interface.target_decoder_ids,
            "target_global_to_decoder": interface.target_global_to_decoder,
            "criterion": dataset.config.criterion,
            "class_weights": dataset.class_weights,
        }
        return json.dumps(definition, sort_keys=True, separators=(",", ":"))

    def _baseline_output(
        self, prepared: PreparedBatch, dataset: Any, decoded_length: int
    ) -> ModelOutput:
        interface = dataset.config.interface
        logits = {}
        for target in interface.target_columns:
            values = dataset.objective.baseline_prediction_values(
                target,
                prepared.features,
                prepared.targets,
                interface.target_column_types[target],
            ).transpose(0, 1)[:, -decoded_length:]
            if interface.target_column_types[target] == "categorical":
                global_ids = values.to(torch.int64)
                lookup = torch.tensor(
                    dataset.runtime_metadata.target_global_to_decoder[target],
                    device=values.device,
                )
                mapped = lookup[global_ids]
                values = one_hot(
                    mapped.clamp_min(0),
                    dataset.runtime_metadata.target_n_classes[target],
                ).to(torch.float32) * (mapped >= 0).unsqueeze(-1)
            logits[target] = values
        return ModelOutput(logits=logits, prediction_positions=slice(None))

    def evaluate(
        self,
        network: Any,
        sources: Sequence[Any],
        datasets: DatasetRuntimeRegistry,
        context: EvaluationContext,
    ) -> EvaluationResult:
        results: dict[str, EvaluationSourceResult] = {}
        was_training = network.training
        network.eval()
        try:
            with torch.no_grad():
                for source_config in sources:
                    started = time.perf_counter()
                    source = build_source_runtime(source_config, datasets)
                    source.set_epoch(context.phase_epoch, "validation")
                    dataset = source.dataset
                    targets = list(dataset.config.interface.target_columns)
                    dtype = (
                        torch.float32 if context.device.type == "mps" else torch.float64
                    )
                    sums = {
                        target: torch.zeros((), device=context.device, dtype=dtype)
                        for target in targets
                    }
                    count = torch.zeros((), device=context.device, dtype=dtype)
                    baseline_sums = {
                        target: torch.zeros((), device=context.device, dtype=dtype)
                        for target in targets
                    }
                    baseline_count = torch.zeros((), device=context.device, dtype=dtype)
                    class_counts = {
                        target: torch.zeros(
                            dataset.runtime_metadata.target_n_classes[target],
                            device=context.device,
                            dtype=torch.int64,
                        )
                        for target in dataset.config.class_share_log_columns
                    }
                    baseline_key = (
                        source_config.source,
                        self._baseline_definition(dataset),
                    )
                    calculate_baseline = baseline_key not in self._baselines
                    for batch_index, runtime_batch in enumerate(
                        source.iter_batches("validation")
                    ):
                        prepared = self.loss_service.prepare_batch(
                            runtime_batch.batch,
                            dataset,
                            context.device,
                            eval_seed=(
                                context.phase_index * 1_000_003
                                + context.phase_epoch * 10_007
                                + batch_index
                            ),
                        )
                        dataset.metrics.evaluation_batches += 1
                        output = network(
                            prepared.features,
                            prepared.metadata,
                            interface_name=dataset.interface_name,
                        )
                        expected_batch_size = next(
                            iter(prepared.features.values())
                        ).shape[0]
                        for target, logits in output.logits.items():
                            if logits.shape[0] != expected_batch_size:
                                raise ValueError(
                                    f"Expected input batch_size {expected_batch_size}, "
                                    f"got {logits.shape[0]} for target {target!r}."
                                )
                        loss = self.loss_service.calculate(
                            output, prepared, dataset, network
                        )
                        for target, value in loss.accounting_sums.items():
                            sums[target] += value.to(dtype)
                        count += loss.accounting_count.to(dtype)
                        if calculate_baseline:
                            baseline_targets = {
                                target: dataset.objective.baseline_target_values(
                                    target, prepared.targets
                                )
                                for target in targets
                            }
                            baseline_prepared = PreparedBatch(
                                prepared.features, baseline_targets, prepared.metadata
                            )
                            baseline = self.loss_service.calculate(
                                self._baseline_output(
                                    prepared,
                                    dataset,
                                    next(iter(output.logits.values())).shape[1],
                                ),
                                baseline_prepared,
                                dataset,
                                network,
                            )
                            for target, value in baseline.accounting_sums.items():
                                baseline_sums[target] += value.to(dtype)
                            baseline_count += baseline.accounting_count.to(dtype)
                        valid_mask = dataset.objective.build_loss_mask(
                            prepared.metadata
                        )
                        _, valid_mask = dataset.objective.transform_targets_for_loss(
                            prepared.targets, valid_mask
                        )
                        decoded_length = next(iter(output.logits.values())).shape[1]
                        mask = valid_mask[:, -decoded_length:].reshape(-1).bool()
                        for target, counts in class_counts.items():
                            predicted = output.logits[target].argmax(dim=-1).reshape(-1)
                            counts += torch.bincount(
                                predicted[mask].to(torch.int64),
                                minlength=counts.numel(),
                            )
                    total, target_values = self.loss_service.finalize_accounting(
                        sums, count, dataset
                    )
                    if calculate_baseline:
                        reduced_sums, reduced_count = (
                            self.loss_service.reduce_accounting(
                                baseline_sums, baseline_count, targets
                            )
                        )
                        self._baselines[baseline_key] = BaselineAccounting(
                            sums={
                                target: value.detach().clone()
                                for target, value in reduced_sums.items()
                            },
                            count=reduced_count.detach().clone(),
                        )
                    baseline_accounting = self._baselines[baseline_key]
                    baseline_total, baseline_values = (
                        self.loss_service.finalize_reduced_accounting(
                            baseline_accounting.sums,
                            baseline_accounting.count,
                            dataset,
                        )
                    )
                    baseline_total_value = float(baseline_total.item())
                    baseline_target_values = {
                        name: float(value.item())
                        for name, value in baseline_values.items()
                    }
                    distributions: dict[str, list[dict[str, Any]]] = {}
                    index_maps = construct_index_maps(
                        dataset.config.interface.id_maps,
                        list(class_counts),
                        True,
                    )
                    for target, counts in class_counts.items():
                        if (
                            torch.distributed.is_available()
                            and torch.distributed.is_initialized()
                        ):
                            torch.distributed.all_reduce(counts)
                        total_count = int(counts.sum().item())
                        decoder_ids = dataset.runtime_metadata.target_decoder_ids[
                            target
                        ]
                        distributions[target] = [
                            {
                                "class_id": decoder_ids[index],
                                "class_label": index_maps[target][decoder_ids[index]],
                                "count": int(value.item()),
                                "total_count": total_count,
                                "share": (
                                    float(value.item()) / total_count
                                    if total_count
                                    else 0.0
                                ),
                            }
                            for index, value in enumerate(counts)
                            if value.item()
                        ]
                    results[source_config.source] = EvaluationSourceResult(
                        source=source_config.source,
                        dataset=source_config.dataset,
                        part=source_config.part,
                        total_loss=float(total.item()),
                        target_losses={
                            name: float(value.item())
                            for name, value in target_values.items()
                        },
                        baseline_loss=baseline_total_value,
                        baseline_target_losses=baseline_target_values,
                        class_distributions=distributions,
                        count=int(count.item()),
                        elapsed_seconds=time.perf_counter() - started,
                    )
        finally:
            network.train(was_training)
        return EvaluationResult(results)
