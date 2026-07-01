import math
from abc import ABC
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from torch import Tensor

from sequifier.special_tokens import SPECIAL_TOKEN_IDS


class Objective(ABC):
    """Closed internal interface for built-in training objectives."""

    name: ClassVar[str]
    forward_looking: ClassVar[bool] = True
    uses_causal_attention: ClassVar[bool] = True

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

    @classmethod
    def default_target_offset(cls) -> int:
        return 1

    @classmethod
    def default_prediction_length(cls, context_length: int) -> int:
        return 1

    @classmethod
    def validate_window_view(cls, context_length: int, target_offset: int) -> None:
        if target_offset < 1:
            raise ValueError(
                "Causal, final_value, and next_occurrence views require "
                "target_offset >= 1"
            )

    @classmethod
    def validate_prediction_length(
        cls,
        prediction_length: int,
        context_length: int,
        *,
        usage: str = "training/inference",
    ) -> None:
        return None

    @classmethod
    def build_attention_mask_policy(cls, context_length: int) -> Tensor:
        if cls.uses_causal_attention:
            return torch.triu(
                torch.ones(context_length, context_length) * float("-inf"),
                diagonal=1,
            )
        return torch.zeros(context_length, context_length)

    def prepare_batch(
        self,
        data_batch: dict[str, Tensor],
        targets_batch: dict[str, Tensor],
        metadata_batch: dict[str, Tensor],
        eval_seed: Optional[int] = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        return data_batch, targets_batch, metadata_batch

    def build_loss_mask(self, metadata: dict[str, Tensor]) -> Tensor:
        valid_mask = metadata["target_valid_mask"].bool()

        if "bert_mask" in metadata:
            valid_mask = valid_mask & metadata["bert_mask"].bool()

        if "sample_valid_mask" in metadata:
            sample_valid_mask = metadata["sample_valid_mask"].bool()

            if sample_valid_mask.ndim != 1:
                raise ValueError("sample_valid_mask must have shape [batch_size].")
            if sample_valid_mask.shape[0] != valid_mask.shape[0]:
                raise ValueError(
                    "sample_valid_mask batch dimension does not match target_valid_mask."
                )

            valid_mask = valid_mask & sample_valid_mask.unsqueeze(1)

        return valid_mask

    def transform_targets_for_loss(
        self,
        targets: dict[str, Tensor],
        valid_mask: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        return targets, valid_mask

    def target_values_for_loss(
        self, target_column: str, targets: dict[str, Tensor]
    ) -> Tensor:
        return targets[target_column]

    def baseline_prediction_values(
        self,
        target_column: str,
        data: dict[str, Tensor],
        targets: dict[str, Tensor],
        target_column_type: str,
    ) -> Tensor:
        return data[target_column].transpose(0, 1)

    def baseline_target_values(
        self, target_column: str, targets: dict[str, Tensor]
    ) -> Tensor:
        return targets[target_column]

    @classmethod
    def item_positions(
        cls,
        start_positions: np.ndarray,
        context_length: int,
        prediction_length: int,
    ) -> np.ndarray:
        base_positions = start_positions + context_length
        position_offsets = np.arange(-prediction_length + 1, 1)
        repeated_bases = np.repeat(base_positions, prediction_length)
        tiled_offsets = np.tile(position_offsets, len(start_positions))
        return repeated_bases + tiled_offsets


class CausalObjective(Objective):
    name = "causal"


class FinalValueObjective(CausalObjective):
    name = "final_value"

    def target_values_for_loss(
        self, target_column: str, targets: dict[str, Tensor]
    ) -> Tensor:
        target_values = targets[target_column]
        return target_values[:, -1:].expand_as(target_values)


class NextOccurrenceObjective(CausalObjective):
    name = "next_occurrence"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        next_occurrence_config = config.training_spec.next_occurrence_config

        self.next_occurrence_column = next_occurrence_config.column_name
        id_map = config.id_maps[self.next_occurrence_column]
        self.next_occurrence_target_ids = [
            id_map[value] for value in next_occurrence_config.target_values
        ]

    def transform_targets_for_loss(
        self,
        targets: dict[str, Tensor],
        valid_mask: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        trigger_values = targets[self.next_occurrence_column]
        target_ids = torch.tensor(
            self.next_occurrence_target_ids,
            device=trigger_values.device,
            dtype=trigger_values.dtype,
        )
        occurrence_mask = (
            trigger_values.unsqueeze(-1) == target_ids.view(1, 1, -1)
        ).any(dim=-1) & valid_mask.bool()

        batch_size, seq_len = occurrence_mask.shape
        position_ids = torch.arange(
            seq_len, device=trigger_values.device, dtype=torch.int64
        ).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        sentinel_positions = torch.full(
            (batch_size, seq_len),
            seq_len,
            device=trigger_values.device,
            dtype=torch.int64,
        )
        occurrence_positions = torch.where(
            occurrence_mask, position_ids, sentinel_positions
        )
        next_positions = torch.cummin(
            occurrence_positions.flip(dims=[1]), dim=1
        ).values.flip(dims=[1])
        has_next_occurrence = next_positions < seq_len
        next_occurrence_mask = valid_mask.bool() & has_next_occurrence
        gather_indices = next_positions.clamp_max(seq_len - 1)

        projected_targets = {}
        for target_column, target_tensor in targets.items():
            projected_targets[target_column] = target_tensor.gather(1, gather_indices)

        return projected_targets, next_occurrence_mask


class BERTObjective(Objective):
    name = "bert"
    forward_looking = False
    uses_causal_attention = False

    @classmethod
    def default_target_offset(cls) -> int:
        return 0

    @classmethod
    def default_prediction_length(cls, context_length: int) -> int:
        return context_length

    @classmethod
    def validate_window_view(cls, context_length: int, target_offset: int) -> None:
        if target_offset != 0:
            raise ValueError("BERT views require target_offset=0")

    @classmethod
    def validate_prediction_length(
        cls,
        prediction_length: int,
        context_length: int,
        *,
        usage: str = "training/inference",
    ) -> None:
        if prediction_length != context_length:
            if usage == "training":
                raise ValueError(
                    "For BERT training, model_spec.prediction_length must be equal "
                    "to context_length "
                    f"(got prediction_length={prediction_length}, "
                    f"context_length={context_length})."
                )
            if usage == "inference":
                raise ValueError(
                    "For BERT inference, prediction_length must be equal to "
                    "context_length "
                    f"(got prediction_length={prediction_length}, "
                    f"context_length={context_length})."
                )
            raise ValueError(
                "For BERT training/inference, prediction_length must be equal to "
                f"context_length (got prediction_length={prediction_length}, "
                f"context_length={context_length})."
            )

    def prepare_batch(
        self,
        data_batch: dict[str, Tensor],
        targets_batch: dict[str, Tensor],
        metadata_batch: dict[str, Tensor],
        eval_seed: Optional[int] = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        return apply_bert_masking(
            data_batch,
            targets_batch,
            metadata_batch,
            self.config,
            eval_seed,
        )

    def build_loss_mask(self, metadata: dict[str, Tensor]) -> Tensor:
        valid_mask = super().build_loss_mask(metadata)
        if "bert_mask" not in metadata:
            raise ValueError("BERT loss masking requires metadata['bert_mask']")
        return valid_mask & metadata["bert_mask"].bool()

    def baseline_prediction_values(
        self,
        target_column: str,
        data: dict[str, Tensor],
        targets: dict[str, Tensor],
        target_column_type: str,
    ) -> Tensor:
        shifted_targets = torch.roll(targets[target_column], shifts=1, dims=1)
        if target_column_type == "categorical":
            shifted_targets[:, 0] = SPECIAL_TOKEN_IDS.unknown
        else:
            shifted_targets[:, 0] = 0.0
        return shifted_targets.transpose(0, 1)

    @classmethod
    def item_positions(
        cls,
        start_positions: np.ndarray,
        context_length: int,
        prediction_length: int,
    ) -> np.ndarray:
        base_positions = start_positions
        position_offsets = np.arange(0, prediction_length)
        repeated_bases = np.repeat(base_positions, prediction_length)
        tiled_offsets = np.tile(position_offsets, len(start_positions))
        return repeated_bases + tiled_offsets


OBJECTIVE_REGISTRY: dict[str, type[Objective]] = {
    CausalObjective.name: CausalObjective,
    BERTObjective.name: BERTObjective,
    FinalValueObjective.name: FinalValueObjective,
    NextOccurrenceObjective.name: NextOccurrenceObjective,
}
OBJECTIVE_NAME_MESSAGE = "'causal', 'bert', 'final_value', and 'next_occurrence'"
ALLOWED_OBJECTIVE_NAMES = frozenset(OBJECTIVE_REGISTRY)


def get_objective_class(name: str) -> type[Objective]:
    try:
        return OBJECTIVE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {name}"
        ) from exc


def create_objective(config: Any) -> Objective:
    return get_objective_class(config.training_spec.training_objective)(config)


def target_offset_for_objective(name: str, configured_target_offset: int = 1) -> int:
    objective_class = get_objective_class(name)
    if objective_class.default_target_offset() == 0:
        return 0
    return configured_target_offset


def forward_objective_names() -> frozenset[str]:
    return frozenset(
        name
        for name, objective_class in OBJECTIVE_REGISTRY.items()
        if objective_class.forward_looking
    )


def _build_bert_span_mask(
    valid_mask: Tensor,
    masking_probability: float,
    span_distribution: Any,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Construct exact-budget, non-overlapping BERT span masks."""
    valid_mask = valid_mask.bool()
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device

    valid_lengths = valid_mask.sum(dim=1, dtype=torch.long)
    budgets = (valid_lengths.to(torch.float32) * masking_probability).to(torch.long)

    max_spans = max(1, math.floor(seq_len * masking_probability) + 10)
    sampled_lengths = span_distribution.sample(
        (batch_size, max_spans),
        device=device,
        generator=generator,
    )
    sampled_lengths = sampled_lengths.to(torch.long).clamp_min_(1)

    used_before = sampled_lengths.cumsum(dim=1) - sampled_lengths
    remaining = (budgets[:, None] - used_before).clamp_min(0)
    span_lengths = torch.minimum(sampled_lengths, remaining)

    n_spans = (span_lengths > 0).sum(dim=1)
    total_gap_length = valid_lengths - budgets

    gap_slot = torch.arange(max_spans + 1, device=device)
    active_gap_slot = gap_slot[None, :] <= n_spans[:, None]

    uniform = torch.rand(
        (batch_size, max_spans + 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    uniform = uniform.clamp_min(torch.finfo(uniform.dtype).tiny)

    gap_weights = torch.where(
        active_gap_slot,
        -torch.log(uniform),
        torch.zeros_like(uniform),
    )
    cumulative_weights = gap_weights.cumsum(dim=1)
    weight_totals = cumulative_weights[:, -1:].clamp_min(
        torch.finfo(gap_weights.dtype).tiny
    )

    gap_edges = torch.floor(
        cumulative_weights
        / weight_totals
        * total_gap_length[:, None].to(gap_weights.dtype)
    ).to(torch.long)
    gap_edges = torch.where(
        gap_slot[None, :] >= n_spans[:, None],
        total_gap_length[:, None],
        gap_edges,
    )

    gaps = torch.diff(
        torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                gap_edges,
            ],
            dim=1,
        ),
        dim=1,
    )

    lengths_before = span_lengths.cumsum(dim=1) - span_lengths
    gaps_through_current = gaps[:, :max_spans].cumsum(dim=1)

    span_starts = lengths_before + gaps_through_current
    span_ends = span_starts + span_lengths

    compact_position = valid_mask.to(torch.long).cumsum(dim=1) - 1
    compact_position = compact_position.clamp_min(0)

    started_spans = torch.searchsorted(
        span_starts.contiguous(),
        compact_position.contiguous(),
        right=True,
    )
    ended_spans = torch.searchsorted(
        span_ends.contiguous(),
        compact_position.contiguous(),
        right=True,
    )

    return valid_mask & (started_spans > ended_spans)


def apply_bert_masking(
    data_batch: dict[str, Tensor],
    targets_batch: dict[str, Tensor],
    metadata_batch: Optional[dict[str, Tensor]],
    config: Any,
    eval_seed: Optional[int] = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Apply BERT span corruption and attach prediction masks."""
    if not metadata_batch or "attention_valid_mask" not in metadata_batch:
        raise ValueError("BERT masking requires metadata['attention_valid_mask']")

    valid_mask = metadata_batch["attention_valid_mask"].bool()
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device

    for target_name, target in targets_batch.items():
        if target.shape != valid_mask.shape:
            raise ValueError(
                f"BERT target {target_name!r} has shape {target.shape}; "
                f"expected {valid_mask.shape}"
            )

    generator: Optional[torch.Generator] = None
    if eval_seed is not None:
        seeded_generator = torch.Generator(device=device)
        seeded_generator.manual_seed(eval_seed)
        generator = seeded_generator

    bert_spec = config.training_spec.bert_spec
    if bert_spec is None:
        raise ValueError("bert_spec must be configured for BERT training")

    bert_mask = _build_bert_span_mask(
        valid_mask,
        bert_spec.masking_probability,
        bert_spec.span_masking,
        generator=generator,
    )

    replacement = bert_spec.replacement_distribution
    p_masked = replacement.masked
    p_random = replacement.random

    replacement_probs = torch.rand(
        (batch_size, seq_len),
        device=device,
        generator=generator,
    )

    mask_token_mask = bert_mask & (replacement_probs < p_masked)
    random_token_mask = (
        bert_mask
        & (replacement_probs >= p_masked)
        & (replacement_probs < (p_masked + p_random))
    )

    masked_data = dict(data_batch)

    for col, tensor in data_batch.items():
        if col in config.categorical_columns:
            output = tensor.clone()

            if p_masked > 0.0:
                output.masked_fill_(mask_token_mask, SPECIAL_TOKEN_IDS.mask)

            if p_random > 0.0:
                random_tokens = torch.randint(
                    low=SPECIAL_TOKEN_IDS.user_start,
                    high=config.n_classes[col],
                    size=tensor.shape,
                    device=device,
                    dtype=tensor.dtype,
                    generator=generator,
                )
                output[random_token_mask] = random_tokens[random_token_mask]

            masked_data[col] = output

        elif col in config.real_columns:
            output = tensor.clone()

            if p_masked > 0.0:
                output.masked_fill_(mask_token_mask, 0.0)

            if p_random > 0.0:
                random_noise = torch.randn(
                    tensor.shape,
                    device=device,
                    dtype=tensor.dtype,
                    generator=generator,
                )
                output[random_token_mask] = random_noise[random_token_mask]

            masked_data[col] = output

    detached_targets = {col: tensor.detach() for col, tensor in targets_batch.items()}
    output_metadata = {key: tensor.detach() for key, tensor in metadata_batch.items()}
    output_metadata["bert_mask"] = bert_mask
    output_metadata["attention_valid_mask"] = valid_mask

    return masked_data, detached_targets, output_metadata
