from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch

from sequifier.typechecking import beartype

REQUIRED_METADATA_KEYS = frozenset({"attention_valid_mask", "target_valid_mask"})


@dataclass(frozen=True)
class SequifierInputs:
    """Named model tensors, independent of labels or learning algorithm."""

    inputs: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]

    @property
    def attention_valid_mask(self) -> torch.Tensor:
        return self.metadata["attention_valid_mask"]

    def to(self, device: str | torch.device) -> "SequifierInputs":
        return SequifierInputs(
            {
                key: value.to(device, non_blocking=True)
                for key, value in self.inputs.items()
            },
            {
                key: value.to(device, non_blocking=True)
                for key, value in self.metadata.items()
            },
        )


def collate_model_inputs(examples: Sequence[SequifierInputs]) -> SequifierInputs:
    if not examples:
        raise ValueError("cannot collate empty model inputs")
    first = examples[0]
    if any(
        value.inputs.keys() != first.inputs.keys()
        or value.metadata.keys() != first.metadata.keys()
        for value in examples
    ):
        raise ValueError("model inputs must share feature and metadata schemas")
    return SequifierInputs(
        {
            key: torch.cat([value.inputs[key] for value in examples])
            for key in first.inputs
        },
        {
            key: torch.cat([value.metadata[key] for value in examples])
            for key in first.metadata
        },
    )


@dataclass(frozen=True)
class SequifierBatch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]
    sequence_ids: Optional[torch.Tensor] = None
    subsequence_ids: Optional[torch.Tensor] = None

    @property
    def model_inputs(self) -> SequifierInputs:
        return SequifierInputs(self.inputs, self.metadata)

    def to(self, device: str | torch.device) -> "SequifierBatch":
        model_inputs = self.model_inputs.to(device)
        return SequifierBatch(
            model_inputs.inputs,
            {
                key: value.to(device, non_blocking=True)
                for key, value in self.targets.items()
            },
            model_inputs.metadata,
            self.sequence_ids.to(device) if self.sequence_ids is not None else None,
            self.subsequence_ids.to(device)
            if self.subsequence_ids is not None
            else None,
        )

    @beartype
    def __post_init__(self) -> None:
        metadata_keys = self.metadata.keys() if self.metadata is not None else set()
        missing_keys = REQUIRED_METADATA_KEYS - metadata_keys
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"SequifierBatch metadata is missing required keys: {missing}"
            )

    @beartype
    def pin_memory(self) -> "SequifierBatch":
        return SequifierBatch(
            inputs={k: v.pin_memory() for k, v in self.inputs.items()},
            targets={k: v.pin_memory() for k, v in self.targets.items()},
            metadata={k: v.pin_memory() for k, v in self.metadata.items()},
            sequence_ids=(
                self.sequence_ids.pin_memory()
                if self.sequence_ids is not None
                else None
            ),
            subsequence_ids=(
                self.subsequence_ids.pin_memory()
                if self.subsequence_ids is not None
                else None
            ),
        )


def collate_sequence_batches(examples: Sequence[SequifierBatch]) -> SequifierBatch:
    model_inputs = collate_model_inputs([value.model_inputs for value in examples])
    first = examples[0]
    if any(value.targets.keys() != first.targets.keys() for value in examples):
        raise ValueError("sequence batches must share target schemas")

    def identities(name):
        values = [getattr(value, name) for value in examples]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"sequence batches must consistently supply {name}")
        return torch.cat(values)

    return SequifierBatch(
        model_inputs.inputs,
        {
            key: torch.cat([value.targets[key] for value in examples])
            for key in first.targets
        },
        model_inputs.metadata,
        identities("sequence_ids"),
        identities("subsequence_ids"),
    )
