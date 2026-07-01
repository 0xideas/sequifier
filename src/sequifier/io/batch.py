from dataclasses import dataclass
from typing import Optional

import torch

REQUIRED_METADATA_KEYS = frozenset({"attention_valid_mask", "target_valid_mask"})


@dataclass(frozen=True)
class SequifierBatch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]
    sequence_ids: Optional[torch.Tensor] = None
    subsequence_ids: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        metadata_keys = self.metadata.keys() if self.metadata is not None else set()
        missing_keys = REQUIRED_METADATA_KEYS - metadata_keys
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"SequifierBatch metadata is missing required keys: {missing}"
            )

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
