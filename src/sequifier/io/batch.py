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
