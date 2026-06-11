from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class SequifierBatch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    metadata: dict[str, torch.Tensor]
    sequence_ids: Optional[torch.Tensor] = None
    subsequence_ids: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterator[object]:
        yield self.inputs
        yield self.targets
        yield self.metadata
        yield self.sequence_ids
        yield self.subsequence_ids
