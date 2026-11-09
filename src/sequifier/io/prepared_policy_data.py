"""The small data contract shared by policy phase preparation."""

from collections.abc import Sized
from typing import TYPE_CHECKING, Protocol

from sequifier.io.policy_dataset import PolicySequenceDataset

if TYPE_CHECKING:
    from sequifier.rl.policy import CategoricalPolicy


class PreparedPolicyData(Protocol):
    @property
    def training(self) -> Sized: ...

    @property
    def contract(self) -> list: ...

    def fitting_dataset(self) -> PolicySequenceDataset: ...

    def validate(self, actor: "CategoricalPolicy") -> None: ...
