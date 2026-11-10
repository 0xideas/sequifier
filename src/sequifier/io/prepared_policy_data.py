"""The small data contract shared by policy phase preparation."""

from collections.abc import Sized
from typing import TYPE_CHECKING, Protocol

from sequifier.io.episode_features import EpisodeFeatures

if TYPE_CHECKING:
    from sequifier.rl.policy import CategoricalPolicy


class PreparedPolicyData(Protocol):
    @property
    def training(self) -> Sized: ...

    @property
    def contract(self) -> list: ...

    def fitting_dataset(self) -> EpisodeFeatures: ...

    def validate(self, actor: "CategoricalPolicy") -> None: ...
