"""Fit and validate frozen policy features over an episode view."""

import polars as pl
import torch

from sequifier.config.policy_config import CategoricalPolicyConfig
from sequifier.io.policy_window import encode_feature
from sequifier.io.preprocessing import (
    _finite_mean_and_std,
    create_id_map,
    get_combined_statistics,
    normalize_values,
)
from sequifier.trajectory.returns import return_to_go


def return_labels(episode, length, gamma):
    if episode.transitions is None or episode.transitions.rewards is None:
        raise ValueError("return-conditioned examples require recorded rewards")
    return return_to_go(episode.transitions.rewards[:length], gamma)


class EpisodeFeatures:
    """Selected episode lengths define fitting coverage independently of batching."""

    def __init__(self, episodes, config, lengths=None):
        self.episodes, self.config = episodes, config
        self.lengths = (
            lengths
            if lengths is not None
            else [episodes.metadata(i).length for i in range(len(episodes))]
        )
        self._validated_config = self._validated_statistics = None
        ids = [episodes.metadata(i).episode_id for i in range(len(episodes))]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episode IDs in fitting view")

    def values(self, episode, length, column):
        return (
            episode.observations[column][:length]
            if column in episode.observations
            else [episode.static_inputs[column]]
        )

    def fit_categorical_config(self) -> CategoricalPolicyConfig:
        """Fit only declared raw categorical features, using the training episodes."""
        values = self.config.model_dump()
        for column in self.config.categorical_columns:
            if (
                column in self.config.id_maps
                or column in self.config.categorical_n_classes
            ):
                continue
            vocabulary = set()
            for episode, length in zip(self.episodes, self.lengths):
                vocabulary.update(self.values(episode, length, column))
            if not vocabulary:
                raise ValueError(f"cannot fit empty categorical column {column!r}")
            if len({isinstance(value, str) for value in vocabulary}) > 1:
                raise ValueError(
                    f"{column}: mixed string/numeric category labels are ambiguous"
                )
            if any(
                not isinstance(value, str) and value != int(value)
                for value in vocabulary
            ):
                raise ValueError(
                    f"{column}: categorical numeric labels must be integers"
                )
            mapping = create_id_map(pl.DataFrame({column: sorted(vocabulary)}), column)
            values["id_maps"][column] = mapping
            key = (
                "observation_n_classes"
                if column in self.config.observation_columns
                else "static_n_classes"
            )
            values[key][column] = max(mapping.values()) + 1
        return CategoricalPolicyConfig.model_validate(values)

    def use_config(self, config: CategoricalPolicyConfig) -> None:
        if self._validated_config == config:
            return
        self.config = config
        self._validated_statistics = None
        # Validate held-out values against the fixed training codec before updates.
        for episode, length in zip(self.episodes, self.lengths):
            for column, values in episode.observations.items():
                for value in values[:length]:
                    encode_feature(value, column, config)
            for column, value in episode.static_inputs.items():
                encode_feature(value, column, config)

        self._validated_config = config.model_copy(deep=True)

    def validate_normalization(self, statistics):
        """Check FP32 transforms episode-by-episode without constructing windows."""
        if self._validated_statistics == statistics:
            return
        for episode, length in zip(self.episodes, self.lengths):
            for column in self.config.observation_columns + self.config.static_columns:
                if column in self.config.categorical_n_classes:
                    continue
                values = self.values(episode, length, column)
                tensor = torch.tensor(values, dtype=torch.float32)
                transformed = normalize_values(tensor, statistics[column])
                if (
                    not torch.isfinite(tensor).all()
                    or not torch.isfinite(transformed).all()
                ):
                    raise ValueError(
                        f"episode {episode.episode_id}, {column}: non-finite FP32 input/normalization"
                    )
            if self.config.return_conditioned:
                returns = return_labels(episode, length, self.config.returns.gamma)
                if not torch.isfinite(
                    torch.tensor(returns, dtype=torch.float32)
                    / self.config.returns.scale
                ).all():
                    raise ValueError(
                        f"episode {episode.episode_id}: non-finite conditioned return"
                    )

        self._validated_statistics = {k: dict(v) for k, v in statistics.items()}

    def normalization_statistics(self):
        """Use the same stable sample moments as supervised preprocessing."""
        moments = {}
        columns = self.config.observation_columns + self.config.static_columns
        for episode, length in zip(self.episodes, self.lengths):
            for column in columns:
                if column in self.config.categorical_n_classes:
                    continue
                values = self.values(episode, length, column)
                mean, std = _finite_mean_and_std(pl.Series(column, values), column)
                size = len(values)
                if column in moments:
                    count, old_mean, old_std = moments[column]
                    mean, std = get_combined_statistics(
                        size, mean, std, count, old_mean, old_std
                    )
                    size += count
                moments[column] = size, mean, std
        return {
            column: {"mean": mean, "std": std}
            for column, (_, mean, std) in moments.items()
        }
