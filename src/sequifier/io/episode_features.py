"""Fit and validate frozen policy features over an episode view."""

from copy import deepcopy
from dataclasses import replace

import polars as pl

from sequifier.config.policy_inputs import PolicyFittingSpec
from sequifier.io.feature_fitting import FeatureStatistics, create_id_map
from sequifier.io.policy_window import encode_feature
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
        self._features_validated = False
        self._validated_statistics = None
        ids = [episodes.metadata(i).episode_id for i in range(len(episodes))]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episode IDs in fitting view")

    def values(self, episode, length, column):
        return (
            episode.observations[column][:length]
            if column in episode.observations
            else [episode.static_inputs[column]]
        )

    def fit_categorical_config(self) -> PolicyFittingSpec:
        """Fit only declared raw categorical features, using the training episodes."""
        if not isinstance(self.config, PolicyFittingSpec):
            raise ValueError("Feature fitting requires a pre-resolution episode view")
        id_maps = self.config.id_maps.copy()
        n_classes = self.config.categorical_n_classes.copy()
        for column in self.config.categorical_columns:
            if column in id_maps or column in n_classes:
                continue
            vocabulary = set()
            for episode, length in zip(self.episodes, self.lengths):
                vocabulary.update(self.values(episode, length, column))
            mapping = create_id_map(vocabulary, column)
            id_maps[column] = mapping
            n_classes[column] = max(mapping.values()) + 1
        return replace(self.config, id_maps=id_maps, categorical_n_classes=n_classes)

    def validate_features(self) -> None:
        if self._features_validated:
            return
        config = self.config
        # Validate held-out values against the fixed training codec before updates.
        for episode, length in zip(self.episodes, self.lengths):
            for column, values in episode.observations.items():
                for value in values[:length]:
                    encode_feature(value, column, config)
            for column, value in episode.static_inputs.items():
                encode_feature(value, column, config)

        self._features_validated = True

    def validate_normalization(self, interface):
        """Check fitted transforms at the input columns' stored precision."""
        statistics = interface.selected_columns_statistics
        signature = (interface.normalize_real_columns, statistics)
        if self._validated_statistics == signature:
            return
        from sequifier.io.policy_window import PolicyWindowView, checked_policy_features

        view = PolicyWindowView(self.config)
        for episode, length in zip(self.episodes, self.lengths):
            for step in range(length):
                try:
                    inputs = view.episode_inputs(episode, step, None)
                    checked_policy_features(inputs, interface)
                except ValueError as error:
                    raise ValueError(
                        f"episode {episode.episode_id}, item {episode.item_positions[step]}: {error}"
                    ) from error
        self._validated_statistics = deepcopy(signature)

    def normalization_statistics(self):
        """Use the same stable sample moments as supervised preprocessing."""
        statistics = FeatureStatistics()
        columns = self.config.observation_columns + self.config.static_columns
        for episode, length in zip(self.episodes, self.lengths):
            for column in columns:
                if column in self.config.categorical_n_classes:
                    continue
                values = self.values(episode, length, column)
                statistics.observe_real(pl.Series(column, values))
        return {
            column: {"mean": values["mean"], "std": values["std"]}
            for column, values in statistics.moments.items()
        }
