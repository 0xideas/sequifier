"""Joint sampling and recorded-prefix scoring over existing model decoders."""

import torch

from sequifier.io.actions import (
    action_component_key,
    action_latent_key,
    action_record,
    decode_action,
)
from sequifier.io.preprocessing import denormalize_values, normalize_values
from sequifier.model.prediction_distribution import RealPredictionDistribution
from sequifier.rl.distributions import CategoricalActionDistribution


class JointActionDistribution:
    """The model owns all weights; this object owns one encoded input window."""

    def __init__(self, actor, inputs):
        self.actor = actor
        self.features, self.metadata = actor.model_inputs(inputs)
        self.representation = actor.network.encode(
            self.features, self.metadata, interface_name=actor.interface_name
        )
        self.batch, self.time = self.representation.shape[:2]

    def _teachers(self, actions):
        interface = self.actor.interface
        teachers = {}
        for name in interface.target_columns:
            categorical = interface.target_column_types[name] == "categorical"
            fill = interface.target_decoder_ids[name][0] if categorical else 0.0
            values = torch.full(
                (self.batch, self.time),
                fill,
                dtype=torch.long if categorical else torch.float32,
                device=self.representation.device,
            )
            if name in actions:
                action = actions[name]
                if categorical:
                    ids = self.actor.action_decoder_ids(name)
                    action = torch.tensor(ids, device=values.device)[action.long()]
                elif (
                    interface.normalize_real_columns
                    and name in interface.selected_columns_statistics
                ):
                    action = normalize_values(
                        action, interface.selected_columns_statistics[name]
                    )
                values[:, self.actor.config.decision_position] = action.to(values.dtype)
            teachers[name] = values
        return teachers

    def _parameters(self, actions):
        return self.actor.network.decode_representation(
            self.representation,
            teacher_targets=self._teachers(actions),
            teacher_valid_mask=self.metadata["attention_valid_mask"],
            interface_name=self.actor.interface_name,
        )

    def _field(self, name, parameters):
        actor, field = self.actor, self.actor.config.actions[name]
        output = parameters[name][:, actor.config.decision_position]
        if field.type == "categorical":
            indices = [
                actor.interface.target_decoder_ids[name].index(i)
                for i in actor.action_decoder_ids(name)
            ]
            return CategoricalActionDistribution(output[:, indices])
        location = output.squeeze(-1)
        stats = actor.interface.selected_columns_statistics.get(name)
        if stats is not None and actor.interface.normalize_real_columns:
            location = denormalize_values(location, stats)
        return RealPredictionDistribution(
            location,
            actor.network.resolve_interface(
                actor.interface_name
            ).prediction_scales.scale(name, field),
            field,
        )

    def _active(self, name, actions):
        mask = torch.ones(
            self.batch, dtype=torch.bool, device=self.representation.device
        )
        for predecessor, labels in self.actor.config.actions[name].active_when.items():
            if predecessor not in actions:
                raise ValueError(
                    "conditional activity requires a preceding action field"
                )
            field = self.actor.config.actions[predecessor]
            allowed = (
                [field.values.index(label) for label in labels]
                if field.type == "categorical"
                else labels
            )
            mask &= torch.isin(
                actions[predecessor], torch.tensor(allowed, device=mask.device)
            )
        return mask

    def select(self, deterministic, generator):
        actions = {}
        for name in self.actor.action_order:
            distribution = self._field(name, self._parameters(actions))
            value = (
                distribution.mode() if deterministic else distribution.sample(generator)
            )
            field = self.actor.config.actions[name]
            inactive = (
                field.values.index(field.inactive_value)
                if field.type == "categorical" and field.active_when
                else (0 if field.type == "categorical" else field.inactive_value)
            )
            active = self._active(name, actions)
            actions[name] = torch.where(active, value, value.new_tensor(inactive))
            if isinstance(distribution, RealPredictionDistribution):
                latent = (
                    distribution.location
                    if deterministic
                    else distribution.sampled_latent
                )
                actions[action_latent_key(name, self.actor.config)] = latent
                actions[action_component_key(name, self.actor.config)] = (
                    torch.full_like(value, -1, dtype=torch.long)
                    if deterministic
                    else distribution.sampled_component
                )
        return actions

    def sample(self, generator=None):
        return self.select(False, generator)

    def mode(self):
        return self.select(True, None)

    def log_prob(self, actions):
        fields = {key for key in actions if key in self.actor.config.actions}
        if fields != set(self.actor.config.actions):
            raise ValueError("joint likelihood requires every action field")
        allowed = set(self.actor.config.actions)
        for name, field in self.actor.config.actions.items():
            if field.type == "real":
                allowed.update(
                    (
                        action_latent_key(name, self.actor.config),
                        action_component_key(name, self.actor.config),
                    )
                )
        if set(actions) - allowed:
            raise ValueError("joint action contains unknown fields")
        parameters = self._parameters(actions)
        score = self.representation.new_zeros(
            self.batch,
            dtype=torch.float32
            if self.representation.device.type == "mps"
            else torch.float64,
        )
        for name in self.actor.action_order:
            value = actions[name]
            if value.shape != (self.batch,):
                raise ValueError(f"{name}: expected one action per decision")
            distribution = self._field(name, parameters)
            active = self._active(name, actions)
            field = self.actor.config.actions[name]
            inactive = (
                field.values.index(field.inactive_value)
                if field.type == "categorical" and field.active_when
                else (0 if field.type == "categorical" else field.inactive_value)
            )
            if (value[~active] != inactive).any():
                raise ValueError(f"{name}: inactive action differs from its point mass")
            safe = torch.where(active, value, distribution.mode().detach())
            latent = actions.get(action_latent_key(name, self.actor.config))
            if (
                isinstance(distribution, RealPredictionDistribution)
                and latent is not None
            ):
                latent = torch.where(active, latent, distribution.location.detach())
                mixture = actions.get(action_component_key(name, self.actor.config))
                if mixture is not None:
                    mixture = torch.where(active, mixture, -1)
                component = distribution.log_prob(
                    safe, latent=latent, component=mixture
                )
            else:
                component = distribution.log_prob(safe)
            score = score + torch.where(active, component, 0.0)
        return score.float()

    def categorical_probabilities(self, actions):
        """Per-field conditional probabilities at the supplied action prefix."""
        parameters = self._parameters(actions)
        result = {}
        for name, field in self.actor.config.actions.items():
            if field.type == "categorical":
                distribution = self._field(name, parameters)
                assert isinstance(distribution, CategoricalActionDistribution)
                probabilities = distribution.distribution.probs
                if field.active_when:
                    point = torch.zeros_like(probabilities)
                    point[:, field.values.index(field.inactive_value)] = 1
                    probabilities = torch.where(
                        self._active(name, actions)[:, None], probabilities, point
                    )
                result[name] = probabilities
        return result

    @property
    def entropy_is_exact(self):
        return not self.actor.autoregressive_actions and all(
            not field.active_when
            and field.lower is None
            and field.upper is None
            and not field.atoms
            for field in self.actor.config.actions.values()
        )

    def entropy(self, generator=None):
        if self.entropy_is_exact:
            parameters = self._parameters({})
            return (
                torch.stack(
                    [
                        self._field(name, parameters).entropy()
                        for name in self.actor.action_order
                    ]
                )
                .sum(dim=0)
                .float()
            )
        # Score-function estimator includes the distribution of sampled prefixes.
        # Its value estimates -E log p; its surrogate has the entropy gradient.
        with torch.no_grad():
            actions = self.sample(generator)
        logp = self.log_prob(actions)
        surrogate = -(logp.detach() + 1) * logp
        return -logp.detach() + surrogate - surrogate.detach()

    def kl(self, reference, generator=None):
        if self.actor.config.actions != reference.actor.config.actions:
            raise ValueError(
                "KL requires matching action measures and physical transforms"
            )
        if not self.actor.autoregressive_actions and not any(
            field.active_when for field in self.actor.config.actions.values()
        ):
            current, frozen = self._parameters({}), reference._parameters({})
            values = []
            for name, field in self.actor.config.actions.items():
                p, q = self._field(name, current), reference._field(name, frozen)
                if isinstance(p, CategoricalActionDistribution) and isinstance(
                    q, CategoricalActionDistribution
                ):
                    values.append(
                        torch.distributions.kl_divergence(
                            p.distribution, q.distribution
                        )
                    )
                else:
                    if not isinstance(p, RealPredictionDistribution) or not isinstance(
                        q, RealPredictionDistribution
                    ):
                        raise ValueError("reference distribution families differ")
                    values.append(
                        (1 - sum(field.atoms.values()))
                        * torch.distributions.kl_divergence(p.normal, q.normal)
                    )
            return torch.stack(values).sum(dim=0).float()
        with torch.no_grad():
            actions = self.sample(generator)
        logp = self.log_prob(actions)
        delta = logp - reference.log_prob(actions)
        surrogate = (delta.detach() + 1) * logp
        return delta.detach() + surrogate - surrogate.detach()

    def domain_actions(self, actions):
        return [
            decode_action(action_record(actions, i), self.actor.config)
            for i in range(self.batch)
        ]
