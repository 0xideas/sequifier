"""Joint sampling and recorded-prefix scoring over existing model decoders."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from sequifier.io.actions import action_record, decode_action
from sequifier.io.preprocessing import denormalize_values, normalize_values
from sequifier.model.prediction_distribution import RealPredictionDistribution
from sequifier.rl.distributions import CategoricalActionDistribution


@dataclass(frozen=True)
class ScoredSelection:
    actions: dict[str, Tensor]
    log_prob: Tensor
    probabilities: dict[str, Tensor]


class JointActionDistribution:
    """The model owns all weights; this object owns one encoded input window."""

    def __init__(self, actor, inputs, *, _encoded=None):
        self.actor = actor
        if _encoded is None:
            self.features, self.metadata = actor.model_inputs(inputs)
            self.representation = actor.network.encode(
                self.features, self.metadata, interface_name=actor.interface_name
            )
        else:
            self.features, self.metadata, self.representation = _encoded
        self.plan = actor.action_plan
        self.route = actor.network.resolve_interface(actor.interface_name)
        self.decoder = self.route.decoder
        self.batch = self.representation.shape[0]
        end = self.representation.shape[1] + self.plan.position + 1
        start = end - self.route.decoding_support
        if start < 0:
            raise ValueError("decision_position lies outside the source decoder output")
        self.context = self.route.decoder_input(self.representation[:, start:end])[:, 0]
        self.valid = self.metadata["attention_valid_mask"][:, self.plan.position]
        self.support, self.activity = self.plan.indices(self.context.device)
        interface = actor.interface
        self.statistics = (
            interface.selected_columns_statistics
            if interface.normalize_real_columns
            else {}
        )

    def row(self, index):
        """View one encoded batch row without re-encoding or sharing action RNG."""
        if not 0 <= index < self.batch:
            raise IndexError("joint-distribution row is outside the batch")
        return type(self)(
            self.actor,
            None,
            _encoded=(
                {key: value[index : index + 1] for key, value in self.features.items()},
                {key: value[index : index + 1] for key, value in self.metadata.items()},
                self.representation[index : index + 1],
            ),
        )

    def _independent_parameters(self):
        outputs = {}
        for branch_name, targets in self.plan.independent.items():
            branch = self.decoder.branches[branch_name]
            outputs.update(branch.decode_targets(self.context, targets))
        return outputs

    def _teacher(self, name, action):
        field = self.plan.fields[name]
        if action is None:
            teacher = self.context.new_zeros(
                self.batch, dtype=torch.long if field.support else torch.float32
            )
        elif field.support:
            teacher = self.support[name][action.long()].masked_fill(~self.valid, 0)
        else:
            stats = self.statistics.get(name)
            teacher = normalize_values(action, stats) if stats is not None else action
            teacher = teacher.float().expand(self.batch)
        return teacher

    def _parameters(self, actions):
        outputs = self._independent_parameters()
        for branch_name, targets in self.plan.autoregressive.items():
            branch = self.decoder.branches[branch_name]
            teachers = {
                name: self._teacher(name, actions.get(name)) for name in targets[:-1]
            }
            prefix = branch.start_prefix(self.context, teachers)
            outputs.update(branch.decode_prefix(prefix, targets))
        return outputs

    def _field(self, name, output):
        field = self.plan.fields[name].spec
        output = self.decoder.resolve_target(name, output)
        if field.type == "categorical":
            return CategoricalActionDistribution(
                output.index_select(-1, self.support[name])
            )
        location = output.squeeze(-1)
        stats = self.statistics.get(name)
        if stats is not None:
            location = denormalize_values(location, stats)
        scale = (
            self.route.prediction_scales.scale(name)
            if field.scale == "model"
            else field.scale.fixed
        )
        return RealPredictionDistribution(location, scale, field)

    def _active(self, name, actions):
        active = None
        for predecessor, allowed in self.activity.get(name, {}).items():
            if predecessor not in actions:
                raise ValueError(
                    "conditional activity requires a preceding action field"
                )
            matches = torch.isin(actions[predecessor], allowed)
            active = matches if active is None else active & matches
        return active

    def _select(self, deterministic, generator):
        actions, fields, prefixes = {}, {}, {}
        outputs = self._independent_parameters()
        for name in self.plan.order:
            if name in self.plan.prefix_steps:
                branch_name, previous = self.plan.prefix_steps[name]
                branch = self.decoder.branches[branch_name]
                prefix = (
                    branch.start_prefix(self.context)
                    if previous is None
                    else branch.append_prefix(
                        prefixes[branch_name],
                        previous,
                        self._teacher(previous, actions[previous]),
                    )
                )
                prefixes[branch_name] = prefix
                outputs.update(branch.decode_prefix(prefix, (name,)))
            distribution = self._field(name, outputs[name])
            value = (
                distribution.mode() if deterministic else distribution.sample(generator)
            )
            field = self.plan.fields[name]
            active = self._active(name, actions)
            actions[name] = (
                value
                if active is None
                else torch.where(active, value, value.new_tensor(field.inactive))
            )
            if isinstance(distribution, RealPredictionDistribution):
                actions[field.latent_key] = (
                    distribution.location
                    if deterministic
                    else distribution.sampled_latent
                )
                actions[field.component_key] = (
                    torch.full_like(value, -1, dtype=torch.long)
                    if deterministic
                    else distribution.sampled_component
                )
            fields[name] = distribution, active
        return actions, fields

    def select(self, deterministic, generator):
        actions, _ = self._select(deterministic, generator)
        return actions

    def select_and_score(
        self, deterministic, generator, *, include_probabilities=False
    ):
        """Score the selected action with the distributions used to select it."""
        actions, fields = self._select(deterministic, generator)
        return ScoredSelection(
            actions,
            self._score(actions, fields),
            self._probabilities(fields) if include_probabilities else {},
        )

    def sample(self, generator=None):
        return self.select(False, generator)

    def mode(self):
        return self.select(True, None)

    def _score(self, actions, fields):
        score = self.context.new_zeros(
            self.batch,
            dtype=torch.float32 if self.context.device.type == "mps" else torch.float64,
        )
        for name in self.plan.order:
            distribution, active = fields[name]
            field = self.plan.fields[name]
            value = actions[name]
            if active is not None:
                if ((value != field.inactive) & ~active).any():
                    raise ValueError(
                        f"{name}: inactive action differs from its point mass"
                    )
                value = torch.where(active, value, distribution.mode().detach())
            latent = actions.get(field.latent_key)
            if (
                isinstance(distribution, RealPredictionDistribution)
                and latent is not None
            ):
                component = actions.get(field.component_key)
                if active is not None:
                    latent = torch.where(active, latent, distribution.location.detach())
                    if component is not None:
                        component = torch.where(active, component, -1)
                log_prob = distribution.log_prob(
                    value, latent=latent, component=component
                )
            else:
                log_prob = distribution.log_prob(value)
            score = score + (
                log_prob if active is None else torch.where(active, log_prob, 0.0)
            )
        return score.float()

    def log_prob(self, actions):
        if not isinstance(actions, Mapping) or self.plan.fields.keys() - actions.keys():
            raise ValueError("joint likelihood requires every action field")
        if actions.keys() - self.plan.record_fields:
            raise ValueError("joint action contains unknown fields")
        for name in self.plan.order:
            if actions[name].shape != (self.batch,):
                raise ValueError(f"{name}: expected one action per decision")
        outputs = self._parameters(actions)
        fields = {
            name: (self._field(name, outputs[name]), self._active(name, actions))
            for name in self.plan.order
        }
        return self._score(actions, fields)

    def _probabilities(self, fields):
        result = {}
        for name in self.actor.config.actions:
            if name not in fields:
                continue
            distribution, active = fields[name]
            if isinstance(distribution, CategoricalActionDistribution):
                probabilities = distribution.distribution.probs
                if active is not None:
                    point = torch.zeros_like(probabilities)
                    point[:, self.plan.fields[name].inactive] = 1
                    probabilities = torch.where(active[:, None], probabilities, point)
                result[name] = probabilities
        return result

    def categorical_probabilities(self, actions):
        """Per-field conditional probabilities at the supplied action prefix."""
        outputs = self._parameters(actions)
        return self._probabilities(
            {
                name: (self._field(name, outputs[name]), self._active(name, actions))
                for name in self.support
            }
        )

    @property
    def entropy_is_exact(self):
        return not self.plan.has_autoregression and all(
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
                        self._field(name, parameters[name]).entropy()
                        for name in self.plan.order
                    ]
                )
                .sum(dim=0)
                .float()
            )
        # Sampled actions are detached. The score-function surrogate includes
        # the likelihood of the sampled prefixes.
        logp = self.select_and_score(False, generator).log_prob
        surrogate = -(logp.detach() + 1) * logp
        return -logp.detach() + surrogate - surrogate.detach()

    def kl(self, reference, generator=None):
        if self.actor.config.actions != reference.actor.config.actions:
            raise ValueError(
                "KL requires matching action measures and physical transforms"
            )
        if not self.plan.has_autoregression and not self.activity:
            current, frozen = self._parameters({}), reference._parameters({})
            values = []
            for name, field in self.actor.config.actions.items():
                p = self._field(name, current[name])
                q = reference._field(name, frozen[name])
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
        selected = self.select_and_score(False, generator)
        delta = selected.log_prob - reference.log_prob(selected.actions)
        surrogate = (delta.detach() + 1) * selected.log_prob
        return delta.detach() + surrogate - surrogate.detach()

    def domain_actions(self, actions):
        return [
            decode_action(action_record(actions, i), self.actor.config)
            for i in range(self.batch)
        ]
