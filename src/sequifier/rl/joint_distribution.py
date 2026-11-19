"""Joint sampling and recorded-prefix scoring over existing model decoders."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from sequifier.io.preprocessing import denormalize_values
from sequifier.model.prediction_distribution import RealPredictionDistribution
from sequifier.rl.action_plan import ActionDecision, ActionExecutionPlan
from sequifier.rl.distributions import CategoricalActionDistribution


@dataclass(frozen=True)
class ScoredSelection:
    decision: ActionDecision
    log_prob: Tensor
    probabilities: dict[str, Tensor]
    _plan: ActionExecutionPlan

    @property
    def actions(self) -> dict[str, Tensor]:
        return self._plan.to_record(self.decision)


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

    def _distributions(self, decision: ActionDecision, *, categorical_only=False):
        for name, output in self.plan.outputs(
            self.decoder, self.context, self.valid, self.statistics, decision
        ):
            if not categorical_only or name in self.support:
                yield name, self._field(name, output)

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
            self.route.prediction_scale(name)
            if field.scale == "model"
            else field.scale.fixed
        )
        return RealPredictionDistribution(location, scale, field)

    def _active(self, name, decision: ActionDecision):
        active = None
        for predecessor, allowed in self.activity.get(name, {}).items():
            if predecessor not in decision.values:
                raise ValueError(
                    "conditional activity requires a preceding action field"
                )
            matches = torch.isin(decision.values[predecessor], allowed)
            active = matches if active is None else active & matches
        return active

    def _select(self, deterministic, generator):
        decision, fields = ActionDecision(), {}
        for name, distribution in self._distributions(decision):
            value = (
                distribution.mode() if deterministic else distribution.sample(generator)
            )
            field = self.plan.fields[name]
            active = self._active(name, decision)
            decision.values[name] = (
                value
                if active is None
                else torch.where(active, value, value.new_tensor(field.inactive))
            )
            if isinstance(distribution, RealPredictionDistribution):
                decision.latents[name] = (
                    distribution.location
                    if deterministic
                    else distribution.sampled_latent
                )
                decision.components[name] = (
                    torch.full_like(value, -1, dtype=torch.long)
                    if deterministic
                    else distribution.sampled_component
                )
            fields[name] = distribution, active
        return decision, fields

    def select_decision(self, deterministic, generator) -> ActionDecision:
        decision, _ = self._select(deterministic, generator)
        return decision

    def select(self, deterministic, generator):
        return self.plan.to_record(self.select_decision(deterministic, generator))

    def select_and_score(
        self, deterministic, generator, *, include_probabilities=False
    ):
        """Score the selected action with the distributions used to select it."""
        decision, fields = self._select(deterministic, generator)
        return ScoredSelection(
            decision,
            self._score(decision, fields),
            self._probabilities(fields) if include_probabilities else {},
            self.plan,
        )

    def sample(self, generator=None):
        return self.select(False, generator)

    def mode(self):
        return self.select(True, None)

    def _score(self, decision: ActionDecision, fields):
        score = self.context.new_zeros(
            self.batch,
            dtype=torch.float32 if self.context.device.type == "mps" else torch.float64,
        )
        for name, (distribution, active) in fields.items():
            field = self.plan.fields[name]
            value = decision.values[name]
            if active is not None:
                if ((value != field.inactive) & ~active).any():
                    raise ValueError(
                        f"{name}: inactive action differs from its point mass"
                    )
                value = torch.where(active, value, distribution.mode().detach())
            if isinstance(distribution, RealPredictionDistribution):
                latent = decision.latents.get(name)
                component = decision.components.get(name)
                if active is not None:
                    if latent is not None:
                        latent = torch.where(
                            active, latent, distribution.location.detach()
                        )
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

    def _decision(
        self, actions: ActionDecision | Mapping[str, Tensor]
    ) -> ActionDecision:
        if isinstance(actions, ActionDecision):
            return actions
        if isinstance(actions, Mapping):
            return self.plan.from_record(actions)
        raise ValueError("joint likelihood requires named action fields")

    def log_prob(self, actions: ActionDecision | Mapping[str, Tensor]):
        decision = self._decision(actions)
        if self.plan.fields.keys() - decision.values.keys():
            raise ValueError("joint likelihood requires every action field")
        if decision.values.keys() - self.plan.fields.keys():
            raise ValueError("joint action contains unknown fields")
        for name in self.plan.order:
            if decision.values[name].shape != (self.batch,):
                raise ValueError(f"{name}: expected one action per decision")
        return self._score(
            decision,
            {
                name: (distribution, self._active(name, decision))
                for name, distribution in self._distributions(decision)
            },
        )

    def _probabilities(self, fields):
        result = {}
        for name in self.plan.action_names:
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
        decision = self._decision(actions)
        return self._probabilities(
            {
                name: (distribution, self._active(name, decision))
                for name, distribution in self._distributions(
                    decision, categorical_only=True
                )
            }
        )

    @property
    def entropy_is_exact(self):
        return not self.plan.has_autoregression and all(
            not field.active_when
            and field.lower is None
            and field.upper is None
            and not field.atoms
            for field in (action.spec for action in self.plan.fields.values())
        )

    def entropy(self, generator=None):
        if self.entropy_is_exact:
            return (
                torch.stack(
                    [
                        distribution.entropy()
                        for _, distribution in self._distributions(ActionDecision())
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
            current = dict(self._distributions(ActionDecision()))
            frozen = dict(reference._distributions(ActionDecision()))
            values = []
            for name, field in self.actor.config.actions.items():
                p, q = current[name], frozen[name]
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
        delta = selected.log_prob - reference.log_prob(selected.decision)
        surrogate = (delta.detach() + 1) * selected.log_prob
        return delta.detach() + surrogate - surrogate.detach()

    def domain_actions(self, actions):
        decision = self._decision(actions).cpu()
        return [self.plan.domain_action(decision, i) for i in range(self.batch)]
