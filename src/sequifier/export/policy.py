"""Portable action selection over the policy's existing decoder and binding."""

import json
import math

import torch
from torch import nn

from sequifier.contract_versions import ContractVersions
from sequifier.export.onnx import OnnxModelExporter
from sequifier.model.execution_schema import (
    MODEL_CONFIG_KEY,
    POLICY_GRAPH_KEY,
    policy_graph_schema,
)
from sequifier.model.prediction_distribution import RealPredictionDistribution
from sequifier.rl.action_plan import ActionDecision
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.joint_distribution import JointActionDistribution


def _real_selection(distribution, stochastic, uniform, normal):
    noise = torch.where(stochastic, normal, 0.0)
    choice = torch.where(stochastic, uniform, -1.0)
    value, latent, component = distribution.select_from_noise(noise, choice)
    log_prob = distribution.continuous_log_prob(latent)
    for atom_index, (_, probability) in enumerate(
        sorted(distribution.field.atoms.items())
    ):
        log_prob = torch.where(component == atom_index, math.log(probability), log_prob)
    return value, latent, component, log_prob


def _categorical_selection(distribution, stochastic, uniform):
    categorical = distribution.distribution
    # Exclude the last boundary so roundoff always selects a valid ID.
    boundaries = categorical.probs.double().cumsum(-1)[:, :-1]
    sample = (uniform[:, None] >= boundaries).long().sum(-1)
    value = torch.where(stochastic, sample, distribution.mode())
    log_prob = categorical.logits.gather(-1, value[:, None]).squeeze(-1)
    return value, log_prob


class PolicyGraph(nn.Module):
    """Select and score actions with caller-owned random draws, in dependency order."""

    def __init__(self, actor, inputs):
        super().__init__()
        self.actor = actor
        self.inputs = inputs

    def forward(self, *values):
        features, metadata = self.inputs.bind(values)
        representation = self.actor.network.encode(
            features, metadata, interface_name=self.actor.interface_name
        )
        joint = JointActionDistribution(
            self.actor,
            None,
            _encoded=(features, metadata, representation),
            exporting=True,
        )
        # Column zero selects stochastic execution. Each field owns a uniform
        # and a standard normal column in the binding's declared action order.
        draws = values[-1]
        stochastic = draws[:, 0] != 0
        decision = ActionDecision()
        fields = {}
        score = representation.new_zeros(
            (representation.shape[0],), dtype=torch.float64
        )
        for name, distribution in joint.field_distributions(decision):
            index = joint.plan.action_names.index(name)
            uniform = draws[:, 1 + 2 * index]
            if isinstance(distribution, RealPredictionDistribution):
                value, latent, component, log_prob = _real_selection(
                    distribution, stochastic, uniform, draws[:, 2 + 2 * index]
                )
                decision.latents[name] = latent
                decision.components[name] = component
            else:
                value, log_prob = _categorical_selection(
                    distribution, stochastic, uniform
                )
            active = joint.active_mask(name, decision)
            fields[name] = distribution, active
            if active is not None:
                value = torch.where(active, value, joint.plan.fields[name].inactive)
                log_prob = torch.where(active, log_prob, 0.0)
            decision.values[name] = value
            score = score + log_prob
        probabilities = joint.field_probabilities(fields)
        return (
            *(decision.values[name] for name in joint.plan.action_names),
            score.float(),
            *probabilities.values(),
            *(
                decision.latents[name]
                for name in joint.plan.action_names
                if name in decision.latents
            ),
            *(
                decision.components[name]
                for name in joint.plan.action_names
                if name in decision.components
            ),
        )


def export_policy_graphs(config, artifact, paths):
    for name, path in paths.items():
        with torch.random.fork_rng(devices=[]):
            actor = load_policy(artifact, interface_name=name)
        inputs, schema = policy_graph_schema(actor.interface, actor.config)
        contract = artifact.model_config.values
        interface_contract = dict(contract["interfaces"][name])
        # JSON object keys would conflate integer labels and their string spellings.
        interface_contract["id_maps"] = {
            column: list(labels.items())
            for column, labels in interface_contract["id_maps"].items()
        }
        metadata = {
            POLICY_GRAPH_KEY: json.dumps(
                {"interface": name, "version": ContractVersions.POLICY_ONNX}
            ),
            MODEL_CONFIG_KEY: json.dumps(
                {**contract, "interfaces": {name: interface_contract}}
            ),
        }
        OnnxModelExporter().export_graph(
            PolicyGraph(actor, inputs),
            schema,
            path,
            interface_name=name,
            batch_size=config.run.export.prediction.onnx.batch_size,
            training=False,
            metadata=metadata,
        )
