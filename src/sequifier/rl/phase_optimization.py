"""Resolve policy optimizer roles and masks before executing a phase."""

from dataclasses import dataclass

from torch import nn

from sequifier.config.policy_optimization import OptimizationConfig
from sequifier.rl.models import HistoryCritic, critic_history_options, make_iql_models


@dataclass(frozen=True)
class PhaseRoles:
    trainable: dict[str, nn.Module]
    frozen: dict[str, nn.Module]

    @classmethod
    def build(cls, actor, phase, *, metadata_only=False, restoring=False):
        trainable, frozen = {"actor": actor.network}, {}
        if phase.algorithm in {"ppo", "iql"}:
            history = (
                {"metadata_only": True}
                if metadata_only
                else critic_history_options(actor, phase, restore=restoring)
            )
            if phase.algorithm == "ppo":
                trainable["value"] = HistoryCritic.ppo_value(actor, **history).eval()
            else:
                value, q1, q2, targets = make_iql_models(
                    actor.execution_config,
                    actor.interface_name,
                    actor.config.actions,
                    **history,
                )
                trainable.update(
                    value=value.eval(), critic=nn.ModuleList([q1, q2]).eval()
                )
                frozen = {
                    f"target_q{i}": target.eval() for i, target in enumerate(targets, 1)
                }
        return cls(trainable, frozen)


@dataclass(frozen=True)
class PhaseOptimization:
    step_budget: int | None
    configs: dict[str, OptimizationConfig]
    masks: dict[str, list[str]]

    @classmethod
    def resolve(cls, phase, roles, training_size):
        budget = phase.optimizer_step_budget(training_size)
        masks = {}
        for role, module in roles.trainable.items():
            frozen = phase.trainability.policy(role).resolve(module)
            masks[role] = sorted(
                name
                for name, parameter in module.named_parameters()
                if parameter.requires_grad and id(parameter) not in frozen
            )
            if not masks[role]:
                raise ValueError(f"{phase.name}: {role} leaves no trainable parameters")
        configs = {
            role: config.resolve_budget(budget)
            for role, config in phase.optimization_configs().items()
        }
        return cls(budget, configs, masks)

    def build(self, roles):
        if roles.trainable.keys() != self.configs.keys():
            raise ValueError("phase optimization roles disagree with trainable modules")
        for role, module in roles.trainable.items():
            parameters = dict(module.named_parameters())
            mask = set(self.masks[role])
            if mask - parameters.keys():
                raise ValueError(
                    f"{role}: trainable parameter names disagree with preflight"
                )
            for name, parameter in parameters.items():
                parameter.requires_grad_(name in mask)
        return {
            role: self.configs[role].build(module, step_budget=self.step_budget)
            for role, module in roles.trainable.items()
        }
