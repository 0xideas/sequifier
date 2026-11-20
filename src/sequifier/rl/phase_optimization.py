"""Resolve policy optimizer roles and masks before executing a phase."""

from dataclasses import dataclass, field

from torch import nn

from sequifier.config.policy_optimization import OptimizationConfig


@dataclass(frozen=True)
class PhaseRoles:
    trainable: dict[str, nn.Module]
    frozen: dict[str, nn.Module]
    fresh_history_components: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, actor, phase, *, metadata_only=False, restoring=False):
        from sequifier.rl.registry import BUILTIN_ALGORITHMS

        factory = BUILTIN_ALGORITHMS[phase.algorithm].build_roles
        if factory is None:
            return cls({"actor": actor.network}, {})
        return factory(actor, phase, metadata_only=metadata_only, restoring=restoring)


@dataclass(frozen=True)
class PhaseOptimization:
    step_budget: int | None
    configs: dict[str, OptimizationConfig]
    masks: dict[str, list[str]]
    fresh_history_components: dict[str, list[str]]

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
        return cls(budget, configs, masks, roles.fresh_history_components)

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
