"""Episode readers for resolved tables, prepared histories, and rollout logs."""

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.config.policy_inputs import PolicyEpisodeSpec
from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.io.sequence import ContextObservation
from sequifier.trajectory.sources import ResolvedSequenceSource, resolve_source
from sequifier.trajectory.tables import table_episodes


def trajectory_seed_context(seed, execution):
    seed_actions = seed.valid_actions
    seed_returns = execution.get("seed_returns") or [None] * len(seed_actions)
    context = tuple(
        ContextObservation(
            observation={
                name: values[i - len(seed_actions)]
                for name, values in seed.observations.items()
                if name != INCOMING_ACTION_PRESENT
            },
            masked=seed.observation_mask[i - len(seed_actions)],
            previous_action=action,
            desired_return=seed_returns[i],
        )
        for i, action in enumerate(seed_actions[:-1])
    )
    return context, seed_actions[-1]


def source_episodes(
    source: SequenceSourceConfig | ResolvedSequenceSource, policy: PolicyEpisodeSpec
):
    resolved: ResolvedSequenceSource = (
        resolve_source(source) if isinstance(source, SequenceSourceConfig) else source
    )
    source_config = resolved.declaration
    policy.validate_source_units(source_config.units)
    provenance = (resolved.manifest or {}).get("provenance", {})
    policy.validate_source_units(provenance.get("policy_binding", {}).get("units"))
    return table_episodes(resolved, policy)
