"""Run-owned source identities, episode catalogs and policy partitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sequifier.config.policy_config import PolicyBinding
from sequifier.config.policy_inputs import PolicyFittingSpec
from sequifier.fingerprints import fingerprint
from sequifier.trajectory.sources import (
    ResolvedSequenceSource,
    file_checksum,
    resolve_source,
)

if TYPE_CHECKING:
    from sequifier.io.policy_dataset import PolicyDataCatalog
    from sequifier.io.prepared_policy_data import PreparedPolicyData


@dataclass(frozen=True)
class RunInputs:
    sources: dict[str, ResolvedSequenceSource]
    checksums: dict[Path, str]
    _identity: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_identity",
            {str(path.resolve()): value for path, value in self.checksums.items()},
        )

    @classmethod
    def resolve(cls, config):
        return PreparedTrainingInputs.for_run(config).resolve(config)

    @property
    def identity(self):
        return dict(self._identity)

    def revalidate(self, expected, field):
        """Rehash every dependency, including manifests and retargeted links."""
        from sequifier.rl.run_state import require_equal

        require_equal(
            expected,
            {str(path.resolve()): file_checksum(path) for path in self.checksums},
            field,
        )


def _data_policy_contract(policy):
    """Episode interpretation and eligibility, independent of fitted feature codecs."""
    binding = policy.binding if isinstance(policy, PolicyFittingSpec) else policy
    values = binding.model_dump(mode="json")
    return {
        "binding": {key: values[key] for key in PolicyBinding.model_fields},
        "context_length": policy.context_length,
        "interface_name": policy.interface_name,
        "observation_columns": policy.observation_columns,
        "static_columns": policy.static_columns,
        "action_columns": policy.action_columns,
        "input_encoding": policy.input_encoding,
    }


@dataclass
class _PolicyPartitions:
    catalog: "PolicyDataCatalog"
    prepared: dict[tuple, "PreparedPolicyData"] = field(default_factory=dict)

    def rebind(self, policy):
        self.catalog.rebind(policy)
        for data in self.prepared.values():
            data.rebind(policy)


class PreparedTrainingInputs:
    """Private, in-memory preparation shared from metadata fitting through execution.

    Source interpretation and split settings key reusable partitions. A changed
    policy interpretation builds new views; changed codecs refresh existing views
    without selecting the partitions again. Callers still revalidate dependencies
    before executing a run and at every committed phase boundary.
    """

    def __init__(self):
        self._sources = {}
        self._checksums = {}
        self._inputs = {}
        self._partitions = {}
        self._preference_manifests = {}

    @classmethod
    def for_run(cls, config):
        if getattr(config, "_prepared_inputs", None) is None:
            config._prepared_inputs = cls()
        return config._prepared_inputs

    def _phase_dependencies(self, config, phase, inputs):
        if phase.algorithm == "dpo":
            from sequifier.trajectory.preferences import load_preferences

            preferences = {
                ref.source: config.preferences[ref.source]
                for ref in [phase.preferences, *(phase.evaluation.sources or [])]
            }
            names = set()
            for path in preferences.values():
                if path not in self._preference_manifests:
                    self._preference_manifests[path] = load_preferences(path).pairs
                for pair in self._preference_manifests[path]:
                    names.update((pair.chosen.source, pair.rejected.source))
            if names - inputs.sources.keys():
                raise ValueError(
                    f"preferences reference unknown sources: {sorted(names - inputs.sources.keys())}"
                )
        else:
            preferences = {}
            names = {
                ref.source
                for ref in [*phase.sources, *(phase.evaluation.sources or [])]
            }
        sources = {name: inputs.sources[name] for name in sorted(names)}
        checksums = {
            str(path): value
            for source in sources.values()
            for path, value in source.checksums.items()
        }
        checksums.update(
            {path: inputs.checksums[Path(path)] for path in preferences.values()}
        )
        return sources, preferences, checksums

    def resolve(self, config):
        declarations = {
            name: source.model_dump(mode="json")
            for name, source in config.sources.items()
        }
        key = fingerprint({"sources": declarations, "preferences": config.preferences})
        if key not in self._inputs:
            sources, checksums = {}, {}
            for name, source in config.sources.items():
                source_key = fingerprint(declarations[name])
                if source_key not in self._sources:
                    self._sources[source_key] = resolve_source(source)
                resolved = self._sources[source_key]
                sources[name] = resolved
                for path, checksum in resolved.checksums.items():
                    if path in self._checksums and self._checksums[path] != checksum:
                        raise ValueError(f"source changed during preparation: {path}")
                    self._checksums[path] = checksum
                    checksums[path] = checksum
            for path in config.preferences.values():
                path = Path(path)
                if path not in self._checksums:
                    self._checksums[path] = file_checksum(path)
                checksums[path] = self._checksums[path]
            self._inputs[key] = RunInputs(sources, checksums)
        return self._inputs[key]

    def prepare_phase(self, config, phase, policy, inputs, *, fitting=False):
        from sequifier.io.policy_dataset import PolicyDataCatalog
        from sequifier.rl.offline_data import prepare_offline
        from sequifier.rl.registry import BUILTIN_ALGORITHMS

        entry = BUILTIN_ALGORITHMS[phase.algorithm]
        prepare = (
            prepare_offline
            if fitting and phase.algorithm == "iql"
            else entry.prepare_data
        )
        if prepare is None:
            return None
        sources, preferences, checksums = self._phase_dependencies(
            config, phase, inputs
        )
        key = fingerprint(
            {
                "sources": {
                    name: {
                        "source": source.source.model_dump(mode="json"),
                        "paths": [str(path) for path in source.paths],
                        "kind": source.kind,
                    }
                    for name, source in sources.items()
                },
                "identity": checksums,
                "preferences": preferences,
                "policy": _data_policy_contract(policy),
                "validation_fraction": config.validation_fraction,
                "source_validation_fractions": getattr(
                    config, "source_validation_fractions", {}
                ),
                "seed": config.seed,
                "episode_cache_size": config.episode_cache_size,
            }
        )
        if key not in self._partitions:
            self._partitions[key] = _PolicyPartitions(
                PolicyDataCatalog(
                    sources,
                    policy,
                    config.episode_cache_size,
                    preference_manifests=self._preference_manifests,
                )
            )
        partitions = self._partitions[key]
        partitions.rebind(policy)
        return prepare(config, phase, policy, partitions.catalog, partitions.prepared)

    def prepare(self, config, policy, inputs):
        return [
            self.prepare_phase(config, phase, policy, inputs) for phase in config.phases
        ]
