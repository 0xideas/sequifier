"""Run-owned source identities, episode catalogs and policy partitions."""

from contextlib import contextmanager
from dataclasses import dataclass

from sequifier.artifacts.dependencies import DependencyManifest
from sequifier.config.policy_config import PolicyBinding
from sequifier.config.policy_inputs import PolicyFittingSpec
from sequifier.fingerprints import fingerprint
from sequifier.io.episode_cache import EpisodeCache
from sequifier.io.memory_budget import resident_bytes
from sequifier.runtime.resource_accounting import ResourceAccounting
from sequifier.trajectory.sources import ResolvedSequenceSource, resolve_source


@dataclass(frozen=True)
class RunInputs:
    sources: dict[str, ResolvedSequenceSource]
    dependencies: DependencyManifest


def _data_policy_contract(policy):
    """Episode interpretation and eligibility, independent of fitted feature codecs."""
    binding = policy.binding if isinstance(policy, PolicyFittingSpec) else policy
    return {
        "binding": binding.model_dump(
            mode="json", include=set(PolicyBinding.model_fields)
        ),
        "context_length": policy.context_length,
        "interface_name": policy.interface_name,
        "observation_columns": policy.observation_columns,
        "static_columns": policy.static_columns,
        "action_columns": policy.action_columns,
        "input_encoding": policy.input_encoding,
    }


class PreparedTrainingInputs:
    """Private, in-memory preparation shared from metadata fitting through execution.

    Source interpretation keys episode records and split membership. Each
    execution codec owns its views and caches for the run. Fitting views are
    released as soon as metadata has been fitted.
    """

    def __init__(self):
        self._sources = {}
        self._baseline = DependencyManifest()
        self._inputs = {}
        self._episode_records = {}
        self._memberships = {}
        self._partitions = {}
        self._catalogs = {}
        self._preference_manifests = {}
        self._episode_caches = []
        self.accounting = ResourceAccounting()
        self._prepared_storage = self.accounting.allocation("prepared_data")
        self._running = False

    def episode_cache(self, capacity):
        cache = EpisodeCache(capacity, self.accounting.allocation("episode_caches"))
        self._episode_caches.append(cache)
        return cache

    def measure(self):
        roots = (
            self._inputs,
            self._sources,
            self._baseline,
            self._episode_records,
            self._memberships,
            self._partitions,
            self._catalogs,
            self._preference_manifests,
        )
        seen = {id(self), *(id(cache) for cache in self._episode_caches)}
        self._prepared_storage.resize(resident_bytes(roots, seen))

    def _clear_data(self):
        for cache in self._episode_caches:
            cache.clear()
        self._episode_caches.clear()
        self._episode_records.clear()
        self._memberships.clear()
        self._partitions.clear()
        self._catalogs.clear()
        self._preference_manifests.clear()

    def release_data(self):
        self._clear_data()
        self.measure()

    def close(self):
        self._clear_data()
        self._inputs.clear()
        self._sources.clear()
        # Keep the resolved config's checksum baseline for subsequent runs.
        self._prepared_storage.resize(0)

    @contextmanager
    def run_scope(self):
        if self._running:
            yield
            return
        self._running = True
        try:
            with self.accounting.activate():
                yield
        finally:
            self.close()
            self._running = False
            self.accounting = ResourceAccounting()
            self._prepared_storage = self.accounting.allocation("prepared_data")

    @classmethod
    def for_run(cls, config):
        if config._prepared_inputs is None:
            config._prepared_inputs = cls()
        return config._prepared_inputs

    def phase_dependencies(self, config, phase, inputs):
        refs = phase.data_sources
        if refs.kind == "preference":
            from sequifier.trajectory.preferences import load_preferences

            preferences = {ref.source: config.preferences[ref.source] for ref in refs}
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
            names = {ref.source for ref in refs}
        sources = {name: inputs.sources[name] for name in sorted(names)}
        dependencies = DependencyManifest()
        for source in sources.values():
            dependencies.merge(source.dependencies)
        dependencies.include_files(inputs.dependencies, preferences.values())
        return sources, preferences, dependencies

    def resolve(self, config):
        declarations = {
            name: {
                **source.model_dump(mode="json"),
                "split_index": source.selected_split_index,
            }
            for name, source in config.sources.items()
        }
        key = fingerprint({"sources": declarations, "preferences": config.preferences})
        if key not in self._inputs:
            sources = {}
            dependencies = DependencyManifest()
            for name, source in config.sources.items():
                source_key = fingerprint(declarations[name])
                if source_key not in self._sources:
                    self._sources[source_key] = resolve_source(source)
                resolved = self._sources[source_key]
                sources[name] = resolved
                dependencies.merge(resolved.dependencies)
            for path in config.preferences.values():
                dependencies.add(path)
            self._baseline.merge(dependencies)
            self._inputs[key] = RunInputs(sources, dependencies)
        return self._inputs[key]

    @contextmanager
    def fitting_phase(self, config, phase, policy, inputs):
        start = len(self._episode_caches)
        try:
            yield self._prepare_phase(config, phase, policy, inputs, fitting=True)
        finally:
            for cache in self._episode_caches[start:]:
                cache.clear()
            del self._episode_caches[start:]
            self.measure()

    def _prepare_phase(self, config, phase, policy, inputs, *, fitting=False):
        from sequifier.io.policy_dataset import PolicyDataCatalog
        from sequifier.rl.registry import BUILTIN_ALGORITHMS

        algorithm = BUILTIN_ALGORITHMS[phase.algorithm]
        prepare = (
            algorithm.prepare_for_fitting if fitting else None
        ) or algorithm.prepare_data
        if prepare is None:
            return None
        sources, preferences, dependencies = self.phase_dependencies(
            config, phase, inputs
        )
        catalog_key = fingerprint(
            {
                "sources": {
                    name: {
                        "source": source.source.model_dump(mode="json"),
                        "split_index": source.source.selected_split_index,
                        "paths": [str(path) for path in source.paths],
                        "kind": source.kind,
                        "checksums": source.dependencies.identity,
                    }
                    for name, source in sources.items()
                },
                "policy": _data_policy_contract(policy),
                "episode_cache_size": config.episode_cache_size,
            }
        )
        partition_key = fingerprint(
            {
                "catalog": catalog_key,
                "preferences": preferences,
                "identity": dependencies.identity,
                "source_validation_fractions": config.source_validation_fractions,
                "seed": config.seed,
            }
        )
        if fitting:
            catalog_views, partitions = {}, {}
            view_key = catalog_key
        else:
            codec_key = fingerprint(policy.model_dump(mode="json"))
            catalog_views = self._catalogs
            view_key = (catalog_key, codec_key)
            partitions = self._partitions.setdefault((partition_key, codec_key), {})
        if view_key not in catalog_views:
            catalog_views[view_key] = PolicyDataCatalog(
                sources,
                policy,
                config.episode_cache_size,
                preference_manifests=self._preference_manifests,
                episode_records=self._episode_records.setdefault(catalog_key, {}),
                cache_owner=self,
            )
        return prepare(
            config,
            phase,
            catalog_views[view_key],
            partitions,
            self._memberships.setdefault(partition_key, {}),
        )

    def prepare(self, config, phases, policy, inputs):
        data = []
        for phase in phases:
            try:
                data.append(self._prepare_phase(config, phase, policy, inputs))
            except (ValueError, KeyError, FileNotFoundError) as error:
                raise ValueError(f"phases.{phase.name}: {error}") from error
        self.measure()
        return data
