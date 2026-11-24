"""Resolve a phase's sequence inputs after its producer has committed."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.artifacts.dependencies import DependencyManifest
from sequifier.artifacts.phase import PhaseArtifacts, verify_artifact
from sequifier.config.composition import load_composed_yaml_config
from sequifier.config.train_config import (
    DatasetPartSpecModel,
    RolloutPhaseSource,
    SequencePartSpecModel,
)
from sequifier.fingerprints import fingerprint
from sequifier.runtime.plan import continuation_groups


def first_consumer(config):
    """Locate the first optimizer group that needs a produced dataset."""
    if not any(
        isinstance(source, RolloutPhaseSource)
        for dataset in config.datasets.values()
        for source in dataset.sources.values()
    ):
        return None
    from sequifier.config.training_resolution import _active_datasets
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    positions = {name: index for index, name in enumerate(config.plan)}
    inputs = PreparedTrainingInputs()
    root = Path(config.run.project_root).resolve()
    first = None
    for index, phases in continuation_groups(
        config.ordered_phases, group_supervised=True
    ):
        for phase in phases:
            datasets = _active_datasets(config, [phase], root, inputs)
            for name, dataset in datasets.items():
                for source in dataset.sources.values():
                    if not isinstance(source, RolloutPhaseSource):
                        continue
                    producer = config.phases.get(source.phase)
                    if positions.get(source.phase, index) >= index or not getattr(
                        getattr(producer, "collection", None), "output", None
                    ):
                        raise ValueError(
                            f"{name} requires a preceding collection phase with sequence output"
                        )
                    if (
                        phase.algorithm == "supervised"
                        and not source.preprocessing_config_path
                    ):
                        raise ValueError(
                            f"{name} requires preprocessing_config_path for supervised learning"
                        )
                    if first is None:
                        first = index
    return first


@dataclass
class PendingGroup:
    index: int
    phases: list
    artifacts: PhaseArtifacts

    @classmethod
    def planned(cls, config, start):
        phases = config._rollout_plan.ordered_phases[start:]
        return [
            cls(
                start + i,
                group,
                PhaseArtifacts.for_phase(
                    config.outputs.phase_directory, start + i, group[0].name
                ),
            )
            for i, group in continuation_groups(phases, group_supervised=True)
        ]

    def compile(self, parent, completed, incoming, prepared):
        from sequifier.config.training_resolution import (
            _active_datasets,
            resolve_training_run,
        )
        from sequifier.training.phase_runners import compile_phase_group

        authored = parent._rollout_plan.model_copy(deep=True)
        root = Path(parent.project_root)
        names = _active_datasets(authored, self.phases, root, prepared)
        dependencies = DependencyManifest()
        for name in sorted(names):
            for part, source in authored.datasets[name].sources.items():
                if not isinstance(source, RolloutPhaseSource):
                    continue
                try:
                    commit = completed[authored.plan.index(source.phase)]
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Uncommitted rollout producer {source.phase!r}"
                    ) from None
                provenance = commit.result["provenance"]
                key = (
                    "committed_sequence_manifest"
                    if "committed_sequence_manifest" in provenance
                    else "sequence_manifest"
                )
                path = Path(provenance[key])
                verify_artifact(
                    path,
                    provenance[key + "_sha256"],
                    "Committed rollout manifest changed",
                )
                from sequifier.config.policy_train_config import SequenceSourceConfig
                from sequifier.trajectory.sources import resolve_source

                data = resolve_source(
                    SequenceSourceConfig(manifest_path=str(path)), root
                )
                source_dependencies = data.dependencies
                if source.preprocessing_config_path:
                    config_path = root / source.preprocessing_config_path
                    source_dependencies.add(config_path)
                    metadata = prepare_source(
                        path, config_path, root, parent.seed, source_dependencies
                    )
                    spec = DatasetPartSpecModel(
                        type="prepared", metadata_config_path=str(metadata)
                    )
                else:
                    spec = SequencePartSpecModel(
                        type="sequence", manifest_path=str(path)
                    )
                dependencies.merge(source_dependencies)
                authored.datasets[name].sources[part] = spec
        authored.plan = [phase.name for phase in self.phases]
        authored.run.reproducibility.seed = parent.seed
        authored.initialize_from = incoming
        authored.actor_initialization = "attach"
        resolved = resolve_training_run(authored)
        from sequifier.config.naming import supervised_phase_seed

        for offset, phase in enumerate(resolved.training_plan):
            if phase.algorithm == "supervised":
                phase.resolved_seed = supervised_phase_seed(
                    parent.seed, self.index + offset
                )
        group = compile_phase_group(
            resolved, resolved.training_plan, self.index, prepared
        )
        return group, dependencies


def prepare_source(manifest, config_path, root, seed, dependencies):
    from sequifier.config.preprocess_config import PreprocessorModel
    from sequifier.config.sequence_import_config import PreparePolicyConfig
    from sequifier.io.prepare_policy import prepare_policy
    from sequifier.preprocess import Preprocessor

    input_paths = []
    values = load_composed_yaml_config(str(config_path), input_paths=input_paths)
    for path in input_paths:
        dependencies.add(path)
    run = values.setdefault("run", {})
    run["project_root"] = str(root)
    reproducibility = run.setdefault("reproducibility", {})
    if reproducibility.get("seed", "random") == "random":
        reproducibility["seed"] = seed
    for name in ("metadata_config_path", "initialize_from", "static_data_path"):
        if values.get(name):
            dependencies.add(root / values[name])
    if values.get("source"):
        for field in ("metadata_config_path", "static_data_path"):
            if values["source"].get(field):
                dependencies.add(root / values["source"][field])
    key = fingerprint(
        {"inputs": dependencies.identity, "seed": seed, "project_root": str(root)}
    )
    output = manifest.parent / (manifest.name + ".parts") / "prepared" / key
    if values.get("operation") == "prepare_policy":
        source = values.setdefault("source", {})
        for location in ("paths", "split"):
            source.pop(location, None)
        source["manifest_path"] = str(manifest)
        values.pop("indexed_output_path", None)
        values["output_path"] = str(output)
        prepare_policy(PreparePolicyConfig.model_validate(values))
    else:
        values.update(
            preprocessing_data_path=str(manifest),
            read_format="sequence",
            indexed_output_path=str(output),
        )
        Preprocessor(PreprocessorModel.model_validate(values)).run()
    return output / "metadata.json"
