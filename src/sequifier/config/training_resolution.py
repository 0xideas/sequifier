"""Resolve the common authored plan into model, dataset and task contracts."""

from dataclasses import dataclass
from pathlib import Path

from sequifier.config.metadata import (
    DatasetMetadata,
    dataset_tensor_kind,
    dataset_tensor_version,
    load_dataset_metadata,
)
from sequifier.config.naming import resolve_run_seed
from sequifier.config.policy_train_config import (
    PolicyDataSettings,
    SequenceSourceConfig,
    resolve_sequence_source,
)
from sequifier.config.run_capabilities import publication_plan, validate_runner_settings
from sequifier.config.train_config import (
    DatasetPartSpecModel,
    OwnedModelSpec,
    PreferencePartSpecModel,
    ResolvedInterfaceStorage,
    ResolvedSequifierConfig,
    SequencePartSpecModel,
    SequifierConfig,
    TrainingRuntimeSettings,
    _assert_compatible,
    _interface_semantics,
    _part_features,
    _resolve_interface,
    resolve_phases,
    resolve_supervised_datasets,
    validate_training_contract,
)
from sequifier.io.indexed_sequences import IndexedView
from sequifier.special_tokens import SPECIAL_TOKEN_IDS


def _normalization(config, name):
    if config.model and name in config.model.interfaces:
        return config.model.interfaces[name].input_contract.normalization
    return None


@dataclass(frozen=True)
class TrainingPart:
    spec: DatasetPartSpecModel | SequencePartSpecModel | PreferencePartSpecModel
    interface: str
    metadata: DatasetMetadata | None
    metadata_path: Path | None
    indexed: IndexedView | None


class TrainingSources:
    """Resolve named parts and share their metadata throughout compilation."""

    def __init__(self, config, root, supplied):
        self.root = root
        self.datasets = config.dataset_training
        self.parts = {}
        self.validation_fractions = {}
        self._views = {}
        self._metadata = {}
        for name, dataset in self.datasets.items():
            for part_name, spec in dataset.parts.items():
                ref = f"{name}.{part_name}"
                view = metadata = path = None
                if (
                    isinstance(spec, DatasetPartSpecModel)
                    and spec.type == "indexed_sequence"
                ):
                    view = self.indexed_view(spec.path)
                    path = view.metadata_path
                    metadata = view.metadata()
                elif (
                    not isinstance(spec, PreferencePartSpecModel)
                    and spec.metadata_config_path
                ):
                    path = root / spec.metadata_config_path
                    metadata = (
                        supplied.get(ref)
                        if supplied is not None
                        and isinstance(spec, DatasetPartSpecModel)
                        else self.read_metadata(path)
                    )
                self.parts[ref] = TrainingPart(
                    spec, dataset.model_interface, metadata, path, view
                )
                self.validation_fractions[ref] = dataset.split.validation_fraction

    def indexed_view(self, path):
        path = (self.root / path).absolute()
        if path not in self._views:
            view = IndexedView.load(path)
            if view.view["view"] == "predict":
                raise ValueError(
                    f"{path}: training requires a split with recorded membership"
                )
            view.verify()
            self._views[path] = view
        return self._views[path]

    def read_metadata(self, path):
        path = (self.root / path).resolve()
        if path not in self._metadata:
            self._metadata[path] = load_dataset_metadata(str(path))
        return self._metadata[path]

    def interface_metadata(self, config, declarations):
        """Check feature semantics once across every source of each model interface."""
        metadata = dict(declarations)
        signatures = {}

        def signature(value, spec, require_table):
            features = _part_features(value, spec, require_static_table=require_table)
            contract = features.signature()
            if not features.normalize_real_columns:
                contract["selected_columns_statistics"] = {}
            return {**contract, **value.raw_input_contract(features.column_data_types)}

        for ref, part in self.parts.items():
            value = part.metadata
            if value is None:
                continue
            name = part.interface
            spec = config.model.interfaces[name]
            prepared = isinstance(part.spec, DatasetPartSpecModel)
            try:
                actual = signature(value, spec, prepared)
                if name in declarations and name not in signatures:
                    signatures[name] = signature(declarations[name], spec, False)
            except ValueError as error:
                raise ValueError(f"{ref}: {error}") from error
            if name in signatures:
                _assert_compatible(signatures[name], actual, ref)
            else:
                signatures[name] = actual
            metadata.setdefault(name, value)
        return metadata

    def expand(self, reference):
        name, _, part = reference.partition(".")
        return (
            [reference]
            if part
            else [f"{name}.{key}" for key in self.datasets[name].parts]
        )

    def policy_sources(self):
        sequences, preferences = {}, {}
        for ref, record in self.parts.items():
            part = record.spec
            if isinstance(part, PreferencePartSpecModel):
                path = self.root / part.manifest_path
                preferences[ref] = str(path.resolve())
                continue
            if isinstance(part, SequencePartSpecModel):
                source = SequenceSourceConfig.model_validate(
                    part.model_dump(exclude={"type"})
                )
            elif (view := record.indexed) is not None:
                source = SequenceSourceConfig(
                    paths=[str(view.path)],
                    metadata_config_path=str(view.metadata_path),
                    format="prepared",
                    encoding="model",
                )
            else:
                source = SequenceSourceConfig(
                    metadata_config_path=part.metadata_config_path,
                    split_index=0,
                    format="prepared",
                    encoding="model",
                )
            sequences[ref] = resolve_sequence_source(
                source, self.root, metadata=record.metadata
            )
        return sequences, preferences


def _resolve_phase_sources(parts, phase, sequences, preferences):
    """Expand named sources for both metadata fitting and final run resolution."""
    refs = phase.data_sources
    if refs.kind == "none":
        return
    available = preferences if refs.kind == "preference" else sequences

    def expand(selected):
        expanded = [
            ref.model_copy(update={"source": full})
            for ref in selected
            for full in parts.expand(ref.source)
        ]
        if not expanded or any(ref.source not in available for ref in expanded):
            raise ValueError(f"{phase.name}: select {refs.kind} parts")
        return expanded

    expanded = expand(refs.training)
    if refs.kind == "preference":
        if len(expanded) != 1:
            raise ValueError(f"{phase.name}: select one training preference part")
        phase.preferences = expanded[0]
    else:
        phase.sources = expanded
    if phase.evaluation.sources is not None:
        phase.evaluation.sources = expand(refs.evaluation)


def phase_interface(config, phase):
    names = {
        config.dataset_training[ref.source.partition(".")[0]].model_interface
        for ref in phase.data_sources
    }
    explicit = phase.model_interface if phase.algorithm != "supervised" else None
    if explicit is not None:
        names.add(explicit)
    if not names:
        names = set(config.policy_bindings) or set(config.model.interfaces)
    if len(names) != 1 or None in names:
        raise ValueError(
            f"training_plan.phases.{phase.name}: select one model_interface"
        )
    name = next(iter(names))
    if name not in config.model.interfaces:
        raise ValueError(f"unknown model interface {name!r}")
    return name


def _inferred_metadata(config, name, binding, settings):
    """Identity numeric inputs and explicitly declared categorical action codecs."""
    interface = config.model.interfaces[name]
    static = (
        interface.static_conditioning.columns if interface.static_conditioning else []
    )
    inputs = set(interface.input_columns + static)
    columns = inputs | set(interface.target_columns)
    types = dict.fromkeys(columns, "Float32")
    counts, maps = {}, {}
    for column, action in binding.actions.items():
        if action.type == "categorical":
            if not action.values:
                raise ValueError(
                    f"{column}: fresh categorical targets need values or metadata"
                )
            types[column] = "Int64"
            counts[column] = SPECIAL_TOKEN_IDS.user_start + len(action.values)
            maps[column] = {
                str(label): SPECIAL_TOKEN_IDS.user_start + index
                for index, label in enumerate(action.values)
            }
    for column, action in binding.previous_action_inputs.items():
        if action not in binding.actions:
            raise ValueError(
                f"{column}: previous_action_inputs references an unknown action"
            )
        if action in maps:
            labels = list(binding.actions[action].values)
            initial = binding.initial_inputs.get(column)
            if initial is not None and initial not in labels:
                labels.append(initial)
            maps[column] = {
                str(label): SPECIAL_TOKEN_IDS.user_start + index
                for index, label in enumerate(labels)
            }
            counts[column] = SPECIAL_TOKEN_IDS.user_start + len(labels)
            types[column] = "Int64"
    normalization = _normalization(config, name)
    statistics = (
        normalization.statistics
        if normalization and normalization.source == "statistics"
        else {}
    )
    return DatasetMetadata(
        window_length=settings.context_length + max(1, settings.target_offset),
        column_data_types=types,
        static_columns=static,
        n_classes=counts,
        id_maps=maps,
        selected_columns_statistics={
            column: statistics.get(column, {"mean": 0.0, "std": 1.0})
            for column in columns
            if column not in counts
            and (normalization is not None or column not in inputs)
        },
    )


def _history_initialized_metadata(config, name, binding, source, storage, settings):
    """Resolve new targets while sourcing observation preprocessing from the artifact."""
    metadata = _inferred_metadata(config, name, binding, settings)
    static = source.static_conditioning
    inputs = set(source.input_columns + (static.columns if static else []))
    targets = set(config.model.interfaces[name].target_columns)
    for key, values, original in (
        ("column_data_types", metadata.column_data_types, source.column_data_types),
        ("n_classes", metadata.n_classes, source.n_classes),
        ("id_maps", metadata.id_maps, source.id_maps),
        (
            "selected_columns_statistics",
            metadata.selected_columns_statistics,
            source.selected_columns_statistics,
        ),
    ):
        for column in inputs:
            if column in targets and column in values and column in original:
                if (
                    key != "selected_columns_statistics"
                    and values[column] != original[column]
                ):
                    raise ValueError(
                        f"history transfer differs at {name}.{key}.{column}"
                    )
            values.pop(column, None)
            if column in original:
                values[column] = original[column]
    tensor_kind = (
        storage.tensor_payload_kind
        if storage
        else dataset_tensor_kind(source.depth_layouts)
    )
    return metadata.model_copy(
        update={
            **(storage.raw_input_contract(inputs) if storage else {}),
            "static_columns": static.columns if static else [],
            "static_features_path": storage.static_features_path if storage else None,
            "image_sources": source.image_sources,
            "preconditioning_image": source.preconditioning_image,
            "special_token_ids": source.special_token_ids,
            "depth_layouts": source.depth_layouts,
            "tensor_payload_kind": tensor_kind,
            "tensor_payload_version": (
                storage.tensor_payload_version
                if storage
                else dataset_tensor_version(tensor_kind)
            ),
            "normalize_real_columns": source.normalize_real_columns,
        }
    )


def _fit_input_metadata(
    config,
    name,
    binding,
    metadata,
    data_settings,
    prepared_inputs,
    settings,
):
    """Fit missing raw codecs/moments on the same episode-safe training split."""
    from sequifier.config.policy_inputs import PolicyFittingSpec, PolicyInputLayout

    phase = next(
        (
            phase
            for phase in config.training_plan.phases
            if phase.algorithm != "supervised" and phase.model_interface == name
        ),
        None,
    )
    if phase is None or phase.source_kind == "none":
        return metadata
    interface = config.model.interfaces[name]
    normalization = _normalization(config, name)
    static = (
        interface.static_conditioning.columns if interface.static_conditioning else []
    )
    layout = PolicyInputLayout.for_fitting(
        interface.input_columns, static, binding, metadata
    )
    observations, categorical = layout.observation_columns, layout.categorical_columns
    missing_codecs = set(categorical) - metadata.n_classes.keys()
    missing_statistics = (
        set(observations + static)
        - set(categorical)
        - metadata.selected_columns_statistics.keys()
    )
    if not missing_codecs and (not missing_statistics or normalization is not None):
        return metadata
    provisional = PolicyFittingSpec(
        **layout.binding_values(),
        binding=binding,
        context_length=settings.context_length,
        interface_name=name,
    )
    available = (
        data_settings.preferences
        if phase.source_kind == "preference"
        else data_settings.sources
    )
    if any(ref.source not in available for ref in phase.data_sources):
        return metadata  # Prepared dataset parts already own frozen preprocessing.
    inputs = prepared_inputs.resolve(data_settings)
    inputs.revalidate(inputs.identity, f"interfaces.{name}.fitting_sources")
    from sequifier.io.episode_features import EpisodeFeatures

    with prepared_inputs.fitting_phase(
        data_settings, phase, provisional, inputs
    ) as data:
        fitting = data.fitting_dataset()
        fitted = fitting.fit_categorical_config()
        values = metadata.model_dump()
        values["n_classes"].update(fitted.categorical_n_classes)
        values["id_maps"].update(fitted.id_maps)
        if (
            normalization is None
            and not metadata.depth_layouts
            and not metadata.image_sources
        ):
            features = EpisodeFeatures(fitting.episodes, fitted, fitting.lengths)
            features.validate_features()
            for column, statistics in features.normalization_statistics().items():
                values["selected_columns_statistics"].setdefault(column, statistics)
    return DatasetMetadata.model_validate(values)


def _validate_indexed_phase_coverage(sources, phases):
    def indexed_parts(refs):
        for ref in refs:
            for name in sources.expand(ref.source):
                if (view := sources.parts[name].indexed) is not None:
                    yield name, view

    supervised_training = [
        (name, view)
        for phase in phases
        if phase.algorithm == "supervised"
        for name, view in indexed_parts(phase.data_sources.training)
    ]
    for phase in phases:
        refs = phase.data_sources
        if phase.algorithm == "supervised" or refs.kind != "sequence":
            continue
        training = list(indexed_parts(refs.training))
        if training and not refs.evaluation:
            raise ValueError(
                f"{phase.name}: indexed policy data requires explicit evaluation.sources "
                "selecting a recorded held-out view"
            )
        for name, held_out in indexed_parts(refs.evaluation):
            held_keys = held_out.selection_keys()
            candidates = [*supervised_training, *training]
            for candidate_name, candidate in candidates:
                keys = candidate.selection_keys()
                if overlap := next(
                    (
                        field
                        for field, values in keys.items()
                        if values & held_keys[field]
                    ),
                    None,
                ):
                    raise ValueError(
                        f"{phase.name}: {candidate_name} overlaps held-out {name} at {overlap}"
                    )
                fit_splits = getattr(candidate.metadata(), "fit_split_indices", [])
                if (
                    candidate.manifest_path == held_out.manifest_path
                    and int(held_out.view["view"]) in fit_splits
                ):
                    raise ValueError(
                        f"{phase.name}: the frozen metadata was fitted on held-out {name}"
                    )


def resolve_training_run(
    authored: SequifierConfig,
    *,
    metadata_by_part: dict[str, DatasetMetadata] | None = None,
    loaded_metadata: dict[str, DatasetMetadata] | None = None,
) -> ResolvedSequifierConfig:
    """Compile the authored run into the records consumed by both runners."""
    from sequifier.artifacts.model_artifact import read_execution_artifact
    from sequifier.artifacts.model_config import model_execution_from_model_config
    from sequifier.model.execution import ModelExecution
    from sequifier.model.execution_options import ModelExecutionOptions
    from sequifier.model.history_transfer import validate_history_transfer
    from sequifier.rl.attachment import apply_binding, resolve_binding
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared_inputs = PreparedTrainingInputs()
    config = authored.model_copy(deep=True)
    phases = config.training_plan.phases
    supervised = [phase for phase in phases if phase.algorithm == "supervised"]
    has_policy = len(supervised) != len(phases)
    validate_runner_settings(config.run, supervised=bool(supervised), policy=has_policy)
    seed = config.run.reproducibility.seed
    if seed == "random":
        from sequifier.artifacts.run_checkpoint import read_resume_seed

        saved_seed = read_resume_seed(config.run)
        seed = resolve_run_seed(seed) if saved_seed is None else saved_seed
        config.run.reproducibility.seed = seed
    needs_policy_contracts = has_policy or bool(
        config.policy_bindings or config.initialize_from
    )
    supervised_names = {
        ref.source.partition(".")[0]
        for phase in supervised
        for ref in phase.data_sources
    }
    if not needs_policy_contracts:
        supervised_names.update(config.dataset_training)
    if (
        len(supervised_names) > 1
        and config.run.execution.distribution.strategy == "fsdp"
    ):
        raise ValueError("Multi-dataset training does not support FSDP")
    root = Path(config.run.project_root).resolve()
    if needs_policy_contracts:
        config.run.project_root = str(root)
    training = config.build_runtime_settings()
    source = None
    history_only = config.actor_initialization == "history_only"
    if config.initialize_from:
        config.initialize_from = str((root / config.initialize_from).resolve())
        source, _ = model_execution_from_model_config(
            read_execution_artifact(config.initialize_from).model_config.values,
            device=config.run.execution.device,
        )
        if config.model is not None and not history_only:
            from sequifier.artifacts.model_config import extend_prediction_distributions

            extend_prediction_distributions(source, config.model)
        if not history_only:
            config.model = OwnedModelSpec.from_execution(source, config.model)
        incoming = source.options
        resolved_fields = dict(training)
        inherited = {}
        for field, actual in (
            ("context_length", incoming.context_length),
            ("training_objective", incoming.training_objective),
            ("target_offset", incoming.target_offset),
            ("forward_attention", incoming.forward_attention),
        ):
            if (
                history_only
                and field in {"training_objective", "target_offset"}
                and field in training.model_fields_set
            ):
                continue
            if field in training.model_fields_set and resolved_fields[field] != actual:
                raise ValueError(
                    f"model.execution.{field} differs from initialize_from"
                )
            inherited[field] = actual
        training = training.model_copy(update=inherited)
        source_dtypes = source.options.layer_type_dtypes
        if (
            "layer_type_dtypes" in config.run.execution.precision.model_fields_set
            and training.layer_type_dtypes != source_dtypes
        ):
            raise ValueError(
                "run.execution.precision.layer_type_dtypes differs from initialize_from"
            )
        training.layer_type_dtypes = source_dtypes
        if "max_concat_tensors" not in config.run.execution.model_fields_set:
            training.max_concat_tensors = source.options.max_concat_tensors
    assert config.model is not None
    assert config.model.backbone is not None
    for dataset_name, dataset in config.dataset_training.items():
        if dataset.model_interface is None:
            if len(config.model.interfaces) != 1:
                raise ValueError("dataset model_interface is required when ambiguous")
            dataset.model_interface = next(iter(config.model.interfaces))
        if dataset.model_interface not in config.model.interfaces:
            raise ValueError(
                f"Dataset {dataset_name!r} references unknown model interface "
                f"{dataset.model_interface!r}."
            )
    if training.context_length is None:
        training.context_length = config.model.backbone.architecture.max_context_length
    if training.training_objective is None:
        if supervised:
            raise ValueError(
                "supervised phases require model.execution.training_objective"
            )
        training.training_objective = "policy"
        training.target_offset = 0
    training = TrainingRuntimeSettings.model_validate(training.model_dump())

    sources = TrainingSources(config, root, metadata_by_part)
    metadata_paths = {
        name: str(root / path)
        for name, spec in config.model.interfaces.items()
        if (path := spec.input_contract.metadata_config_path)
    }
    declarations = {
        name: sources.read_metadata(path) for name, path in metadata_paths.items()
    }
    if loaded_metadata is not None:
        loaded_metadata.update(
            {
                ref: part.metadata
                for ref, part in sources.parts.items()
                if part.metadata is not None
            }
        )

    source_fractions = sources.validation_fractions
    sequences, preferences = (
        sources.policy_sources() if needs_policy_contracts else ({}, {})
    )
    policy_data = PolicyDataSettings(
        sources=sequences,
        preferences=preferences,
        source_validation_fractions=source_fractions,
        seed=seed,
        episode_cache_size=training.episode_cache_size,
    )
    fitting_sequences = {
        ref: sequence
        for ref, sequence in sequences.items()
        if isinstance(sources.parts[ref].spec, SequencePartSpecModel)
    }

    fitting_data = policy_data.model_copy(update={"sources": fitting_sequences})
    for phase in phases:
        if phase.algorithm == "supervised":
            continue
        phase.optimization = phase.optimization_configs()
        phase.model_interface = phase_interface(config, phase)
        _resolve_phase_sources(sources, phase, sequences, preferences)
        reference = phase.reference_config
        if reference and reference.path:
            reference.path = str((root / reference.path).resolve())
        archive = phase.rollout_archive_directory
        if archive:
            phase.archive_directory = str((root / archive).resolve())

    if needs_policy_contracts:
        _validate_indexed_phase_coverage(sources, phases)

    from sequifier.runtime.plan import validate_continuation

    for index, phase in enumerate(phases):
        validate_continuation(
            phases[index - 1] if index else None, phase, resolved=True
        )

    resolved_phases = resolve_phases(phases, config.dataset_training, seed=seed)
    validate_training_contract(config, resolved_phases, training)

    metadata = sources.interface_metadata(config, declarations)
    datasets, dataset_interfaces, dataset_storage = resolve_supervised_datasets(
        config,
        sources,
        resolved_phases,
        supervised_names,
        training,
        declarations,
        metadata_from_file=metadata_by_part is None,
    )

    interfaces = {}
    storage = {}
    history_interface = phases[0].model_interface if history_only else None
    if history_only:
        assert source is not None
        if history_interface not in source.interfaces:
            raise ValueError(
                f"history transfer requires interface {history_interface!r}"
            )
    interface_order = dict.fromkeys(
        config.model.interfaces
        if needs_policy_contracts
        else [*dataset_interfaces, *config.model.interfaces]
    )
    for name in interface_order:
        spec = config.model.interfaces[name]
        value = metadata.get(name)
        binding = config.policy_bindings.get(name)
        if source is not None and not history_only:
            interface = source.interface(name)
            selected_storage = source.interface_storage.get(name)
            if value is not None:
                candidate = _resolve_interface(name, spec, value, training)
                _assert_compatible(interface.signature(), candidate.signature(), name)
                _assert_compatible(
                    interface.target_decoder_ids, candidate.target_decoder_ids, name
                )
                if selected_storage is not None:
                    columns = interface.column_data_types
                    _assert_compatible(
                        selected_storage.raw_input_contract(columns),
                        value.raw_input_contract(columns),
                        name,
                    )
        elif name in dataset_interfaces and not needs_policy_contracts:
            interface = dataset_interfaces[name]
            selected_storage = dataset_storage[name]
        else:
            if name == history_interface and value is None:
                assert source is not None
                if binding is None:
                    raise ValueError(f"{name}: history_only requires a policy binding")
                value = _history_initialized_metadata(
                    config,
                    name,
                    binding,
                    source.interface(name),
                    source.interface_storage.get(name),
                    training,
                )
            if value is None:
                if binding is None:
                    raise ValueError(f"{name}: model interface requires metadata")
                value = _inferred_metadata(config, name, binding, training)
            if binding is not None and name != history_interface:
                value = _fit_input_metadata(
                    config,
                    name,
                    binding,
                    value,
                    fitting_data,
                    prepared_inputs,
                    training,
                )
            interface = _resolve_interface(name, spec, value, training)
            selected_storage = dataset_storage.get(name)
            if selected_storage is None:
                selected_storage = ResolvedInterfaceStorage.from_metadata(value)
            from sequifier.helpers import resolve_window_view

            resolve_window_view(selected_storage.storage_layout, interface.window_view)
        if name in dataset_interfaces and interface is not dataset_interfaces[name]:
            canonical = _interface_semantics(interface)
            for key in ("window_view", "policy", "prediction_distributions"):
                canonical.pop(key)
            _assert_compatible(
                canonical, _interface_semantics(dataset_interfaces[name]), name
            )
        if selected_storage is not None:
            storage[name] = selected_storage
        interfaces[name] = interface

    execution = ModelExecution(
        config.model.architecture_model(),
        interfaces,
        ModelExecutionOptions.from_training(training),
        project_root=config.run.project_root,
        model_name=config.run.model_name,
        device=config.run.execution.device,
        seed=seed,
        embedding_layer_names=config.run.export.embedding_layer_names,
        interface_storage=storage,
    )
    if history_only:
        validate_history_transfer(source, execution, history_interface)
    for name, binding in config.policy_bindings.items():
        if name not in interfaces:
            raise ValueError(f"policy_bindings: unknown interface {name!r}")
        apply_binding(execution, name, resolve_binding(execution, name, binding))

    for phase in phases:
        if phase.algorithm == "supervised":
            continue
        name = phase.model_interface
        policy = interfaces[name].policy
        if policy is None:
            raise ValueError(
                f"{phase.name}: interface {name!r} requires a policy binding"
            )
        for environment in (
            phase.training_environment,
            phase.evaluation_environment,
        ):
            if environment and environment.units != policy.units:
                raise ValueError(
                    f"{phase.name}: environment units differ from policy units"
                )

    curriculum = supervised[0].curriculum if supervised else None
    resolved = ResolvedSequifierConfig(
        ownership=config.model_dump(mode="json"),
        authored=config,
        runtime_settings=training,
        execution=execution,
        model_metadata_paths=metadata_paths,
        dataset_training=datasets,
        training_plan=resolved_phases,
        initialize_from=config.initialize_from,
        actor_initialization=config.actor_initialization,
        model_path=config.run.outputs.model_path,
        report_path=config.run.outputs.report_path,
        phase_directory=config.run.outputs.phase_directory,
        policy_data=policy_data if has_policy else None,
        curriculum_training=curriculum is not None,
        curriculum_column=curriculum.column if curriculum else None,
        publication=publication_plan(config.run, resolved_phases),
    )
    if has_policy:
        resolved._prepared_inputs = prepared_inputs
    from sequifier.config.input_contract_resolution import validate_input_contracts
    from sequifier.io.output_paths import resolve_training_output_paths

    validate_input_contracts(config, resolved)
    resolve_training_output_paths(resolved)
    return resolved


def policy_run_config(
    config: ResolvedSequifierConfig,
    phases,
    *,
    incoming=None,
    index=0,
    resuming=None,
):
    """Select resolved inputs for one native policy lifecycle."""
    from sequifier.artifacts.phase import PhaseArtifacts
    from sequifier.artifacts.run_checkpoint import checkpoint_path
    from sequifier.config.components import ResumeConfig
    from sequifier.rl.run_config import ResolvedPolicyRun

    names = {phase.model_interface for phase in phases}
    if len(names) != 1:
        raise ValueError("a native policy lifecycle requires one interface")
    name = next(iter(names))
    initialization = (
        incoming if index or incoming is not None else config.initialize_from
    )
    actor_initialization = config.actor_initialization if index == 0 else "attach"
    binding = config.interfaces[name].policy
    if binding is None:
        raise ValueError("selected interface has no policy binding")
    resume = config.runtime_settings.resume
    checkpoint = checkpoint_path(config)
    if resuming is not None:
        checkpoint = PhaseArtifacts.for_phase(
            config.phase_directory, index, phases[0].name
        ).recovery
    return ResolvedPolicyRun(
        parent=config,
        phase_offset=index,
        policy_interface=name,
        policy=binding,
        phases=tuple(phases),
        initialize_from=initialization,
        actor_initialization=actor_initialization,
        resume=ResumeConfig(
            checkpoint_path=str(checkpoint),
            policy=("required" if resuming else "never")
            if resuming is not None
            else (resume.policy if resume else "never"),
            mode=resume.mode if resume else "reset_environments",
        ),
        model_path=config.model_path,
        report_path=config.report_path,
        checkpoint_path=str(checkpoint),
        phase_directory=config.phase_directory,
    )
