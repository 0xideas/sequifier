"""Resolve the common authored plan into model, dataset and task contracts."""

from pathlib import Path

from sequifier.config.metadata import (
    DatasetMetadata,
    dataset_tensor_kind,
    dataset_tensor_version,
    load_dataset_metadata,
)
from sequifier.config.naming import resolve_run_seed
from sequifier.config.policy_train_config import (
    SequenceSourceConfig,
    resolve_sequence_source,
)
from sequifier.config.run_capabilities import publication_plan, validate_runner_settings
from sequifier.config.sources import phase_sources
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
    _resolve_interface,
    resolve_phases,
    resolve_supervised_datasets,
    validate_training_contract,
)
from sequifier.special_tokens import SPECIAL_TOKEN_IDS


def _normalization(config, name):
    if config.model and name in config.model.interfaces:
        return config.model.interfaces[name].input_contract.normalization
    return None


def _source_fractions(config):
    return {
        f"{name}.{part}": spec.split.validation_fraction
        for name, spec in config.dataset_training.items()
        for part in spec.parts
    }


def source_parts(config, reference):
    """Expand a reference without silently filtering out incompatible parts."""
    dataset, _, part = reference.partition(".")
    spec = config.dataset_training[dataset]
    return [
        (f"{dataset}.{name}", value)
        for name, value in spec.parts.items()
        if not part or name == part
    ]


class TrainingSources:
    """Named authored parts shared by metadata fitting and run resolution."""

    def __init__(self, config, root):
        self.root = root
        self.parts = {
            f"{dataset_name}.{part_name}": (dataset.model_interface, part)
            for dataset_name, dataset in config.dataset_training.items()
            for part_name, part in dataset.parts.items()
        }

    def resolve(self, metadata_by_part):
        sequences, preferences = {}, {}
        for ref, (_, part) in self.parts.items():
            if isinstance(part, PreferencePartSpecModel):
                path = self.root / part.manifest_path
                preferences[ref] = str(path.resolve())
                continue
            if isinstance(part, SequencePartSpecModel):
                source = SequenceSourceConfig.model_validate(
                    part.model_dump(exclude={"type"})
                )
            else:
                source = SequenceSourceConfig(
                    metadata_config_path=part.metadata_config_path,
                    split_index=0,
                    format="prepared",
                    encoding="model",
                )
            sequences[ref] = resolve_sequence_source(
                source, self.root, metadata=metadata_by_part.get(ref)
            )
        return sequences, preferences


def _resolve_phase_sources(config, phase, available):
    """Expand named sources for both metadata fitting and final run resolution."""
    preference = phase.algorithm == "dpo"
    fields = (
        [(phase, "preferences"), (phase.evaluation, "sources")]
        if preference
        else [(phase, "sources"), (phase.evaluation, "sources")]
    )
    for owner, key in fields:
        refs = getattr(owner, key, None)
        if refs is None:
            continue
        single = key == "preferences"
        expanded = [
            ref.model_copy(update={"source": full})
            for ref in ([refs] if single else refs)
            for full, _ in source_parts(config, ref.source)
        ]
        if not expanded or any(ref.source not in available for ref in expanded):
            raise ValueError(
                f"{phase.name}: select {'preference' if preference else 'sequence'} parts"
            )
        if single and len(expanded) != 1:
            raise ValueError(f"{phase.name}: select one training preference part")
        setattr(owner, key, expanded[0] if single else expanded)


def phase_interface(config, phase):
    names = {
        config.dataset_training[ref.source.partition(".")[0]].model_interface
        for ref in phase_sources(phase)
    }
    explicit = getattr(phase, "model_interface", None)
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


def _inferred_metadata(config, name, binding):
    """Identity numeric inputs and explicitly declared categorical action codecs."""
    interface = config.model.interfaces[name]
    columns = set(interface.input_columns + interface.target_columns)
    if interface.static_conditioning is not None:
        columns.update(interface.static_conditioning.columns)
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
        window_length=config.runtime_settings.context_length
        + max(1, config.runtime_settings.target_offset),
        column_data_types=types,
        n_classes=counts,
        id_maps=maps,
        selected_columns_statistics={
            column: statistics.get(column, {"mean": 0.0, "std": 1.0})
            for column in columns
            if column not in counts
            and (normalization is not None or column not in interface.input_columns)
        },
    )


def _history_initialized_metadata(config, name, binding, source, storage):
    """Resolve new targets while sourcing observation preprocessing from the artifact."""
    metadata = _inferred_metadata(config, name, binding)
    static = source.static_conditioning
    inputs = set(source.input_columns + (static.columns if static else []))
    targets = set(config.model.interfaces[name].target_columns)
    for key in (
        "column_data_types",
        "n_classes",
        "id_maps",
        "selected_columns_statistics",
    ):
        values = getattr(metadata, key)
        original = getattr(source, key)
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
    sources,
    preferences,
    source_fractions,
    prepared_inputs,
):
    """Fit missing raw codecs/moments on the same episode-safe training split."""
    from types import SimpleNamespace

    from sequifier.config.policy_inputs import PolicyFittingSpec, PolicyInputLayout

    phase = next(
        (
            phase
            for phase in config.training_plan.phases
            if phase.algorithm != "supervised" and phase.model_interface == name
        ),
        None,
    )
    if phase is None or phase.algorithm not in {"policy_imitation", "iql", "dpo"}:
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
        context_length=config.runtime_settings.context_length,
        interface_name=name,
    )
    available = preferences if phase.algorithm == "dpo" else sources
    if any(ref.source not in available for ref in phase_sources(phase)):
        return metadata  # Prepared dataset parts already own frozen preprocessing.
    run = SimpleNamespace(
        validation_fraction=0.2,
        source_validation_fractions=source_fractions,
        seed=config.run.reproducibility.seed,
        preferences=preferences,
        sources=sources,
        episode_cache_size=config.runtime_settings.episode_cache_size,
    )
    inputs = prepared_inputs.resolve(run)
    inputs.revalidate(inputs.identity, f"interfaces.{name}.fitting_sources")
    from sequifier.io.episode_features import EpisodeFeatures

    with prepared_inputs.fitting_phase(run, phase, provisional, inputs) as data:
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
        for ref in phase_sources(phase)
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
    training = config.runtime_settings
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
        for field in (
            "context_length",
            "training_objective",
            "target_offset",
            "forward_attention",
        ):
            actual = getattr(source.options, field)
            if (
                history_only
                and field in {"training_objective", "target_offset"}
                and field in training.model_fields_set
            ):
                continue
            if (
                field in training.model_fields_set
                and getattr(training, field) != actual
            ):
                raise ValueError(
                    f"model.execution.{field} differs from initialize_from"
                )
            setattr(training, field, actual)
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
    config.runtime_settings = training
    if needs_policy_contracts:
        for field in (
            "context_length",
            "training_objective",
            "target_offset",
            "forward_attention",
        ):
            setattr(config.model.execution, field, getattr(training, field))

    # Keep each physical metadata read shared by model assertions and source resolution.
    metadata_cache: dict[Path, DatasetMetadata] = {}

    def read_metadata(path):
        resolved = (root / path).resolve()
        if resolved not in metadata_cache:
            metadata_cache[resolved] = load_dataset_metadata(str(resolved))
        return metadata_cache[resolved]

    metadata_paths = {
        name: str(root / path)
        for name, spec in config.model.interfaces.items()
        if (path := spec.input_contract.metadata_config_path)
    }
    declarations = {name: read_metadata(path) for name, path in metadata_paths.items()}
    part_metadata = {}
    for dataset_name, dataset in config.dataset_training.items():
        for part_name, part in dataset.parts.items():
            ref = f"{dataset_name}.{part_name}"
            path = getattr(part, "metadata_config_path", None)
            if metadata_by_part is not None and isinstance(part, DatasetPartSpecModel):
                if ref in metadata_by_part:
                    part_metadata[ref] = metadata_by_part[ref]
            elif path:
                part_metadata[ref] = read_metadata(path)
    if loaded_metadata is not None:
        loaded_metadata.update(part_metadata)

    source_fractions = _source_fractions(config)
    sources = TrainingSources(config, root)
    sequences, preferences = (
        sources.resolve(metadata_by_part=part_metadata)
        if needs_policy_contracts
        else ({}, {})
    )
    fitting_sequences = {
        ref: sequence
        for ref, sequence in sequences.items()
        if isinstance(sources.parts[ref][1], SequencePartSpecModel)
    }

    for phase in phases:
        if phase.algorithm == "supervised":
            continue
        phase.optimization = phase.optimization_configs()
        phase.model_interface = phase_interface(config, phase)
        _resolve_phase_sources(
            config, phase, preferences if phase.algorithm == "dpo" else sequences
        )
        reference = getattr(phase, "reference", None)
        if reference and reference.path:
            reference.path = str((root / reference.path).resolve())
        archive = getattr(phase, "archive_directory", None)
        if archive:
            phase.archive_directory = str((root / archive).resolve())

    from sequifier.runtime.plan import validate_continuation

    for index, phase in enumerate(phases):
        validate_continuation(
            phases[index - 1] if index else None, phase, resolved=True
        )

    resolved_phases = resolve_phases(phases, config.dataset_training, seed=seed)
    validate_training_contract(config, resolved_phases)

    datasets, dataset_interfaces, dataset_storage = resolve_supervised_datasets(
        config,
        part_metadata,
        resolved_phases,
        supervised_names,
        metadata_from_file=metadata_by_part is None,
    )
    metadata = dict(declarations)
    if needs_policy_contracts:
        for ref, value in part_metadata.items():
            name = config.dataset_training[ref.partition(".")[0]].model_interface
            if name in metadata:
                for key in (
                    "column_data_types",
                    "n_classes",
                    "id_maps",
                    "selected_columns_statistics",
                    "normalize_real_columns",
                ):
                    if getattr(metadata[name], key) != getattr(value, key):
                        raise ValueError(f"{ref}: interface metadata differs at {key}")
            metadata[name] = value

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
            if value is not None:
                candidate = _resolve_interface(name, spec, value, training)
                for key in (
                    "column_data_types",
                    "id_maps",
                    "selected_columns_statistics",
                    "target_decoder_ids",
                ):
                    if getattr(candidate, key) != getattr(interface, key):
                        raise ValueError(
                            f"{name}: metadata differs from initialization at {key}"
                        )
            selected_storage = source.interface_storage.get(name)
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
                )
            if value is None:
                if binding is None:
                    raise ValueError(f"{name}: model interface requires metadata")
                value = _inferred_metadata(config, name, binding)
            if binding is not None and name != history_interface:
                value = _fit_input_metadata(
                    config,
                    name,
                    binding,
                    value,
                    fitting_sequences,
                    preferences,
                    source_fractions,
                    prepared_inputs,
                )
            interface = _resolve_interface(name, spec, value, training)
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
            getattr(phase, "environment", None),
            getattr(phase.evaluation, "environment", None),
        ):
            if environment and environment.units != policy.units:
                raise ValueError(
                    f"{phase.name}: environment units differ from policy units"
                )

    curriculum = supervised[0].curriculum if supervised else None
    resolved = ResolvedSequifierConfig(
        ownership=config.model_dump(mode="json"),
        authored=config,
        project_root=config.run.project_root,
        model_name=config.run.model_name,
        device=config.run.execution.device,
        seed=seed,
        runtime_settings=training,
        model_options=execution.options,
        model=execution.model,
        interfaces=interfaces,
        interface_storage=storage,
        model_metadata_paths=metadata_paths,
        dataset_training=datasets,
        training_plan=resolved_phases,
        initialize_from=config.initialize_from,
        actor_initialization=config.actor_initialization,
        model_path=config.run.outputs.model_path,
        report_path=config.run.outputs.report_path,
        phase_directory=config.run.outputs.phase_directory,
        sequence_sources=sequences,
        preference_sources=preferences,
        source_validation_fractions=source_fractions,
        policy_bindings={
            name: interface.policy
            for name, interface in interfaces.items()
            if interface.policy
        },
        curriculum_training=curriculum is not None,
        curriculum_column=curriculum.column if curriculum else None,
        publication=publication_plan(config.run, resolved_phases),
    )
    if has_policy:
        resolved._prepared_inputs = prepared_inputs
    from sequifier.config.input_contract_resolution import validate_input_contracts
    from sequifier.io.output_paths import resolve_training_output_paths

    validate_input_contracts(config, resolved, declarations)
    resolve_training_output_paths(resolved)
    return resolved


def policy_run_config(
    config: ResolvedSequifierConfig,
    phases,
    *,
    incoming=None,
    index=0,
    destination=None,
    resuming=None,
):
    """Select resolved inputs for one native policy lifecycle."""
    from sequifier.artifacts.run_checkpoint import checkpoint_path
    from sequifier.config.components import ResumeConfig
    from sequifier.model.execution import ModelExecution
    from sequifier.rl.run_config import ResolvedPolicyRun

    names = {phase.model_interface for phase in phases}
    if len(names) != 1:
        raise ValueError("a native policy lifecycle requires one interface")
    name = next(iter(names))
    execution = ModelExecution.from_config(config)
    initialization = (
        incoming if index or incoming is not None else config.initialize_from
    )
    actor_initialization = config.actor_initialization if index == 0 else "attach"
    binding = execution.interface(name).policy
    if binding is None:
        raise ValueError("selected interface has no policy binding")
    training = config.runtime_settings
    resume = training.resume
    checkpoint = checkpoint_path(config)
    if resuming is not None:
        checkpoint = (
            Path(config.phase_directory) / f"{index:03d}-{phases[0].name}" / "run.pt"
        )
    return ResolvedPolicyRun(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        phase_offset=index,
        enforce_determinism=training.enforce_determinism,
        policy=binding,
        execution=execution,
        policy_interface=name,
        initialize_from=initialization,
        metadata_paths=[config.model_metadata_paths[name]]
        if name in config.model_metadata_paths
        else [],
        actor_initialization=actor_initialization,
        normalization=config.normalization_by_interface.get(name),
        units=binding.units,
        sources=config.sequence_sources,
        preferences=config.preference_sources,
        phases=phases,
        validation_fraction=0.2,
        source_validation_fractions=config.source_validation_fractions,
        latest_checkpoint_interval_minutes=training.latest_checkpoint_interval_minutes,
        episode_cache_size=training.episode_cache_size,
        log_interval_iterations=training.log_interval_iterations,
        resume=ResumeConfig(
            checkpoint_path=str(checkpoint),
            policy=("required" if resuming else "never")
            if resuming is not None
            else (resume.policy if resume else "never"),
            mode=resume.mode if resume else "reset_environments",
        ),
        embedding_layer_names=config.embedding_layer_names,
        model_path=str(destination or config.model_path),
        report_path=config.report_path
        if destination is None
        else str(Path(destination).with_name("report.json")),
        checkpoint_path=str(checkpoint),
        phase_directory=config.phase_directory
        if destination is None
        else str(Path(destination).parent / "native"),
    )
