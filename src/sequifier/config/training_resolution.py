"""Resolve the common authored plan into model, dataset and task contracts."""

from pathlib import Path

from sequifier.config.metadata import DatasetMetadata, load_dataset_metadata
from sequifier.config.policy_train_config import (
    SequenceSourceConfig,
    resolve_sequence_source,
)
from sequifier.config.train_config import (
    DatasetPartSpecModel,
    OwnedModelSpec,
    PreferencePartSpecModel,
    ResolvedSequifierConfig,
    SequencePartSpecModel,
    SequifierConfig,
    TrainingRuntimeSettings,
    _resolve_interface,
    resolve_sequifier_config,
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

    def resolve(self, *, prepared=True):
        sequences, preferences = {}, {}
        for ref, (_, part) in self.parts.items():
            if isinstance(part, PreferencePartSpecModel):
                path = self.root / part.manifest_path
                preferences[ref] = str(path.resolve() if prepared else path)
                continue
            if isinstance(part, SequencePartSpecModel):
                source = SequenceSourceConfig.model_validate(
                    part.model_dump(exclude={"type"})
                )
            elif prepared and isinstance(part, DatasetPartSpecModel):
                source = SequenceSourceConfig(
                    metadata_config_path=part.metadata_config_path,
                    split_index=0,
                    format="prepared",
                    encoding="model",
                )
            else:
                continue
            sequences[ref] = resolve_sequence_source(source, self.root)
        return sequences, preferences


def _resolve_phase_sources(config, phase, available, *, strict=True):
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
            if not strict:
                return False
            raise ValueError(
                f"{phase.name}: select {'preference' if preference else 'sequence'} parts"
            )
        if single and len(expanded) != 1:
            raise ValueError(f"{phase.name}: select one training preference part")
        setattr(owner, key, expanded[0] if single else expanded)
    return True


def phase_interface(config, phase):
    names = set()
    refs = list(getattr(phase, "sources", []))
    if phase.algorithm in {"policy_imitation", "iql"}:
        refs.extend(phase.evaluation.sources or [])
    for ref in refs:
        names.add(config.dataset_training[ref.source.partition(".")[0]].model_interface)
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


def _history_initialized_metadata(config, name, binding, source):
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
    return metadata.model_copy(
        update={
            "static_columns": static.columns if static else [],
            "static_features_path": source.static_features_path,
            "image_sources": source.image_sources,
            "preconditioning_image": source.preconditioning_image,
            "special_token_ids": source.special_token_ids,
            "depth_layouts": source.depth_layouts,
            "tensor_payload_version": source.tensor_payload_version,
            "normalize_real_columns": source.normalize_real_columns,
        }
    )


def _fit_input_metadata(config, name, binding, metadata, root, prepared_inputs):
    """Fit missing raw codecs/moments on the same episode-safe training split."""
    from types import SimpleNamespace

    from sequifier.config.policy_inputs import PolicyFittingSpec, PolicyInputLayout

    phase = next(
        (
            phase
            for phase in config.training_plan.phases
            if phase.algorithm != "supervised"
            and phase_interface(config, phase) == name
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
    sources, preferences = TrainingSources(config, root).resolve(prepared=False)
    phase = phase.model_copy(deep=True)
    if not _resolve_phase_sources(
        config,
        phase,
        preferences if phase.algorithm == "dpo" else sources,
        strict=False,
    ):
        return metadata  # Prepared sources already own frozen preprocessing metadata.
    run = SimpleNamespace(
        validation_fraction=0.2,
        source_validation_fractions=_source_fractions(config),
        seed=config.run.reproducibility.seed,
        preferences=preferences,
        sources=sources,
        episode_cache_size=config.runtime_settings.episode_cache_size,
    )
    inputs = prepared_inputs.resolve(run)
    inputs.revalidate(inputs.identity, f"interfaces.{name}.fitting_sources")
    data = prepared_inputs.prepare_phase(run, phase, provisional, inputs, fitting=True)
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
        fitting.use_config(fitted)
        for column, statistics in fitting.normalization_statistics().items():
            values["selected_columns_statistics"].setdefault(column, statistics)
    prepared_inputs.measure()
    return DatasetMetadata.model_validate(values)


def _supervised_subset(config):
    phases = [
        phase
        for phase in config.training_plan.phases
        if phase.algorithm == "supervised"
    ]
    names = {ref.source.partition(".")[0] for phase in phases for ref in phase.sources}
    names.update(
        ref.source.partition(".")[0]
        for phase in phases
        if phase.evaluation is not None
        for ref in phase.evaluation.sources
    )
    return config.model_copy(
        update={
            "dataset_training": {
                name: dataset
                for name, dataset in config.dataset_training.items()
                if name in names
            },
            "training_plan": config.training_plan.model_copy(update={"phases": phases}),
            "policy_bindings": {},
        }
    )


def resolve_training_run(authored: SequifierConfig):
    from sequifier.artifacts.model_artifact import read_execution_artifact
    from sequifier.artifacts.model_config import model_execution_from_model_config
    from sequifier.model.execution import ModelExecution
    from sequifier.model.history_transfer import validate_history_transfer
    from sequifier.rl.attachment import apply_binding, resolve_binding
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs

    prepared_inputs = PreparedTrainingInputs()
    config = authored.model_copy(deep=True)
    root = Path(config.run.project_root).resolve()
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
            actual = getattr(source.runtime_settings, field)
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
        if history_only:
            if (
                "layer_type_dtypes" in config.run.execution.precision.model_fields_set
                and training.layer_type_dtypes
                != source.runtime_settings.layer_type_dtypes
            ):
                raise ValueError("history transfer differs at layer_type_dtypes")
            training.layer_type_dtypes = source.runtime_settings.layer_type_dtypes
    assert config.model is not None
    assert config.model.backbone is not None
    for dataset in config.dataset_training.values():
        if dataset.model_interface is None:
            if len(config.model.interfaces) != 1:
                raise ValueError("dataset model_interface is required when ambiguous")
            dataset.model_interface = next(iter(config.model.interfaces))
        if dataset.model_interface not in config.model.interfaces:
            raise ValueError(
                f"unknown dataset model_interface {dataset.model_interface!r}"
            )
    training = config.runtime_settings
    if training.context_length is None:
        training.context_length = config.model.backbone.architecture.max_context_length
    if training.training_objective is None:
        if any(
            phase.algorithm == "supervised" for phase in config.training_plan.phases
        ):
            raise ValueError(
                "supervised phases require model.execution.training_objective"
            )
        training.training_objective = "policy"
        training.target_offset = 0
    training = TrainingRuntimeSettings.model_validate(training.model_dump())
    config.runtime_settings = training
    for field in (
        "context_length",
        "training_objective",
        "target_offset",
        "forward_attention",
    ):
        setattr(config.model.execution, field, getattr(training, field))

    # All declared metadata for an interface is an assertion of the same facts.
    metadata = {}
    declarations = {
        name: value.input_contract.metadata_config_path
        for name, value in config.model.interfaces.items()
        if value.input_contract.metadata_config_path
    }
    for name, path in (declarations or {}).items():
        if name not in config.model.interfaces:
            raise ValueError(f"unknown metadata model_interface {name!r}")
        metadata[name] = load_dataset_metadata(str(root / path))
    part_metadata = {}
    for dataset_name, dataset in config.dataset_training.items():
        for part_name, part in dataset.parts.items():
            path = getattr(part, "metadata_config_path", None)
            if path:
                value = load_dataset_metadata(str(root / path))
                part_metadata[f"{dataset_name}.{part_name}"] = value
                existing = metadata.get(dataset.model_interface)
                if existing is not None:
                    for key in (
                        "column_data_types",
                        "n_classes",
                        "id_maps",
                        "selected_columns_statistics",
                        "normalize_real_columns",
                    ):
                        if getattr(existing, key) != getattr(value, key):
                            raise ValueError(
                                f"{dataset_name}.{part_name}: interface metadata differs at {key}"
                            )
                metadata[dataset.model_interface] = value
    interfaces = {}
    history_interface = (
        phase_interface(config, config.training_plan.phases[0])
        if history_only
        else None
    )
    if history_only:
        assert source is not None
        if history_interface not in source.interfaces:
            raise ValueError(
                f"history transfer requires interface {history_interface!r}"
            )
    for name, interface in config.model.interfaces.items():
        if source is not None and not history_only:
            interfaces[name] = source.interface(name).model_copy(deep=True)
            if name in metadata:
                candidate = _resolve_interface(
                    name,
                    interface,
                    metadata[name],
                    training,
                    metadata[name].storage_layout,
                )
                for key in (
                    "column_data_types",
                    "id_maps",
                    "selected_columns_statistics",
                    "target_decoder_ids",
                ):
                    if getattr(candidate, key) != getattr(interfaces[name], key):
                        raise ValueError(
                            f"{name}: metadata differs from initialization at {key}"
                        )
        else:
            value = metadata.get(name)
            binding = config.policy_bindings.get(name)
            if name == history_interface and value is None:
                assert source is not None
                if binding is None:
                    raise ValueError(f"{name}: history_only requires a policy binding")
                value = _history_initialized_metadata(
                    config, name, binding, source.interface(name)
                )
            if value is None:
                if binding is None:
                    raise ValueError(f"{name}: model interface requires metadata")
                value = _inferred_metadata(config, name, binding)
            if binding is not None and name != history_interface:
                value = _fit_input_metadata(
                    config, name, binding, value, root, prepared_inputs
                )
            interfaces[name] = _resolve_interface(
                name, interface, value, training, value.storage_layout
            )
    execution = ModelExecution(config.model.architecture_model(), interfaces, training)
    if history_only:
        validate_history_transfer(source, execution, history_interface)
    for name, binding in config.policy_bindings.items():
        if name not in interfaces:
            raise ValueError(f"policy_bindings: unknown interface {name!r}")
        apply_binding(execution, name, resolve_binding(execution, name, binding))

    sequences, preferences = TrainingSources(config, root).resolve()
    for phase in config.training_plan.phases:
        phase.optimization = phase.optimization_configs()
        if phase.algorithm == "supervised":
            for ref in phase.sources:
                for _, part in source_parts(config, ref.source):
                    if not isinstance(part, DatasetPartSpecModel):
                        raise ValueError(
                            f"{phase.name}: supervised sources require prepared parts"
                        )
            continue
        name = phase_interface(config, phase)
        phase.model_interface = name
        if interfaces[name].policy is None:
            raise ValueError(
                f"{phase.name}: interface {name!r} requires a policy binding"
            )
        _resolve_phase_sources(
            config, phase, preferences if phase.algorithm == "dpo" else sequences
        )
        for environment in (
            getattr(phase, "environment", None),
            getattr(getattr(phase, "evaluation", None), "environment", None),
        ):
            if environment and environment.units != interfaces[name].policy.units:
                raise ValueError(
                    f"{phase.name}: environment units differ from policy units"
                )
        reference = getattr(phase, "reference", None)
        if reference and reference.path:
            reference.path = str((root / reference.path).resolve())
        archive = getattr(phase, "archive_directory", None)
        if archive:
            phase.archive_directory = str((root / archive).resolve())
    from sequifier.runtime.plan import validate_continuation

    for previous, current in zip(
        config.training_plan.phases, config.training_plan.phases[1:]
    ):
        validate_continuation(previous, current, resolved=True)
    if any(phase.algorithm == "supervised" for phase in config.training_plan.phases):
        subset = _supervised_subset(config).model_copy(
            update={"initialize_from": None, "actor_initialization": "attach"}
        )
        resolved = resolve_sequifier_config(subset, part_metadata, execution=execution)
    else:
        resolved = ResolvedSequifierConfig(
            model=execution.model,
            interfaces=execution.interfaces,
            runtime_settings=training,
            dataset_training={},
            training_plan=[],
            device=config.run.execution.device,
            project_root=str(root),
            model_name=config.run.model_name,
            seed=config.run.reproducibility.seed,
            export_generative_model=config.run.export.generative_model,
            export_embedding_model=config.run.export.embedding_model,
            embedding_layer_names=config.run.export.embedding_layer_names,
            export_onnx=config.run.export.onnx,
            export_pt=config.run.export.pt,
        )
    resolved.ownership = config.model_dump(mode="json")
    resolved.source_validation_fractions = _source_fractions(config)
    resolved.normalization_by_interface = {
        name: value
        for name in interfaces
        if (value := _normalization(config, name)) is not None
    }
    resolved.policy_bindings = {
        name: interface.policy
        for name, interface in interfaces.items()
        if interface.policy is not None
    }
    resolved._prepared_inputs = prepared_inputs
    resolved.authored = config
    resolved.initialize_from = config.initialize_from
    resolved.actor_initialization = config.actor_initialization
    from sequifier.io.output_paths import resolve_training_output_paths

    for name in ("model_path", "report_path", "phase_directory"):
        setattr(resolved, name, getattr(config.run.outputs, name))
    resolve_training_output_paths(resolved)
    resolved.sequence_sources = sequences
    resolved.preference_sources = preferences
    supervised_by_name = {phase.name: phase for phase in resolved.training_plan}
    resolved.training_plan = [
        supervised_by_name[phase.name] if phase.algorithm == "supervised" else phase
        for phase in config.training_plan.phases
    ]
    for index, phase in enumerate(resolved.training_plan):
        if phase.algorithm == "supervised":
            phase.resolved_seed = config.run.reproducibility.seed + index
    export = config.run.export
    resolved.export_generative_model = export.generative_model
    resolved.export_embedding_model = export.embedding_model
    resolved.embedding_layer_names = export.embedding_layer_names
    resolved.export_onnx = export.onnx
    resolved.export_pt = export.pt
    resolved.export_with_dropout = export.with_dropout
    from sequifier.config.input_contract_resolution import validate_input_contracts

    validate_input_contracts(config, resolved)
    return resolved


def policy_run_config(
    config,
    phases,
    *,
    incoming=None,
    index=0,
    destination=None,
    resuming=None,
):
    """Select resolved inputs for one native policy lifecycle."""
    from sequifier.artifacts.dependencies import model_metadata_paths
    from sequifier.artifacts.model_artifact import read_execution_artifact
    from sequifier.artifacts.model_config import model_execution_from_model_config
    from sequifier.artifacts.model_export import model_execution_config
    from sequifier.config.components import ResumeConfig
    from sequifier.model.execution import ModelExecution
    from sequifier.rl.attachment import apply_binding, resolve_binding
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs
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
    if initialization and actor_initialization == "attach":
        execution, _ = model_execution_from_model_config(
            read_execution_artifact(initialization).model_config.values,
            device=config.device,
            interface_name=name,
        )
        from sequifier.artifacts.model_config import extend_prediction_distributions

        extend_prediction_distributions(execution, config.model)
        apply_binding(
            execution,
            name,
            resolve_binding(execution, name, config.policy_bindings[name]),
        )
    binding = execution.interface(name).policy
    if binding is None:
        raise ValueError("selected interface has no policy binding")
    training = config.runtime_settings
    resume = training.resume
    root = Path(config.project_root)
    checkpoint = (
        (root / resume.checkpoint_path).resolve()
        if resume and resume.checkpoint_path
        else root
        / f"checkpoints/runs/{config.model_name}/{config.model_name}-latest.pt"
    )
    if resuming is not None:
        checkpoint = (
            Path(config.phase_directory) / f"{index:03d}-{phases[0].name}" / "run.pt"
        )
    run = ResolvedPolicyRun(
        project_root=config.project_root,
        model_name=config.model_name,
        device=config.device,
        seed=config.seed,
        enforce_determinism=training.enforce_determinism,
        model=execution.model,
        policy=binding,
        execution_model=model_execution_config(execution),
        policy_interface=name,
        initialize_from=initialization,
        metadata_paths=model_metadata_paths(config, name),
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
    run._prepared_inputs = PreparedTrainingInputs.for_run(config)
    return run
