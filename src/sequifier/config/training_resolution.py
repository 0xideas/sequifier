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
)
from sequifier.config.run_capabilities import (
    training_run_mode,
    validate_runner_settings,
)
from sequifier.config.sources import SourceReference, TrainingSourceSpecModel
from sequifier.config.train_config import (
    DatasetPartSpecModel,
    OwnedModelSpec,
    PreferencePartSpecModel,
    ResolvedInterfaceStorage,
    ResolvedSequifierConfig,
    ResolvedTrainingSource,
    SequencePartSpecModel,
    SequifierConfig,
    _assert_compatible,
    _part_features,
    _resolve_interface,
    resolve_phases,
    resolve_supervised_datasets,
    validate_training_contract,
)
from sequifier.helpers import (
    ModelWindowView,
    derive_target_column_types,
    normalize_path,
    resolve_window_view,
)
from sequifier.io.indexed_sequences import IndexedView, is_indexed_view
from sequifier.model.execution_options import ModelExecutionOptions
from sequifier.objectives import target_offset_for_objective


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


@dataclass(frozen=True)
class IndexedSelection:
    source: str
    view: IndexedView

    def validate_holdout(self, training, phase):
        view = self.view
        policy = phase.algorithm != "supervised"
        keys = view.selection_keys() if policy else {}
        bounds = {} if policy else view.split_bounds()
        for candidate in training:
            other = candidate.view
            same_store = other.manifest_path.resolve() == view.manifest_path.resolve()
            if other.path.resolve() == view.path.resolve():
                overlap = "split manifest"
            elif policy:
                overlap = next(
                    (
                        key
                        for key, values in other.selection_keys().items()
                        if not values.isdisjoint(keys[key])
                    ),
                    None,
                )
            elif same_store and any(
                seq in bounds and max(low, bounds[seq][0]) < min(high, bounds[seq][1])
                for seq, (low, high) in other.split_bounds().items()
            ):
                overlap = "target positions"
            else:
                overlap = None
            if overlap:
                raise ValueError(
                    f"{phase.name}: {candidate.source} overlaps held-out {self.source} at {overlap}"
                )
            if policy and same_store and view.is_fitting_split:
                raise ValueError(
                    f"{phase.name}: the frozen metadata was fitted on held-out {self.source}"
                )


class TrainingSources:
    """Resolve named parts and share their metadata throughout compilation."""

    def __init__(self, config, root, supplied):
        self.root = root
        self.project_root = config.run.project_root
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

    def split_path(self, ref, role):
        part = self.parts[ref]
        if part.indexed is not None:
            return str(part.indexed.path)
        index = 0 if role == "training" else 1
        if part.metadata is None or index >= len(part.metadata.split_paths):
            return None
        return normalize_path(part.metadata.split_paths[index], self.project_root)

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

    def part_names(self, ref: SourceReference) -> tuple[str, ...]:
        return (ref.part,) if ref.part else tuple(self.datasets[ref.dataset].parts)

    def expand(self, refs):
        for ref in refs:
            for part in self.part_names(ref):
                yield ref.model_copy(update={"source": f"{ref.dataset}.{part}"})

    def supervised_source(
        self, ref: TrainingSourceSpecModel, accumulation_batches: int = 1
    ) -> ResolvedTrainingSource:
        return ResolvedTrainingSource(
            source=ref.source,
            parts=self.part_names(ref),
            selection_weight=ref.selection_weight or 1.0,
            batches_per_selection=ref.batches_per_selection or accumulation_batches,
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
            sequences[ref] = source
        return sequences, preferences


def _resolve_phase_sources(parts, phase, sequences, preferences):
    """Expand named sources for both metadata fitting and final run resolution."""
    refs = phase.data_sources
    if refs.kind == "none":
        return
    available = preferences if refs.kind == "preference" else sequences

    def expand(selected):
        expanded = list(parts.expand(selected))
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
        config.dataset_training[ref.dataset].model_interface
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
    maps = binding.categorical_id_maps()
    types.update(dict.fromkeys(maps, "Int64"))
    counts = {column: max(mapping.values()) + 1 for column, mapping in maps.items()}
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
    from sequifier.config.policy_inputs import PolicyFittingSpec

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
    provisional = PolicyFittingSpec.from_metadata(
        name,
        interface.input_columns,
        static,
        binding,
        metadata,
        settings.context_length,
    )
    observations, categorical = (
        provisional.observation_columns,
        provisional.categorical_columns,
    )
    missing_codecs = set(categorical) - metadata.n_classes.keys()
    missing_statistics = (
        set(observations + static)
        - set(categorical)
        - metadata.selected_columns_statistics.keys()
    )
    if not missing_codecs and (not missing_statistics or normalization is not None):
        return metadata
    available = (
        data_settings.preferences
        if phase.source_kind == "preference"
        else data_settings.sources
    )
    if any(ref.source not in available for ref in phase.data_sources):
        return metadata  # Prepared dataset parts already own frozen preprocessing.
    inputs = prepared_inputs.resolve(data_settings)
    inputs.dependencies.revalidate()
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


def _validate_indexed_phase_coverage(sources, phases, sequences, prepared_inputs):
    def selections(phase, role):
        refs = phase.data_sources
        selected = refs.training if role == "training" else refs.evaluation
        for ref in sources.expand(selected):
            part = sources.parts[ref.source]
            if part.indexed is not None:
                yield IndexedSelection(ref.source, part.indexed)
                continue
            if phase.algorithm == "supervised":
                paths = [sources.split_path(ref.source, role)]
            else:
                source = sequences[ref.source]
                if source.format != "prepared" and source.split_index is None:
                    continue
                paths = prepared_inputs.source(
                    source, sources.root, metadata=part.metadata
                ).paths
            for path in paths:
                if is_indexed_view(path):
                    yield IndexedSelection(
                        ref.source, sources.indexed_view(Path(path).resolve())
                    )

    supervised = []
    for phase in phases:
        if phase.algorithm == "supervised":
            training = list(selections(phase, "training"))
            supervised.extend(training)
            for held in selections(phase, "validation"):
                held.validate_holdout(training, phase)
    for phase in phases:
        if phase.algorithm == "supervised" or phase.source_kind != "sequence":
            continue
        training = list(selections(phase, "training"))
        if training and not phase.data_sources.evaluation:
            raise ValueError(
                f"{phase.name}: indexed policy data requires explicit evaluation.sources "
                "selecting a recorded held-out view"
            )
        for held in selections(phase, "validation"):
            held.validate_holdout([*supervised, *training], phase)


def initialization_model_config(path: Path):
    from sequifier.artifacts.model_artifact import read_execution_artifact

    return read_execution_artifact(path).model_config.values


def resolve_execution_settings(
    config: SequifierConfig, load_model_config=initialization_model_config
):
    """Resolve the model and execution defaults before opening any datasets."""
    from sequifier.artifacts.model_config import model_execution_from_model_config

    model = config.model
    values = model.execution.model_dump(exclude_unset=True) if model else {}
    run_execution = config.run.execution
    dtypes = run_execution.precision.layer_type_dtypes
    max_concat = run_execution.max_concat_tensors
    source = None
    history_only = config.actor_initialization == "history_only"
    if config.initialize_from:
        path = (Path(config.run.project_root) / config.initialize_from).resolve()
        source, _ = model_execution_from_model_config(load_model_config(path))
        if model is not None and not history_only:
            from sequifier.artifacts.model_config import extend_prediction_distributions

            extend_prediction_distributions(source, model)
        if not history_only:
            model = OwnedModelSpec.from_execution(source, model)
        incoming = source.options
        for field, actual in (
            ("context_length", incoming.context_length),
            ("training_objective", incoming.training_objective),
            ("target_offset", incoming.target_offset),
            ("forward_attention", incoming.forward_attention),
        ):
            if (
                history_only
                and field in {"training_objective", "target_offset"}
                and field in values
            ):
                continue
            if field in values and values[field] != actual:
                raise ValueError(
                    f"model.execution.{field} differs from initialize_from"
                )
            values[field] = actual
        source_dtypes = source.options.layer_type_dtypes
        if (
            "layer_type_dtypes" in config.run.execution.precision.model_fields_set
            and dtypes != source_dtypes
        ):
            raise ValueError(
                "run.execution.precision.layer_type_dtypes differs from initialize_from"
            )
        dtypes = source_dtypes
        if "max_concat_tensors" not in config.run.execution.model_fields_set:
            max_concat = source.options.max_concat_tensors
    assert model is not None
    assert model.backbone is not None
    if values.get("context_length") is None:
        values["context_length"] = model.backbone.architecture.max_context_length
    if values.get("training_objective") is None:
        if any(
            phase.algorithm == "supervised" for phase in config.training_plan.phases
        ):
            raise ValueError(
                "supervised phases require model.execution.training_objective"
            )
        values.update(training_objective="policy", target_offset=0)
    run_execution.precision.layer_type_dtypes = dtypes
    run_execution.max_concat_tensors = max_concat
    training = ModelExecutionOptions(
        **values, layer_type_dtypes=dtypes, max_concat_tensors=max_concat
    )
    if run_execution.distribution.strategy == "fsdp" and dtypes is not None:
        raise ValueError("FSDP does not support manual layer pre-casting")
    validate_training_contract(config.run, model, config.training_plan.phases, training)
    return model, training, source


def resolve_training_run(
    authored: SequifierConfig,
    *,
    metadata_by_part: dict[str, DatasetMetadata] | None = None,
    loaded_metadata: dict[str, DatasetMetadata] | None = None,
) -> ResolvedSequifierConfig:
    """Compile the authored run into the records consumed by both runners."""
    from sequifier.model.execution import ModelExecution
    from sequifier.model.history_transfer import validate_history_transfer
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
        ref.dataset for phase in supervised for ref in phase.data_sources
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
    if config.initialize_from:
        config.initialize_from = str((root / config.initialize_from).resolve())
    config.model, training, source = resolve_execution_settings(config)
    history_only = config.actor_initialization == "history_only"
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
        project_root=str(root),
        sources=sequences,
        preferences=preferences,
        source_validation_fractions=source_fractions,
        seed=seed,
        episode_cache_size=config.run.execution.data_loading.episode_cache_size,
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

    _validate_indexed_phase_coverage(sources, phases, sequences, prepared_inputs)

    from sequifier.runtime.plan import validate_continuation

    for index, phase in enumerate(phases):
        validate_continuation(
            phases[index - 1] if index else None, phase, resolved=True
        )

    resolved_phases = resolve_phases(phases, sources, seed=seed)

    metadata = sources.interface_metadata(config, declarations)
    window_view = ModelWindowView(
        context_length=training.context_length,
        objective=training.training_objective,
        target_offset=target_offset_for_objective(
            training.training_objective, training.target_offset
        ),
    )
    datasets, dataset_inputs = resolve_supervised_datasets(
        config,
        sources,
        resolved_phases,
        supervised_names,
        window_view,
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
        else [*dataset_inputs, *config.model.interfaces]
    )
    for name in interface_order:
        spec = config.model.interfaces[name]
        dataset_metadata, dataset_storage = dataset_inputs.get(name, (None, None))
        value = (
            metadata.get(name)
            if needs_policy_contracts
            else dataset_metadata or metadata.get(name)
        )
        binding = config.policy_bindings.get(name)
        if source is not None and not history_only:
            interface = source.interface(name)
            selected_storage = source.interface_storage.get(name)
            if value is not None:
                features = _part_features(value, spec, require_static_table=False)
                _assert_compatible(interface.signature(), features.signature(), name)
                target_types = derive_target_column_types(
                    interface.target_columns, features.column_data_types
                )
                _assert_compatible(interface.target_column_types, target_types, name)
                if selected_storage is not None:
                    columns = interface.column_data_types
                    _assert_compatible(
                        selected_storage.raw_input_contract(columns),
                        value.raw_input_contract(columns),
                        name,
                    )
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
            interface = _resolve_interface(name, spec, value, window_view)
            selected_storage = (
                dataset_storage or ResolvedInterfaceStorage.from_metadata(value)
            )

            resolve_window_view(selected_storage.storage_layout, interface.window_view)
        if dataset_metadata is not None and value is not dataset_metadata:
            # Fitting or an explicit declaration must agree with the encoded supervised data.
            features = _part_features(dataset_metadata, spec)
            _assert_compatible(interface.signature(), features.signature(), name)
        if selected_storage is not None:
            storage[name] = selected_storage
        interfaces[name] = interface

    for name, binding in config.policy_bindings.items():
        if name not in interfaces:
            raise ValueError(f"policy_bindings: unknown interface {name!r}")
        interfaces[name] = interfaces[name].model_copy(update={"policy": binding})

    if config.model.backbone is None:
        raise ValueError("model backbone has not been resolved")
    execution = ModelExecution(
        config.model.backbone,
        interfaces,
        training,
        embedding_layer_names=config.run.export.embedding_layer_names,
        interface_storage=storage,
    )
    if history_only:
        validate_history_transfer(source, execution, history_interface)
    curriculum = supervised[0].curriculum if supervised else None
    resolved = ResolvedSequifierConfig(
        run=config.run,
        execution=execution,
        model_metadata_paths=metadata_paths,
        dataset_training=datasets,
        training_plan=resolved_phases,
        initialize_from=config.initialize_from,
        actor_initialization=config.actor_initialization,
        policy_data=policy_data if has_policy else None,
        curriculum_training=curriculum is not None,
        curriculum_column=curriculum.column if curriculum else None,
        run_mode=training_run_mode(resolved_phases),
    )
    if has_policy:
        resolved._prepared_inputs = prepared_inputs
    from sequifier.config.input_contract_resolution import validate_input_contracts
    from sequifier.io.output_paths import resolve_training_output_paths

    validate_input_contracts(config, resolved)
    resolve_training_output_paths(resolved.run)
    return resolved


def policy_run_config(
    config: ResolvedSequifierConfig,
    phases,
    *,
    index=0,
    resume=None,
):
    """Select resolved inputs for one native policy lifecycle."""
    from sequifier.artifacts.run_checkpoint import checkpoint_path
    from sequifier.config.components import ResumeConfig
    from sequifier.rl.run_config import ResolvedPolicyRun

    names = {phase.model_interface for phase in phases}
    if len(names) != 1:
        raise ValueError("a native policy lifecycle requires one interface")
    name = next(iter(names))
    actor_initialization = config.actor_initialization if index == 0 else "attach"
    resume = resume or config.run.recovery.resume or ResumeConfig()
    path = str(checkpoint_path(config, resume=resume))
    resume = resume.model_copy(update={"checkpoint_path": path})
    return ResolvedPolicyRun(
        parent=config,
        phase_offset=index,
        policy=config.policy_binding(name),
        phases=tuple(phases),
        initialize_from=config.initialize_from if index == 0 else None,
        actor_initialization=actor_initialization,
        resume=resume,
        checkpoint_path=path,
    )
