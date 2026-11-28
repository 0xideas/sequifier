import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from sequifier.config.column_names import NEXT_ACTION_OFFSET, PREDICTION_COORDINATES
from sequifier.config.composition import (
    load_composed_yaml_config,
    merge_config_fragments,
)
from sequifier.config.image_config import MediaPath
from sequifier.config.inference_common import (
    InferenceExecutionConfig,
    ResolvedInferenceModel,
    inference_image_store,
)
from sequifier.config.metadata import (
    DatasetMetadata,
    FeatureSelection,
    load_dataset_metadata,
    validate_authored_metadata,
    validate_persisted_metadata_versions,
)
from sequifier.config.model_interface import (
    ResolvedInterfaceStorage,
    ResolvedModelInterface,
)
from sequifier.config.policy_infer_config import (
    PolicyInferenceConfig,
    ResolvedPolicyInferenceConfig,
)
from sequifier.config.rollout_config import RolloutConfig
from sequifier.helpers import (
    ModelWindowView,
    StoredWindowLayout,
    canonicalize_polars_dtype_name,
    derive_target_column_types,
    normalize_path,
    resolve_window_view,
    try_catch_excess_keys,
)
from sequifier.io.window_placement import PredictionAlignment
from sequifier.objectives import (
    ALLOWED_OBJECTIVE_NAMES,
    OBJECTIVE_NAME_MESSAGE,
    get_objective_class,
    target_offset_for_objective,
)
from sequifier.typechecking import beartype


def _is_temporal_dtype_name(dtype_name: str) -> bool:
    """Return whether a metadata dtype is a supported timestamp type."""
    return dtype_name.strip().startswith(("Date", "Datetime"))


def _canonicalize_inference_dtype_name(dtype_name: str) -> str:
    """Preserve timestamp metadata while canonicalizing model-compatible dtypes."""
    if _is_temporal_dtype_name(dtype_name):
        return dtype_name.strip()
    return canonicalize_polars_dtype_name(dtype_name)


def _reject_temporal_model_columns(
    columns: list[str], column_data_types: dict[str, str]
) -> None:
    temporal = [
        name
        for name in dict.fromkeys(columns)
        if name in column_data_types
        and _is_temporal_dtype_name(column_data_types[name])
    ]
    if temporal:
        raise ValueError(
            f"Timestamp columns cannot be used as inference inputs or targets: {temporal}"
        )


@beartype
def _comparable_value(field: str, value: Any) -> Any:
    if field == "column_data_types" and isinstance(value, dict):
        return {
            column: _canonicalize_inference_dtype_name(dtype)
            for column, dtype in value.items()
        }
    return value


def bind_indexed_input_resources(
    metadata: DatasetMetadata, stored: DatasetMetadata
) -> DatasetMetadata:
    """Use view-local resources with the model's complete feature transforms."""
    return metadata.model_copy(
        update={
            **stored.select_media(
                metadata.image_sources, metadata.preconditioning_image is not None
            ),
            "static_features_path": stored.static_features_path,
            "source_observations_path": stored.source_observations_path,
        }
    )


@beartype
def _merge_loaded_execution_values(
    values: dict[str, Any], loaded: dict[str, Any], source: str
) -> None:
    """Fill omitted model fields and reject authored compatibility mismatches."""

    for field, loaded_value in loaded.items():
        configured_value = values.get(field)
        if configured_value is not None and _comparable_value(
            field, configured_value
        ) != _comparable_value(field, loaded_value):
            raise ValueError(
                f"Inference config field {field!r} does not match {source}: "
                f"configured {configured_value!r}, loaded {loaded_value!r}."
            )
        values[field] = loaded_value


def _execution_values(
    settings: Any, interface: Any, *, policy: bool = False
) -> dict[str, Any]:
    return {
        "model_interface": interface.name,
        "input_columns": list(interface.input_columns),
        "static_columns": (
            list(interface.static_conditioning.columns)
            if interface.static_conditioning
            else []
        ),
        "target_columns": list(interface.target_columns),
        "column_data_types": dict(interface.column_data_types),
        "target_column_types": dict(interface.target_column_types),
        "training_objective": settings.training_objective,
        "context_length": settings.context_length,
        "target_offset": NEXT_ACTION_OFFSET if policy else settings.target_offset,
        "prediction_length": interface.decoder.prediction_length,
        "decoder_support_length": interface.decoder.support_length,
    }


InferenceMetadataLayout = Literal["raw", "indexed", "stored"]


def _metadata_layout(request, view) -> InferenceMetadataLayout:
    if view is not None:
        return "raw" if view.view["view"] == "predict" else "indexed"
    if request.input_layout == "raw":
        return "raw"
    return "stored"


@dataclass(frozen=True)
class InferenceMetadataSource:
    interface: ResolvedModelInterface
    storage: ResolvedInterfaceStorage | None
    label: str

    @property
    def media(self):
        selection = self.interface.feature_selection
        return selection.images, selection.use_prefix

    def validate(self, metadata: DatasetMetadata, layout: InferenceMetadataLayout):
        selection = self.interface.feature_selection
        columns = selection.columns
        expected = self.interface.signature()
        actual = selection.resolve(metadata).signature()
        if layout == "raw" and self.storage is not None:
            expected.update(self.storage.raw_input_contract(columns))
            actual.update(metadata.raw_input_contract(columns))
        for field, value in expected.items():
            found = actual[field]
            if field in {
                "column_data_types",
                "n_classes",
                "id_maps",
                "selected_columns_statistics",
            }:
                found = {name: found.get(name) for name in value}
            if value != found:
                raise ValueError(
                    f"Inference metadata field {field!r} does not match {self.label}."
                )
        if layout == "stored" and self.storage is not None:
            stored = self.storage.storage_layout
            if (metadata.max_target_offset, metadata.stored_window_layout_version) != (
                stored.max_target_offset,
                stored.version,
            ):
                raise ValueError(
                    f"Inference metadata storage_layout does not match {self.label}."
                )


def validate_policy_metadata(
    execution, interface_name, path, *, layout: InferenceMetadataLayout = "raw"
):
    source = InferenceMetadataSource(
        execution.interface(interface_name),
        execution.interface_storage.get(interface_name),
        "policy metadata",
    )
    source.validate(load_dataset_metadata(str(path)), layout)


@beartype
def _assert_folder_metadata_matches(
    data_path: str,
    selected_metadata: DatasetMetadata,
    selection: FeatureSelection,
    *,
    allow_wider_layout: bool = False,
) -> DatasetMetadata | None:
    """Validate a preprocessed folder's contract against selected metadata."""
    if not os.path.isdir(data_path):
        return None

    metadata_path = os.path.join(data_path, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise ValueError(
            f"Inference data folder {data_path!r} has no metadata.json; "
            "the stored-window contract cannot be validated."
        )
    with open(metadata_path) as metadata_file:
        folder_values = json.load(metadata_file)
    validate_persisted_metadata_versions(folder_values)
    required_semantics = {
        "id_maps",
        "selected_columns_statistics",
        "normalize_real_columns",
    }
    missing_semantics = sorted(required_semantics - folder_values.keys())
    if missing_semantics:
        raise ValueError(
            f"Inference data folder {data_path!r} has metadata.json missing "
            f"semantic fields {missing_semantics}; reprocess this data to verify "
            "its category and normalization meanings."
        )
    folder_metadata = load_dataset_metadata(metadata_path)

    fields = selection.columns, selection.images, selection.use_prefix
    expected = selected_metadata.feature_contract(*fields)
    actual = folder_metadata.feature_contract(*fields)
    mismatches = [key for key, value in expected.items() if actual[key] != value]
    if selection.static and (
        not set(selection.static) <= set(folder_metadata.static_columns)
        or folder_metadata.static_features_path is None
    ):
        mismatches.append("static_features")
    expected_layout = selected_metadata.storage_layout
    if folder_metadata.split_index is not None:
        expected_layout = selected_metadata.layout_for_split(
            folder_metadata.split_index
        )
    folder_layout = folder_metadata.storage_layout
    layout_matches = folder_layout == expected_layout or (
        allow_wider_layout
        and folder_layout.window_length > expected_layout.window_length
        and folder_layout.max_target_offset == expected_layout.max_target_offset
        and folder_layout.version == expected_layout.version
    )
    if not layout_matches:
        mismatches.append("storage_layout")
    if (
        folder_metadata.prediction_aligned_splits
        != selected_metadata.prediction_aligned_splits
        or folder_metadata.prediction_length != selected_metadata.prediction_length
        or folder_metadata.target_offset != selected_metadata.target_offset
        or folder_metadata.prediction_alignment_steps
        != selected_metadata.prediction_alignment_steps
        or (
            folder_metadata.split_index is not None
            and selected_metadata.split_index is not None
            and folder_metadata.split_index != selected_metadata.split_index
        )
    ):
        mismatches.append("prediction_alignment")
    if folder_metadata.tensor_payload_kind != selected_metadata.tensor_payload_kind:
        mismatches.append("tensor_payload_kind")
    if mismatches:
        raise ValueError(
            f"Inference data folder {data_path!r} has metadata fields that do not "
            f"match the selected metadata: {mismatches}."
        )
    return folder_metadata


def _resolve_policy_inference(
    request: PolicyInferenceConfig,
) -> ResolvedPolicyInferenceConfig:
    (model_path,) = request.model_paths
    if model_path.lower().endswith(".onnx") and request.device.split(":")[0] not in {
        "cpu",
        "cuda",
    }:
        raise ValueError("ONNX policy inference requires CPU or CUDA")
    model = ResolvedInferenceModel.load(
        normalize_path(model_path, request.project_root),
        interface_name=request.model_interface,
        task=request.task,
    )
    execution, selected = model.execution, model.interface_name
    assert execution is not None and selected is not None
    interface = execution.interface(selected)
    policy = interface.policy
    if policy is None:
        raise ValueError("Selected interface has no policy binding")
    if (
        request.rollout is not None
        and request.rollout.environment.units != policy.units
    ):
        raise ValueError("rollout environment units differ from policy artifact")
    reserved = set(PREDICTION_COORDINATES)
    if reserved & set(policy.actions) or (
        request.output_probabilities and "action_log_probability" in policy.actions
    ):
        raise ValueError(
            "policy action names overlap prediction identity/probability columns"
        )
    if request.output_probabilities and any(
        len({str(value) for value in field.values}) != len(field.values)
        for field in policy.actions.values()
        if field.type == "categorical"
    ):
        raise ValueError("action labels must have unique probability column names")
    authored = request.model_dump(exclude_unset=True)
    _merge_loaded_execution_values(
        authored,
        _execution_values(execution.options, interface, policy=True),
        f"policy artifact {model_path!r}",
    )
    from sequifier.io.prepare_inference import (
        original_metadata_path,
        resolve_inference_view,
    )

    root = Path(request.project_root)
    metadata = None
    if request.data_path is None:
        if request.input_layout == "raw" or not request.metadata_config_path:
            raise ValueError("Select input.path or a split from preprocessing metadata")
        metadata = load_dataset_metadata(str(root / request.metadata_config_path))
        authored["data_path"] = _metadata_input_path(metadata, request.split)
    config = ResolvedPolicyInferenceConfig.model_validate(authored)
    assert config.data_path is not None
    indexed = resolve_inference_view(config)
    metadata_path = original_metadata_path(config, indexed)
    prepared_source = None
    storage = execution.interface_storage.get(selected)
    if metadata_path is not None:
        if indexed is not None or metadata is None:
            metadata = load_dataset_metadata(str(metadata_path))
    else:
        metadata = interface.preprocessing_metadata(storage)
    layout = _metadata_layout(config, indexed)
    InferenceMetadataSource(interface, storage, "policy metadata").validate(
        metadata, layout
    )
    if layout == "stored":
        from sequifier.config.policy_train_config import SequenceSourceConfig
        from sequifier.trajectory.sources import resolve_source

        path = (root / config.data_path).resolve()
        split_paths = [(root / part).resolve() for part in metadata.split_paths]
        split = (
            metadata.split_names[split_paths.index(path)]
            if metadata_path is not None and path in split_paths
            else None
        )
        folder = _assert_folder_metadata_matches(
            str(path),
            metadata,
            interface.feature_selection,
            allow_wider_layout=metadata_path is None,
        )
        if folder is not None:
            if split is not None and folder.split_index not in (
                None,
                metadata.named_split_index(split),
            ):
                raise ValueError("Prepared folder disagrees with the selected split")
            metadata = folder
            metadata_path = path / "metadata.json"
            split = None
        source = SequenceSourceConfig(
            split=split,
            paths=[] if split is not None else [str(path)],
            metadata_config_path=str(metadata_path.resolve())
            if metadata_path
            else None,
            static_data_path=config.static_data_path,
        )
        prepared_source = resolve_source(source, root, metadata=metadata, prepared=True)
    config._model = model
    config._indexed_view = indexed
    config._prepared_source = prepared_source
    return config


@beartype
def load_inferer_config(
    config_path: str, args_config: dict, skip_metadata: bool, *, workspace_stack=None
) -> "ResolvedInferenceConfig | ResolvedPolicyInferenceConfig | None":
    """Compose inference YAML, load model defaults, and resolve metadata."""

    config_values = load_composed_yaml_config(config_path)

    cli_values = {
        key: value for key, value in args_config.items() if key != "skip_metadata"
    }
    from sequifier.config.command_ownership import apply_run_overrides

    config_values, cli_values = apply_run_overrides(config_values, cli_values)
    from sequifier.config.inference_request import REQUESTS, inference_overrides

    composed = merge_config_fragments((config_values, inference_overrides(cli_values)))
    public_request = REQUESTS.validate_python(composed)
    if workspace_stack is not None:
        from sequifier.io.inference_workspace import inference_workspace

        authored_seed = composed.get("run", {}).get("reproducibility", {}).get("seed")
        workspace = workspace_stack.enter_context(
            inference_workspace(
                public_request,
                random_seed=authored_seed == "random",
                skip_metadata=skip_metadata,
            )
        )
        if workspace is None:
            return None
        public_request.run.reproducibility.seed = workspace.record["seed"]
    return _resolve_inference_request(public_request, skip_metadata, config_path)


def _policy_training_route(values):
    """Select the same physical prepared part that the training dataset view owns."""
    from sequifier.config.train_config import load_train_config_with_source
    from sequifier.config.training_resolution import TrainingSources

    root = Path(values.get("run", {}).get("project_root", "."))
    path = (root / values["training_config_path"]).resolve()
    values["training_config_path"] = str(path)
    loaded = load_train_config_with_source(str(path), {}, False)
    config = loaded.config
    name = values.pop("dataset", None)
    if name is None and len(config.datasets) == 1:
        name = next(iter(config.datasets))
    if name not in config.datasets:
        raise ValueError(
            "Select an input.training_route.dataset from the training config"
        )
    sources = TrainingSources(
        config, Path(config.run.project_root).resolve(), loaded.metadata
    )
    dataset = config.datasets[name]
    interface = dataset.model_interface
    if interface is None and len(loaded.resolved.interfaces) == 1:
        interface = next(iter(loaded.resolved.interfaces))
    if interface:
        _merge_loaded_execution_values(
            values,
            _execution_values(
                loaded.resolved.execution.options,
                loaded.resolved.interface(interface),
                policy=True,
            ),
            f"training dataset {name!r}",
        )
    view = values.pop("part", None)
    if view is None and len(dataset.views) == 1:
        view = next(iter(dataset.views))
    if view is None and values.get("data_path"):
        view = next(iter(dataset.views))
    if view not in dataset.views:
        raise ValueError(
            f"Select an input.training_route.view from {list(dataset.views)}"
        )
    refs = sources.members[f"{name}.{view}"]
    if len(refs) != 1 and not values.get("data_path"):
        raise ValueError("A multi-part inference view requires an explicit input.path")
    ref = f"{name}.{refs[0]}"
    part = sources.parts[ref]
    if part.metadata_path is not None:
        values.setdefault("metadata_config_path", str(part.metadata_path.resolve()))
    if not values.get("data_path"):
        if part.indexed is None and part.spec.type not in {
            "indexed_sequence",
            "prepared",
        }:
            raise ValueError(
                "A policy training route requires a prepared view or explicit input.path"
            )
        path = sources.split_path(ref)
        if path is None:
            raise ValueError(f"{ref}: training view has no prepared data path")
        values["data_path"] = str(Path(path).resolve())
        if hasattr(part.spec, "read_format"):
            values.setdefault("read_format", part.spec.read_format)


def _resolve_inference_request(public_request, skip_metadata, config_path):
    authored_values = public_request.resolver_values()
    inline_metadata = authored_values.pop("inline_metadata", None)
    if inline_metadata is not None:
        if not skip_metadata:
            raise ValueError("input.inline_metadata requires --skip-metadata")
        validate_authored_metadata(inline_metadata)
    if authored_values.get("task") == "policy":
        if skip_metadata:
            raise ValueError("skip_metadata is not supported for policy inference")
        if authored_values.get("training_config_path"):
            _policy_training_route(authored_values)
        return _resolve_policy_inference(
            PolicyInferenceConfig.model_validate(authored_values)
        )

    if route_path := authored_values.get("training_config_path"):
        root = authored_values.get("run", {}).get("project_root", ".")
        authored_values["training_config_path"] = str(
            (Path(root) / route_path).resolve()
        )
    models = []
    request = InferenceConfig.model_validate(authored_values)
    authored_values["run"] = request.run.model_dump(mode="python", exclude_unset=True)

    project_root = request.project_root
    selected_metadata = None
    selected_metadata_path = None
    sources: list[InferenceMetadataSource] = []

    dataset_name = request.dataset
    training_config_path = request.training_config_path
    training_source = None
    if training_config_path is not None:
        from sequifier.config.train_config import load_train_config

        training = load_train_config(training_config_path, {}, False)
        datasets = training.dataset_training
        if dataset_name is not None:
            if dataset_name not in datasets:
                raise ValueError(f"Unknown inference dataset {dataset_name!r}")
            interface = training.dataset_interface(dataset_name)
            storage = training.dataset_storage(dataset_name)
            source = f"training config dataset {dataset_name!r}"
        elif authored_values.get("model_interface") is not None:
            interface_name = authored_values["model_interface"]
            if interface_name not in training.interfaces:
                raise ValueError(f"Unknown model interface {interface_name!r}")
            interface = training.interface(interface_name)
            storage = training.execution.interface_storage.get(interface_name)
            source = f"training config interface {interface_name!r}"
        elif len(datasets) == 1:
            dataset_name = next(iter(datasets))
            interface = training.dataset_interface(dataset_name)
            storage = training.dataset_storage(dataset_name)
            source = f"training config dataset {dataset_name!r}"
        else:
            raise ValueError(
                "A dataset or model_interface selection is required for the "
                "training config"
            )
        training_source = InferenceMetadataSource(interface, storage, source)
        sources.append(training_source)
        _merge_loaded_execution_values(
            authored_values,
            _execution_values(training.execution.options, interface),
            source,
        )

        if dataset_name is not None:
            dataset = datasets[dataset_name]
            part_name = authored_values.get("part")
            if part_name is None and len(dataset.views) == 1:
                part_name = next(iter(dataset.views))
            if part_name is None:
                if request.data_path is None:
                    raise ValueError(
                        f"Select an input.training_route.view from {list(dataset.views)}"
                    )
                # The input supplies its own rows. Route metadata supplies the codec.
                part = next(iter(dataset.parts.values()))
            else:
                if part_name not in dataset.views:
                    raise ValueError(f"Unknown input view {dataset_name}.{part_name}")
                members = dataset.views[part_name]
                if len(members) != 1 and request.data_path is None:
                    raise ValueError(
                        "A multi-part inference view requires an explicit input.path"
                    )
                part = dataset.parts[members[0]]
                if request.data_path is None:
                    authored_values["data_path"] = part.view_path
                    authored_values.setdefault("read_format", part.read_format)
            selected_metadata = part.metadata
            selected_metadata_path = part.metadata_config_path

    autoregressive_decoder_targets: set[str] = set()
    for model_path in request.model_paths:
        model = ResolvedInferenceModel.load(
            normalize_path(model_path, project_root),
            interface_name=authored_values.get("model_interface"),
            task=request.task,
        )
        models.append(model)
        execution, interface_name = model.execution, model.interface_name
        if execution is None:
            continue
        assert interface_name is not None
        interface = execution.interface(interface_name)
        decoder = interface.decoder
        branches = (
            decoder.branches.values() if decoder.type == "composite" else (decoder,)
        )
        autoregressive_decoder_targets.update(
            target
            for branch in branches
            if branch.type == "autoregressive_transformer"
            for target in branch.target_columns
        )
        label = f"Model artifact {model_path!r}"
        sources.append(
            InferenceMetadataSource(
                interface, execution.interface_storage.get(interface_name), label
            )
        )
        _merge_loaded_execution_values(
            authored_values, _execution_values(execution.options, interface), label
        )

    config = try_catch_excess_keys(config_path, InferenceConfig, authored_values)
    sampled_autoregressive = autoregressive_decoder_targets.intersection(
        config.sample_from_distribution_columns or ()
    )
    if sampled_autoregressive:
        raise ValueError(
            "External per-column sampling is unsupported for autoregressive "
            "transformer decoder targets because later distributions depend on "
            "the internally generated prefix; requested "
            f"{sorted(sampled_autoregressive)!r}."
        )
    from sequifier.io.prepare_inference import (
        original_metadata_path,
        resolve_inference_view,
    )

    indexed_view = resolve_inference_view(config, skip_metadata=skip_metadata)
    if indexed_view is not None:
        config.metadata_config_path = str(original_metadata_path(config, indexed_view))
    metadata_path = _effective_metadata_config_path(config)
    artifact_only_metadata = False
    if skip_metadata and inline_metadata is not None:
        metadata = DatasetMetadata.model_validate(inline_metadata)
    elif not skip_metadata and metadata_path is not None:
        metadata = load_dataset_metadata(
            normalize_path(metadata_path, config.project_root)
        )
    elif selected_metadata is not None:
        metadata = selected_metadata
    elif fallback := next(
        (entry for entry in sources if entry.storage is not None), None
    ):
        metadata = fallback.interface.preprocessing_metadata(fallback.storage)
        artifact_only_metadata = fallback is not training_source
    elif skip_metadata:
        raise ValueError(
            "--skip-metadata requires input.inline_metadata, a "
            "selected training dataset, or a self-describing PT artifact."
        )
    else:
        raise ValueError(
            f"Inference config '{config_path}' must define metadata_config_path "
            "unless a selected model artifact contains model "
            "metadata."
        )

    if metadata_path is None and selected_metadata_path is not None:
        config.metadata_config_path = selected_metadata_path

    if (
        indexed_view is None
        and config.input_layout != "raw"
        and config.data_path is None
        and metadata.split_paths
    ):
        config.data_path = _metadata_input_path(metadata, config.split)
        indexed_view = resolve_inference_view(config, skip_metadata=skip_metadata)
        if indexed_view is not None:
            config.metadata_config_path = str(
                original_metadata_path(config, indexed_view)
            )
            metadata = load_dataset_metadata(config.metadata_config_path)

    if sources:
        media = sources[0].media
        if any(entry.media != media for entry in sources[1:]):
            raise ValueError("Inference model artifacts select different image views")
        metadata = metadata.model_copy(update=metadata.select_media(*media))
    if config.input_layout == "indexed" and indexed_view is None:
        raise ValueError("Indexed inference requires an indexed split manifest")
    layout = _metadata_layout(config, indexed_view)
    for entry in sources:
        entry.validate(metadata, layout)
    return _resolve_inference_config(
        config,
        metadata,
        models=tuple(models),
        allow_wider_artifact_folder=artifact_only_metadata,
        indexed_view=indexed_view,
    )


def _metadata_input_path(metadata, split):
    names = metadata.split_names
    if split is None:
        if len(names) == 1:
            split = names[0]
        elif "test" in names:
            split = "test"
        else:
            raise ValueError(f"Select input.path or a named input.split from {names}")
    return metadata.split_paths[metadata.named_split_index(split)]


@beartype
def _effective_metadata_config_path(config: "InferenceConfig") -> Optional[str]:
    if config.metadata_config_path:
        return config.metadata_config_path
    return None


@beartype
def _resolve_inference_config(
    config: Any,
    metadata: DatasetMetadata,
    *,
    models: tuple[ResolvedInferenceModel, ...],
    allow_wider_artifact_folder: bool = False,
    indexed_view: Any = None,
) -> "ResolvedInferenceConfig":
    """Return an inference config with all metadata-derived values populated."""

    if config.context_length is None or config.training_objective is None:
        raise ValueError("Inference requires context_length and training_objective")

    fresh = config.input_layout == "raw" or (
        indexed_view is not None and indexed_view.view["view"] == "predict"
    )
    if (
        not fresh
        and config.prediction_alignment_steps != metadata.prediction_alignment_steps
    ):
        raise ValueError(
            "Inference config prediction_alignment_steps does not match dataset "
            "metadata: configured "
            f"{config.prediction_alignment_steps}, "
            f"metadata {metadata.prediction_alignment_steps}."
        )
    storage_layout = metadata.storage_layout
    available_column_data_types = config.column_data_types or metadata.column_data_types
    input_columns = (
        [
            column
            for column, dtype in available_column_data_types.items()
            if not _is_temporal_dtype_name(dtype)
            and column not in metadata.static_columns
        ]
        if config.input_columns is None
        else config.input_columns
    )
    static_columns = config.static_columns
    selection = FeatureSelection(
        input_columns,
        config.target_columns,
        static_columns,
        set(metadata.image_sources),
        metadata.preconditioning_image is not None,
    )
    _reject_temporal_model_columns(selection.columns, available_column_data_types)
    features = selection.resolve(
        metadata.model_copy(update={"column_data_types": available_column_data_types})
    )
    column_data_types = features.column_data_types
    categorical_columns, real_columns = features.split_columns(input_columns)
    if (
        not categorical_columns
        and not real_columns
        and not metadata.image_sources
        and metadata.preconditioning_image is None
    ):
        raise ValueError("No columns found in resolved inference config")

    if features.depth_layouts:
        if (
            config.read_format != "pt"
            and indexed_view is None
            and config.input_layout != "raw"
        ):
            raise ValueError("Preprocessed depth inputs require read_format: pt")
    target_column_types = config.target_column_types or derive_target_column_types(
        config.target_columns, column_data_types
    )
    window_view = ModelWindowView(
        context_length=config.context_length,
        objective=config.training_objective,
        target_offset=target_offset_for_objective(
            config.training_objective, config.target_offset
        ),
    )
    if config.data_path is None and not metadata.split_paths:
        raise ValueError(
            "Resolved inference config needs data_path when metadata does not "
            "provide split_paths."
        )
    data_path = config.data_path or _metadata_input_path(metadata, config.split)
    normalized_data_path = normalize_path(data_path, config.project_root)
    if config.input_layout == "raw":
        normalized_data_path = os.path.abspath(normalized_data_path)
    placement_metadata = (
        indexed_view.metadata() if indexed_view is not None else metadata
    )
    normalized_splits = [
        normalize_path(path, config.project_root)
        for path in placement_metadata.split_paths
    ]
    folder_metadata = (
        None
        if config.input_layout == "raw"
        else _assert_folder_metadata_matches(
            normalized_data_path,
            metadata,
            selection,
            allow_wider_layout=allow_wider_artifact_folder,
        )
    )
    media_metadata = metadata
    if folder_metadata is not None:
        media_metadata = metadata.model_copy(
            update={
                **folder_metadata.select_media(selection.images, selection.use_prefix),
                "source_observations_path": folder_metadata.source_observations_path,
            }
        )
    split_index = (
        normalized_splits.index(normalized_data_path)
        if normalized_data_path in normalized_splits
        else metadata.split_index
    )
    if split_index is None and folder_metadata is not None:
        split_index = folder_metadata.split_index
    if split_index is not None:
        storage_layout = metadata.layout_for_split(split_index)
    if (
        allow_wider_artifact_folder
        and folder_metadata is not None
        and folder_metadata.window_length > storage_layout.window_length
    ):
        storage_layout = folder_metadata.storage_layout
    if fresh:
        storage_layout = StoredWindowLayout(
            window_view.context_length + window_view.target_offset,
            window_view.target_offset,
            storage_layout.version,
        )
    elif indexed_view is not None:
        split_index = int(indexed_view.view["view"])
        storage_layout = indexed_view.layout(window_view)
    if indexed_view is not None:
        media_metadata = bind_indexed_input_resources(metadata, placement_metadata)
    resolve_window_view(storage_layout, window_view)
    if not fresh and metadata.prediction_aligned_splits and not normalized_splits:
        folder_index = None
        if os.path.isdir(normalized_data_path):
            match = re.search(r"-split(\d+)$", os.path.basename(normalized_data_path))
            if match is not None:
                folder_index = int(match.group(1))
        if (
            split_index is not None
            and folder_index is not None
            and split_index != folder_index
        ):
            raise ValueError("Metadata split_index does not match data folder")
        split_index = split_index if split_index is not None else folder_index
        if split_index is None:
            raise ValueError(
                "Prediction aligned inference requires split_paths or a split_index "
                "in metadata (or a standard -splitN data folder)"
            )
    alignment = (
        None
        if fresh
        else PredictionAlignment.for_split(placement_metadata, split_index)
    )
    if alignment:
        alignment.validate_width(
            storage_layout.window_length, window_view, normalized_data_path
        )
        if config.autoregressive_generation:
            raise ValueError(
                "Prediction aligned inference does not support autoregressive output"
            )
        prediction_length = config.prediction_length
        if prediction_length is None:
            prediction_length = get_objective_class(
                config.training_objective
            ).default_prediction_length(window_view.context_length)
        alignment.validate_decoder(window_view, prediction_length, normalized_data_path)
        if config.window_stride is not None:
            raise ValueError(
                "Prediction aligned inference requires window_stride: null"
            )
    values = config.model_dump(mode="python")
    values["run"] = config.run.model_dump(mode="python", exclude_unset=True)
    values.update(
        {
            "metadata_config_path": _effective_metadata_config_path(config),
            "data_path": normalized_data_path,
            "input_columns": input_columns,
            "static_columns": static_columns,
            "static_features_path": (
                config.static_features_path or (media_metadata.static_features_path)
            ),
            "column_data_types": column_data_types,
            "categorical_columns": categorical_columns,
            "real_columns": real_columns,
            "target_column_types": target_column_types,
            "storage_layout": storage_layout,
            "window_view": window_view,
            "dataset_metadata": media_metadata,
            "aligned_data_path": alignment is not None,
        }
    )
    resolved = ResolvedInferenceConfig.model_validate(values)
    resolved._models = models
    resolved._indexed_view = indexed_view
    return resolved


_DataPathT = TypeVar("_DataPathT")
_InputColumnsT = TypeVar("_InputColumnsT")
_ColumnTypesT = TypeVar("_ColumnTypesT")
_ObjectiveT = TypeVar("_ObjectiveT", bound=str | None)
_ContextLengthT = TypeVar("_ContextLengthT", bound=int | None)


class _InferenceConfigBase(
    InferenceExecutionConfig,
    Generic[_DataPathT, _InputColumnsT, _ColumnTypesT, _ObjectiveT, _ContextLengthT],
):
    """Shared fields and validation for authored and resolved inference config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    task: Literal["forecast", "embedding"]
    training_objective: _ObjectiveT
    data_path: _DataPathT
    fresh_prediction_positions: Literal["right_edge", "observed"] | None = None
    training_config_path: Optional[str] = None
    dataset: Optional[str] = None
    part: Optional[str] = None

    input_columns: _InputColumnsT
    static_columns: list[str] = Field(default_factory=list)
    static_features_path: Optional[str] = None
    image_paths: MediaPath = None
    preconditioning_image_paths: MediaPath = None
    target_columns: list[str]
    column_data_types: _ColumnTypesT
    target_column_types: _ColumnTypesT

    output_probabilities: bool = Field(default=False)
    decode_categories: bool = Field(default=True)
    context_length: _ContextLengthT = Field(gt=0)
    target_offset: int = Field(default=1, ge=0)
    window_stride: Optional[int] = Field(default=None, gt=0)
    prediction_length: Optional[int] = None
    decoder_support_length: int = Field(default=1, gt=0)
    prediction_alignment_steps: int = Field(default=1, gt=0)

    sample_from_distribution_columns: Optional[list[str]] = Field(default=None)
    infer_with_dropout: bool = Field(default=False)
    rollout: RolloutConfig | None = None
    autoregressive_generation: bool = Field(default=False)
    generation_steps: Optional[int] = Field(default=None)

    def input_paths(self) -> list[str]:
        inputs = super().input_paths()
        if (
            self.rollout
            and (metadata := getattr(self, "dataset_metadata", None))
            and metadata.source_observations_path
        ):
            inputs.append(
                normalize_path(metadata.source_observations_path, self.project_root)
            )
        if self.static_features_path:
            inputs.append(normalize_path(self.static_features_path, self.project_root))
        if self.training_config_path:
            inputs.append(self.training_config_path)
        return inputs

    @model_validator(mode="after")
    @beartype
    def validate_route_selection(self):
        for usage, value in (
            ("dataset", self.dataset),
            ("part", self.part),
        ):
            if value is not None and ("." in value or not value.isidentifier()):
                raise ValueError(f"{usage} must be a valid identifier without '.'")
        return self

    @field_validator("training_objective")
    @classmethod
    @beartype
    def validate_authored_training_objective(cls, value: str | None) -> str | None:
        if value is not None and value not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(
                f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {value}"
            )
        return value

    @model_validator(mode="after")
    @beartype
    def validate_authored_paths(self):
        if self.context_length is None or self.training_objective is None:
            return self
        ModelWindowView(
            context_length=self.context_length,
            objective=self.training_objective,
            target_offset=target_offset_for_objective(
                self.training_objective, self.target_offset
            ),
        )
        return self


class InferenceConfig(
    _InferenceConfigBase[
        Optional[str],
        Optional[list[str]],
        Optional[dict[str, str]],
        str | None,
        int | None,
    ]
):
    """User-authored configuration for one inference run."""

    @model_validator(mode="after")
    def validate_embedding_options(self):
        generative_fields = {
            "sample_from_distribution_columns",
            "decode_categories",
            "output_probabilities",
            "autoregressive_generation",
            "generation_steps",
            "rollout",
        }
        if self.task == "embedding" and (
            invalid := self.model_fields_set & generative_fields
        ):
            raise ValueError(f"embedding inference does not support {sorted(invalid)}")
        return self

    training_objective: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    input_columns: list[str] | None = None
    target_columns: list[str] = Field(default_factory=list)
    data_path: Optional[str] = None
    column_data_types: Optional[dict[str, str]] = None
    target_column_types: Optional[dict[str, str]] = None


class ResolvedInferenceConfig(
    _InferenceConfigBase[str, list[str], dict[str, str], str, int]
):
    """Internal inference config after dataset metadata has been resolved."""

    categorical_columns: list[str]
    real_columns: list[str]
    storage_layout: StoredWindowLayout
    aligned_data_path: bool = False
    window_view: ModelWindowView
    dataset_metadata: DatasetMetadata = Field(exclude=True)
    _image_store: Any = PrivateAttr(default=None)
    _models: tuple[ResolvedInferenceModel, ...] = PrivateAttr()
    _indexed_view: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_numeric_rollout(self):
        if self.rollout is None:
            return self
        if self.autoregressive_generation or self.generation_steps is not None:
            raise ValueError(
                "rollout uses max_steps, not autoregressive/generation_steps"
            )
        if self.device != "cpu" or self.task != "forecast":
            raise ValueError("numeric rollout requires a CPU generative model")
        if (
            self.training_objective not in {"causal", "final_value"}
            or self.window_view.target_offset != 1
            or self.prediction_length not in (None, 1)
        ):
            raise ValueError(
                "forecast rollout requires a one-step causal/final_value forecast with target_offset=1"
            )
        if self.output_probabilities or self.sample_from_distribution_columns:
            raise ValueError(
                "numeric rollout does not support categorical probabilities or sampling"
            )
        paths = self.model_paths
        if len(paths) != 1:
            raise ValueError("rollout requires exactly one entry in model_paths")
        fields = self.input_columns + self.static_columns + self.target_columns
        if not self.input_columns or any(
            "float" not in self.column_data_types[column].lower() for column in fields
        ):
            raise ValueError(
                "numeric rollout supports real scalar inputs, static inputs, and targets only"
            )
        if set(self.input_columns) & set(self.static_columns):
            raise ValueError("rollout static and changing inputs must be disjoint")
        if any(kind != "real" for kind in self.target_column_types.values()):
            raise ValueError("numeric rollout requires real forecast targets")
        metadata = self.dataset_metadata
        if (
            metadata.image_sources
            or metadata.preconditioning_image is not None
            or metadata.depth_layouts.relevant_layouts(fields)
        ):
            raise ValueError("numeric rollout does not support image or depth inputs")
        self.rollout.validate_bindings(self.input_columns, self.target_columns)
        return self

    def image_store(self):
        if self._image_store is None:
            self._image_store = inference_image_store(
                self, self.dataset_metadata, self._indexed_view
            )
        return self._image_store

    @model_validator(mode="before")
    @classmethod
    @beartype
    def derive_optional_config_values(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        window_view = values.get("window_view")
        if "context_length" not in values and isinstance(window_view, dict):
            values["context_length"] = window_view.get("context_length")
        elif "context_length" not in values and isinstance(
            window_view, ModelWindowView
        ):
            values["context_length"] = window_view.context_length
        if "target_offset" not in values and isinstance(window_view, dict):
            values["target_offset"] = window_view.get("target_offset", 1)
        elif "target_offset" not in values and isinstance(window_view, ModelWindowView):
            values["target_offset"] = window_view.target_offset
        if (
            values.get("target_column_types") is None
            and values.get("column_data_types") is not None
        ):
            values["target_column_types"] = derive_target_column_types(
                values.get("target_columns", []),
                values["column_data_types"],
            )
        return values

    @model_validator(mode="after")
    @beartype
    def validate_required_paths(self):
        if self.data_path is None:
            raise ValueError(
                "data_path must be provided or resolved from preprocessing metadata"
            )
        return self

    @model_validator(mode="after")
    @beartype
    def normalize_prediction_length(self):
        if self.window_view.objective != self.training_objective:
            raise ValueError(
                "window_view objective must match training_objective "
                f"({self.window_view.objective} != {self.training_objective})."
            )
        objective_class = get_objective_class(self.training_objective)
        if self.prediction_length is None:
            self.prediction_length = objective_class.default_prediction_length(
                self.window_view.context_length
            )
        if objective_class.forward_looking:
            decoded_length = (
                self.window_view.context_length - self.decoder_support_length + 1
            )
            if not 1 <= self.prediction_length <= decoded_length:
                raise ValueError(
                    "Forward inference prediction_length must be between 1 and "
                    f"context_length - decoder_support_length + 1 ({decoded_length})"
                )
        objective_class.validate_prediction_length(
            self.prediction_length,
            self.window_view.context_length,
            usage="inference",
        )
        if objective_class.forward_looking:
            resolve_window_view(self.storage_layout, self.window_view)
        return self

    @field_validator("training_objective")
    @classmethod
    @beartype
    def validate_training_objective(cls, v):
        if v not in ALLOWED_OBJECTIVE_NAMES:
            raise ValueError(f"Only {OBJECTIVE_NAME_MESSAGE} are allowed, found {v}")
        return v

    @field_validator("output_probabilities")
    @classmethod
    @beartype
    def validate_output_probabilities(cls, v: bool, info: Any) -> bool:
        if v and info.data.get("task") == "embedding":
            raise ValueError(
                "For embedding models, 'output_probabilities' must be set to false"
            )
        return v

    @model_validator(mode="after")
    @beartype
    def validate_training_config_path(self):
        model_paths = self.model_paths
        if not any(path.lower().endswith(".pt") for path in model_paths):
            return self

        if self.training_config_path is not None and not os.path.exists(
            self.training_config_path
        ):
            raise ValueError(f"{self.training_config_path} does not exist")
        return self

    @field_validator("generation_steps")
    @classmethod
    @beartype
    def validate_generation_steps(cls, v: Optional[int], info: Any) -> Optional[int]:
        if v is None and info.data.get("autoregressive_generation") is True:
            raise ValueError(
                "If autoregressive_generation==True, 'generation_steps' needs to be set to an integer value."
            )
        if v is not None and v < 1:
            raise ValueError("generation_steps must by >= 1.")
        if v is not None and v > 1:
            if not info.data.get("autoregressive_generation"):
                raise ValueError(
                    f"'generation_steps' can only be larger than 1 if 'autoregressive_generation' is true: {info.data.get('autoregressive_generation')}"
                )

            if info.data.get("rollout") is None:
                inputs = info.data.get("input_columns") or []
                missing = set(inputs).difference(info.data.get("target_columns") or [])
                if missing:
                    raise ValueError(
                        "Autoregressive generation requires a prediction for each "
                        f"temporal input. Missing targets: {sorted(missing)}"
                    )

        return v

    @field_validator("autoregressive_generation")
    @classmethod
    @beartype
    def validate_autoregressive(cls, v: bool, info: Any):
        if v and info.data.get("task") == "embedding":
            raise ValueError(
                "Autoregressive inference is not possible for embedding models"
            )
        if (
            v
            and info.data.get("prediction_length") is not None
            and info.data.get("prediction_length") > 1
        ):
            raise ValueError(
                "Autoregressive inference is not possible for models with prediction_length > 1"
            )
        if (
            v
            and info.data.get("training_objective") is not None
            and not get_objective_class(
                info.data.get("training_objective")
            ).forward_looking
        ):
            raise ValueError(
                "Autoregressive inference is not possible with BERT-style models."
            )

        return v

    @field_validator("data_path")
    @classmethod
    @beartype
    def validate_data_path(cls, v: Optional[str], info: Any) -> Optional[str]:
        if v is None:
            return v
        v2 = normalize_path(v, info.data["run"].project_root)
        if not os.path.exists(v2):
            raise ValueError(f"{v2} does not exist")
        return v

    @field_validator("target_column_types")
    @classmethod
    @beartype
    def validate_target_column_types(
        cls, v: Optional[dict], info: Any
    ) -> Optional[dict]:
        if v is None:
            return v
        if not all(vv in ["categorical", "real"] for vv in v.values()):
            raise ValueError(
                "Target column types must be either 'categorical' or 'real'"
            )
        if list(v.keys()) != info.data.get("target_columns", []):
            raise ValueError(
                "target_columns and target_column_types must contain the same keys in the same order"
            )
        return v

    @field_validator("column_data_types")
    @classmethod
    @beartype
    def validate_column_types(cls, v: Optional[dict], info: Any) -> Optional[dict]:
        if v is None:
            return v
        normalized = {
            column: _canonicalize_inference_dtype_name(dtype)
            for column, dtype in v.items()
        }
        input_columns = info.data.get("input_columns") or []
        missing_input_columns = [
            column for column in input_columns if column not in normalized
        ]
        if missing_input_columns:
            raise ValueError(
                "column_data_types must include every input column. "
                f"Missing: {missing_input_columns}"
            )
        return normalized

    @model_validator(mode="after")
    @beartype
    def reject_temporal_model_columns(self):
        if self.column_data_types is None:
            return self
        selected_columns = [
            *self.input_columns,
            *self.target_columns,
            *self.static_columns,
        ]
        if missing := set(self.target_columns) - self.column_data_types.keys():
            raise ValueError(
                f"column_data_types must include every target: {sorted(missing)}"
            )
        _reject_temporal_model_columns(selected_columns, self.column_data_types)
        return self
