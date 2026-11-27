"""Apply an existing preprocessing contract to fresh raw data."""

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from sequifier.config.column_names import INCOMING_ACTION_PRESENT
from sequifier.config.image_config import bind_image_paths
from sequifier.config.metadata import load_dataset_metadata
from sequifier.config.preprocess_config import PreprocessorModel
from sequifier.io.categorical import is_hash_bucket_label
from sequifier.io.indexed_sequences import (
    IndexedSequenceWriter,
    IndexedView,
    is_indexed_view,
)


def is_sequence_input(path, read_format):
    return path.is_file() and (
        read_format == "sequence" or path.suffix.lower() == ".json"
    )


def prepare_inference(config, *, required_columns=(), optional_passthrough=()):
    root = Path(config.run.project_root).resolve()
    return _prepare_inference(
        config,
        root=root,
        raw_path=root / config.preprocessing_data_path,
        destination=root / config.output_path,
        read_format=config.read_format,
        num_workers=config.run.execution.data_loading.num_workers,
        required_columns=required_columns,
        passthrough_columns=config.passthrough_columns,
        optional_passthrough=optional_passthrough,
    )


def _prepare_inference(
    config,
    *,
    root,
    raw_path,
    destination,
    read_format,
    required_columns,
    passthrough_columns,
    optional_passthrough,
    num_workers=1,
    sequence_table=None,
):
    from sequifier.preprocess import Preprocessor, _folder_input_files

    metadata_path = root / config.metadata_config_path
    original_contents = metadata_path.read_bytes()
    metadata = load_dataset_metadata(str(metadata_path))
    if metadata_path.read_bytes() != original_contents:
        raise ValueError(
            "The original preprocessing metadata changed during preparation"
        )
    persisted_mask = metadata.raw_mask_column
    if persisted_mask is not None and config.mask_column not in (None, persisted_mask):
        raise ValueError("mask_column differs from the original preprocessing metadata")
    mask_column = config.mask_column or persisted_mask
    if sequence_table is None and is_sequence_input(raw_path, read_format):
        from sequifier.trajectory.sequence_output import supervised_manifest_input

        with supervised_manifest_input(
            raw_path,
            mask_column=mask_column,
        ) as table:
            return _prepare_inference(
                config,
                root=root,
                raw_path=raw_path,
                destination=destination,
                read_format="parquet",
                required_columns=required_columns,
                passthrough_columns=passthrough_columns,
                optional_passthrough=optional_passthrough,
                num_workers=num_workers,
                sequence_table=table,
            )
    read_format = read_format or raw_path.suffix.lower().lstrip(".")
    if read_format not in {"csv", "parquet"}:
        raise ValueError("Fresh raw folders require read_format: csv or parquet")
    static_path = (
        str(root / config.static_data_path) if config.static_data_path else None
    )
    manifest = None
    if sequence_table is not None:
        raw_path, manifest, recorded_static = sequence_table
        if static_path is None and recorded_static is not None:
            static_path = str(recorded_static)
    files = (
        [str(raw_path)]
        if raw_path.is_file()
        else _folder_input_files(str(raw_path), read_format, static_path)
    )
    if not files:
        raise ValueError("Fresh data requires at least one input file")
    scan = pl.scan_csv if read_format == "csv" else pl.scan_parquet
    schemas = [scan(file).collect_schema() for file in files]
    available = set.intersection(*(set(schema.names()) for schema in schemas))
    from sequifier.io.sequence import DOMAIN_OBSERVATIONS, DOMAIN_STATIC

    if any(
        {DOMAIN_OBSERVATIONS, DOMAIN_STATIC}.intersection(schema) for schema in schemas
    ):
        raise ValueError("Fresh data uses reserved rollout seed column names")
    required_columns = set(required_columns)
    missing = required_columns - available
    missing |= {"sequenceId", "itemPosition"} - available
    if missing:
        raise ValueError(f"Fresh data is missing model columns: {sorted(missing)}")
    optional = set(optional_passthrough)
    for schema in schemas:
        if partial := (optional & set(schema.names())) - available:
            raise ValueError(
                f"Auxiliary columns must occur in every input file: {sorted(partial)}"
            )
    passthrough = set(passthrough_columns) | ((optional & available) - required_columns)
    if manifest is not None:
        action_mapping = manifest.get("action_columns", {})
        passthrough.update(
            physical
            for name, physical in action_mapping.items()
            if name in optional and physical not in required_columns
        )
    if conflicts := passthrough & required_columns:
        raise ValueError(
            f"Raw auxiliary columns also used as model inputs: {sorted(conflicts)}"
        )
    if missing := passthrough - available:
        raise ValueError(f"Fresh data is missing auxiliary columns: {sorted(missing)}")
    temporal_columns = [
        column
        for column in metadata.column_data_types
        if column in available
        and column not in passthrough
        and column not in metadata.static_columns
        and not metadata.column_data_types[column]
        .lower()
        .startswith(("date", "datetime"))
    ]
    if manifest is not None:
        mask_column = manifest.get("mask_column") or mask_column
    if mask_column and mask_column not in available:
        raise ValueError(
            f"Fresh data is missing the original mask column {mask_column!r}"
        )
    if any(not schema["sequenceId"].is_integer() for schema in schemas):
        raise ValueError("sequenceId must contain non-null integers")
    depth = bool(metadata.depth_layouts)
    media_columns = {
        key
        for name in metadata.image_sources
        for key in (f"__image:{name}", f"__image_mask:{name}")
    }
    inline_media = False
    if (
        sequence_table is not None
        and (media_columns or metadata.preconditioning_image)
        and not (config.image_paths or config.preconditioning_image_paths)
    ):
        from sequifier.io.indexed_sequences import PREFIX_IMAGE
        from sequifier.trajectory.tables import scan_tables

        prefix_available = metadata.preconditioning_image is None or (
            static_path
            and PREFIX_IMAGE in scan_tables((Path(static_path),)).collect_schema()
        )
        inline_media = bool(media_columns <= available and prefix_available)
    sequence_provenance = dict((manifest or {}).get("provenance", {}))
    if manifest is not None:
        sequence_provenance["action_columns"] = {
            name: physical
            for name, physical in manifest.get("action_columns", {}).items()
            if physical in available
        }
    if sequence_table is not None and (depth or inline_media):
        _prepare_structured_rollout(
            raw_path,
            static_path,
            destination,
            original_contents,
            metadata,
            temporal_columns,
            passthrough,
            mask_column,
            inline_media,
            sequence_provenance=sequence_provenance,
        )
        return IndexedView.load(destination / "predict.split.json")
    image_sources, prefix = bind_image_paths(
        metadata.image_sources,
        metadata.preconditioning_image,
        config.image_paths,
        config.preconditioning_image_paths,
    )
    with IndexedSequenceWriter(destination) as writer:
        snapshot = writer.root / "original-metadata.json"
        snapshot.write_bytes(original_contents)
        options = PreprocessorModel.model_validate(
            {
                "run": {
                    "project_root": str(root),
                    "reproducibility": {"seed": config.run.reproducibility.seed},
                    "execution": {
                        "data_loading": {
                            "num_workers": 1 if num_workers is None else num_workers
                        }
                    },
                },
                "preprocessing_data_path": str(raw_path),
                "metadata_config_path": str(snapshot),
                "read_format": read_format,
                "write_format": "pt" if depth else "parquet",
                "merge_output": not depth,
                "selected_columns": temporal_columns,
                "splits": {
                    "train": {"fraction": 1.0, "window_length": metadata.window_length}
                },
                "max_target_offset": metadata.max_target_offset,
                "normalize_real_columns": metadata.normalize_real_columns,
                "static_data_path": static_path,
                "depth_layouts": metadata.depth_layouts,
                "image_sources": image_sources,
                "preconditioning_image": prefix,
                "time_alignment_column": metadata.time_alignment_column,
                "mask_column": mask_column,
                "indexed_output_path": str(destination),
                "indexed_passthrough": sorted(set(passthrough_columns)),
            }
        )
        options.indexed_passthrough = sorted(passthrough)
        run = Preprocessor(options)
        run.sequence_table_provenance = sequence_provenance
        run.run(indexed_writer=writer, indexed_prediction=True)
    return IndexedView.load(destination / "predict.split.json")


def encode_seed_context(payload, metadata, observations):
    from sequifier.io.prepare_policy import _mask_features
    from sequifier.io.sequence import CONTEXT_RECORDS

    context = CONTEXT_RECORDS.validate_json(payload)
    if not context:
        return payload
    rows = pl.DataFrame([step.observation for step in context])
    active = [
        name
        for name in observations
        if name in metadata.column_data_types and name in rows.columns
    ]
    encoded, fitted = _encode_recorded_features(rows, active, metadata)
    encoded = _mask_features(
        encoded,
        active,
        fitted,
        metadata.depth_layouts,
        masked=pl.lit(pl.Series([step.masked for step in context])),
    )
    return CONTEXT_RECORDS.dump_json(
        tuple(
            step.model_copy(update={"observation": row})
            for step, row in zip(context, encoded.to_dicts(), strict=True)
        )
    ).decode()


def _encode_recorded_features(rows, columns, metadata):
    from sequifier.io.feature_fitting import FittedColumns
    from sequifier.io.prepare_policy import _encoded_observation

    layouts = metadata.depth_layouts
    fitted = FittedColumns.from_metadata(columns, dict(metadata))
    encoded = rows.with_columns(
        _encoded_observation(
            rows,
            name,
            layouts,
            fitted.column_transform(
                name,
                rows.schema[name].inner
                if layouts.is_deep_column(name)
                else rows.schema[name],
                metadata.normalize_real_columns,
                metadata.cardinality_config,
            ),
        )
        for name in columns
    )
    return encoded, fitted


def _prepare_structured_rollout(
    source,
    static_path,
    destination,
    original,
    metadata,
    columns,
    passthrough,
    mask_column,
    inline_media,
    *,
    sequence_provenance=None,
):
    from sequifier.config.depth_layout import depth_mask_metadata_key
    from sequifier.io.indexed_sequences import PREFIX_IMAGE
    from sequifier.io.prepare_policy import _mask_features
    from sequifier.io.sequence import (
        DOMAIN_OBSERVATIONS,
        DOMAIN_STATIC,
        with_domain_record,
    )
    from sequifier.trajectory.tables import IndexedTable, observation_mask, scan_tables

    if (metadata.image_sources or metadata.preconditioning_image) and not inline_media:
        raise ValueError(
            "Structured rollout preparation requires the recorded image inputs"
        )
    layouts = metadata.depth_layouts.relevant_layouts(columns)
    extra = [depth_mask_metadata_key(name) for name in layouts.root]
    extra.extend(
        key
        for name in metadata.image_sources
        for key in (f"__image:{name}", f"__image_mask:{name}")
    )
    scan = pl.scan_parquet(source)
    mask = observation_mask(scan, mask_column)
    schema = scan.collect_schema()
    required = {*extra, *columns, *passthrough}
    if missing := required - set(schema.names()):
        raise ValueError(f"Rollout lacks structured input columns: {sorted(missing)}")
    data = scan
    if mask_column:
        data = data.with_columns(mask.alias(mask_column))
        required.add(mask_column)
    data = data.select("sequenceId", "itemPosition", *sorted(required))
    table = IndexedTable(data.sort("itemPosition"), "sequenceId")
    static = scan_tables((Path(static_path),)).collect() if static_path else None
    static_columns = metadata.static_columns
    if static_columns or metadata.preconditioning_image:
        required_static = [
            *static_columns,
            *([PREFIX_IMAGE] if metadata.preconditioning_image else []),
        ]
        if static is None or set(required_static) - set(static.columns):
            raise ValueError(f"Rollout lacks static input columns: {required_static}")
        if static["sequenceId"].n_unique() != len(static) or not set(
            table.ranges
        ) <= set(static["sequenceId"]):
            raise ValueError("Static rollout inputs require one row per sequence")
        static = with_domain_record(
            static.select("sequenceId", *required_static), static_columns, DOMAIN_STATIC
        )
        static, _ = _encode_recorded_features(static, static_columns, metadata)
    with IndexedSequenceWriter(destination) as writer:
        snapshot = writer.root / "original-metadata.json"
        snapshot.write_bytes(original)
        if static is not None:
            writer.write_static(static)
        for sequence in table.ranges:
            rows = with_domain_record(
                table.load(sequence), columns, DOMAIN_OBSERVATIONS
            )
            encoded, fitted = _encode_recorded_features(rows, columns, metadata)
            actions = set(
                (sequence_provenance or {}).get("action_columns", {}).values()
            )
            if actions:
                from sequifier.special_tokens import SPECIAL_TOKEN_IDS

                encoded = encoded.with_columns(
                    pl.col(name).fill_null(
                        SPECIAL_TOKEN_IDS.mask if name in fitted.id_maps else 0
                    )
                    for name in actions.intersection(columns)
                )
            writer.add(
                _mask_features(
                    encoded,
                    [
                        name
                        for name in columns
                        if name not in actions and name != INCOMING_ACTION_PRESENT
                    ],
                    fitted,
                    layouts,
                    masked=pl.col(mask_column) if mask_column else pl.lit(False),
                ),
                {},
            )
        prefix = metadata.preconditioning_image
        writer.publish(
            {
                **dict(metadata),
                "split_paths": ["predict.split.json"],
                "split_names": ["predict"],
                "indexed_views": ["predict"],
                "prediction_aligned_splits": [],
                "fit_split_indices": [],
                "window_lengths": None,
                "split_index": None,
                "split_ratios": None,
                "split_column": None,
                "split_values": None,
                "indexed_curriculum_columns": [],
                "indexed_passthrough": sorted(passthrough),
                "indexed_derived_features": [],
                "depth_layouts": layouts,
                "static_features_path": "static.parquet"
                if static is not None
                else None,
                "indexed_inline_media": inline_media,
                "raw_mask_column": mask_column,
                "image_sources": {
                    name: config.compatibility_signature()
                    for name, config in metadata.image_sources.items()
                },
                "preconditioning_image": prefix.model_copy(
                    update={
                        "paths": None,
                        "view_names": sorted(prefix.views),
                        "signature": None,
                    }
                )
                if prefix
                else None,
            },
            feature_columns=columns,
            original_metadata=snapshot,
            provenance=sequence_provenance,
        )


def resolve_inference_view(config, *, skip_metadata=False):
    path = Path(config.project_root) / config.data_path if config.data_path else None
    view = IndexedView.load(path) if is_indexed_view(path) else None
    if config.input_layout == "raw":
        if view is not None or path is None:
            raise ValueError("Raw inference requires a fresh raw data_path")
        if skip_metadata or not config.metadata_config_path:
            raise ValueError("Raw inference requires the original metadata_config_path")
    if config.input_layout == "indexed" and path is not None and view is None:
        raise ValueError("Indexed inference requires an indexed split manifest")
    if view is not None:
        if skip_metadata:
            raise ValueError(
                "Indexed inference requires original preprocessing metadata"
            )
        view.verify()
    return view


def original_metadata_path(config, view):
    if view is not None:
        return (
            view.original_metadata_path
            if view.view["view"] == "predict"
            else view.metadata_path
        )
    if config.metadata_config_path:
        return Path(config.project_root) / config.metadata_config_path
    return None


@contextmanager
def fresh_view_from_raw(
    config, *, required_columns=(), passthrough_columns=(), optional_passthrough=()
):
    root = Path(config.project_root).resolve()
    raw_path = root / config.data_path
    with TemporaryDirectory(prefix="sequifier-inference-") as temporary:
        yield _prepare_inference(
            config,
            root=root,
            raw_path=raw_path,
            destination=Path(temporary) / "dataset",
            read_format=config.read_format
            if raw_path.is_dir() or config.read_format == "sequence"
            else None,
            required_columns=required_columns,
            passthrough_columns=passthrough_columns,
            optional_passthrough=optional_passthrough,
        )


def export_policy_metadata(config):
    """Export the frozen contract of a policy trained before explicit preparation."""
    from sequifier.config.inference_common import ResolvedInferenceModel

    root = Path(config.run.project_root)
    path = root / config.model_path
    model = ResolvedInferenceModel.load(
        str(path), interface_name=config.model_interface, task="policy"
    )
    execution, name = model.execution, model.interface_name
    assert execution is not None and name is not None
    interface = execution.interface(name)
    if interface.policy is None:
        raise ValueError("Metadata export requires a policy-bound model")
    storage = execution.interface_storage.get(name)
    metadata = interface.preprocessing_metadata(storage)
    if any(
        not metadata.cardinality_config.get(column, {}).get("hashing")
        and any(is_hash_bucket_label(label) for label in mapping)
        for column, mapping in metadata.id_maps.items()
    ):
        raise ValueError(
            "Original preprocessing metadata is required for this actor's categorical hashing"
        )
    values = metadata.model_dump(mode="json")
    values["split_names"] = []
    values["split_paths"] = []
    values["prediction_aligned_splits"] = []
    values["split_index"] = None
    values["window_lengths"] = None
    output = root / config.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2)
