"""Apply an existing preprocessing contract to fresh raw data."""

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from sequifier.config.image_config import bind_image_paths
from sequifier.config.metadata import load_dataset_metadata
from sequifier.config.preprocess_config import PreprocessorModel
from sequifier.io.categorical import is_hash_bucket_label
from sequifier.io.indexed_sequences import (
    IndexedSequenceWriter,
    IndexedView,
    feature_contract_id,
    is_indexed_view,
)


def prepare_inference(
    config, *, required_columns=(), optional_passthrough=(), sequence_id_staging=None
):
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
        sequence_id_staging=sequence_id_staging,
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
    sequence_id_staging,
    num_workers=1,
):
    from sequifier.preprocess import Preprocessor, _folder_input_files

    metadata_path = root / config.metadata_config_path
    original_contents = metadata_path.read_bytes()
    metadata = load_dataset_metadata(str(metadata_path))
    if metadata_path.read_bytes() != original_contents:
        raise ValueError(
            "The original preprocessing metadata changed during preparation"
        )
    read_format = read_format or raw_path.suffix.lower().lstrip(".")
    if read_format not in {"csv", "parquet"}:
        raise ValueError("Fresh raw folders require read_format: csv or parquet")
    static_path = (
        str(root / config.static_data_path) if config.static_data_path else None
    )
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
    required_columns = set(required_columns)
    missing = required_columns - available
    missing |= {"sequenceId", "itemPosition"} - available
    if missing:
        raise ValueError(f"Fresh data is missing model columns: {sorted(missing)}")
    passthrough = set(passthrough_columns)
    passthrough.update((set(optional_passthrough) & available) - required_columns)
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
    image_sources, prefix = bind_image_paths(
        metadata.image_sources,
        metadata.preconditioning_image,
        config.image_paths,
        config.preconditioning_image_paths,
    )
    persisted_mask = metadata.raw_mask_column
    if persisted_mask is not None and config.mask_column not in (None, persisted_mask):
        raise ValueError("mask_column differs from the original preprocessing metadata")
    mask_column = config.mask_column or persisted_mask
    if mask_column and mask_column not in available:
        raise ValueError(
            f"Fresh data is missing the original mask column {mask_column!r}"
        )
    sequence_labels = None
    id_types = {schema["sequenceId"] for schema in schemas}
    if sequence_id_staging and pl.String in id_types:
        if any(not dtype.is_integer() for dtype in id_types - {pl.String}):
            raise ValueError(
                "Fresh policy data requires string or integer sequenceId values"
            )
        raw_path, static_path, sequence_labels = _stage_sequence_ids(
            sequence_id_staging,
            files,
            read_format,
            static_path,
            bool(image_sources or prefix),
        )
        read_format = "parquet"
    depth = bool(metadata.depth_layouts)
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
                "indexed_passthrough": sorted(passthrough),
            }
        )
        Preprocessor(options).run(indexed_writer=writer, indexed_prediction=True)
    view = IndexedView.load(destination / "predict.split.json")
    view.sequence_labels = sequence_labels
    return view


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
        if config.metadata_config_path and view.view["view"] == "predict":
            explicit = Path(config.project_root) / config.metadata_config_path
            if feature_contract_id(explicit) != feature_contract_id(
                view.original_metadata_path
            ):
                raise ValueError(
                    "metadata_config_path differs from the original metadata used to prepare this input"
                )
    return view


def original_metadata_path(config, view):
    if config.metadata_config_path:
        return Path(config.project_root) / config.metadata_config_path
    if view is None:
        return None
    return (
        view.original_metadata_path
        if view.view["view"] == "predict"
        else view.metadata_path
    )


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
            read_format=config.read_format if raw_path.is_dir() else None,
            required_columns=required_columns,
            passthrough_columns=passthrough_columns,
            optional_passthrough=optional_passthrough,
            sequence_id_staging=Path(temporary) / "raw"
            if config.task == "policy"
            else None,
        )


def _stage_sequence_ids(directory, files, read_format, static_path, has_media):
    """Give the preprocessor integer coordinates and retain the domain labels."""
    key = "sequenceId"

    def scan_labels(path, file_format):
        if file_format == "csv":
            return pl.scan_csv(path, schema_overrides={key: pl.String})
        return pl.scan_parquet(path).with_columns(pl.col(key).cast(pl.String))

    sources = [scan_labels(file, read_format) for file in files]
    labels = dict.fromkeys(
        label
        for source in sources
        for label in source.select(pl.col(key).unique(maintain_order=True)).collect()[
            key
        ]
    )
    if None in labels:
        raise ValueError("Fresh policy sequenceId values must be non-null")
    ids = range(len(labels))
    if has_media:
        # External media filenames use integer sequence IDs.
        ids = pl.Series(list(labels)).cast(pl.Int64, strict=False)
        if ids.null_count() or ids.n_unique() != len(ids):
            raise ValueError("External media requires distinct integer sequence IDs")
    mapping = dict(zip(labels, ids, strict=True))
    encode = pl.col(key).replace_strict(mapping, default=None, return_dtype=pl.Int64)

    directory.mkdir()
    for index, source in enumerate(sources):
        source.with_columns(encode).sink_parquet(directory / f"{index:08d}.parquet")
    if static_path:
        static_format = "parquet" if static_path.lower().endswith(".parquet") else "csv"
        static = scan_labels(static_path, static_format)
        static_path = str(directory.parent / "static.parquet")
        static.with_columns(encode).sink_parquet(static_path)
    raw_path = directory / "00000000.parquet" if len(files) == 1 else directory
    return raw_path, static_path, {code: label for label, code in mapping.items()}


def export_policy_metadata(config):
    """Export the frozen contract of a policy trained before explicit preparation."""
    from sequifier.artifacts.model_artifact import read_execution_artifact
    from sequifier.config.inference_common import resolve_inference_model

    root = Path(config.run.project_root)
    path = root / config.model_path
    artifact = read_execution_artifact(str(path))
    execution, name = resolve_inference_model(
        artifact.model_config.values,
        interface_name=config.model_interface,
        task="policy",
    )
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
