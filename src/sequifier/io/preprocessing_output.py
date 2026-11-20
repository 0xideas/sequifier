"""Preprocessing destinations for materialized windows and indexed sequences."""

import json
import shutil
from functools import cached_property
from pathlib import Path
from typing import Literal

import polars as pl

from sequifier.io.pt_payload import load_pt_payload
from sequifier.preprocess import (
    SPLIT_VALUE_COLUMN,
    _batch_to_arrays,
    _folder_input_files,
    _place_sequence_windows_for_splits,
    _process_batches_single_file,
    combine_multiprocessing_outputs,
    create_file_paths_for_single_file,
    create_split_file_path,
    delete_files,
)
from sequifier.trajectory.sources import file_checksum


def _add_static_metadata(metadata, path, columns, fields):
    metadata.update(static_features_path=str(path), static_columns=columns)
    for name, additions in fields.items():
        metadata[name].update(additions)


class WindowOutput:
    indexed: Literal[False] = False

    def __init__(self, run):
        self.run = run
        self.fragments = {}
        self.metadata_path = (
            Path(run.project_root)
            / "configs"
            / "metadata_configs"
            / f"{run.data_name_root}.json"
        )
        self.time_index_name = f"data/{run.data_name_root}-image-time-index.parquet"
        self.video_index_dir = f"data/{run.data_name_root}-video-indexes"

    def time_index_path(self):
        return Path(self.run.project_root) / self.time_index_name

    def bind_media(self, manifest):
        """Materialized media references already use project-relative paths."""

    def training_sequence_ids(self):
        run = self.run
        path = Path(run.split_paths[0])
        folder = path.with_suffix("")
        files = (
            sorted(folder.glob(f"*.{run.write_format}")) if folder.is_dir() else [path]
        )
        ids: set[int] = set()
        for file in files:
            if run.write_format == "pt":
                payload = load_pt_payload(file, layouts=run.depth_layouts)
                ids.update(payload.sequence_ids.tolist())
            else:
                scan = (
                    pl.scan_parquet(file)
                    if run.write_format == "parquet"
                    else pl.scan_csv(file)
                )
                ids.update(
                    scan.select(pl.col("sequenceId").cast(pl.Int64).unique())
                    .collect()
                    .get_column("sequenceId")
                    .to_list()
                )
        return ids

    def write_metadata(self, metadata, columns):
        self.feature_columns = columns
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata))

    @cached_property
    def schema(self):
        return self.run._create_schema(
            self.run.output_column_data_types, self.run.storage_layout.window_length
        )

    def write_sequences(self, data, assignments, *, bucket=None, pool=None):
        run = self.run
        fragment = bucket is not None
        name = (
            f"{run.data_name_root}-fragment-{bucket}"
            if fragment
            else run.data_name_root
        )
        paths = (
            [
                str(
                    Path(path).with_name(
                        Path(path).name.replace(run.data_name_root, name, 1)
                    )
                )
                for path in run.split_paths
            ]
            if fragment
            else run.split_paths
        )
        types = run.output_column_data_types
        n_batches = _process_batches_single_file(
            run.project_root,
            name,
            data,
            self.schema,
            run.num_workers,
            run.split_layouts,
            run.window_stride,
            self.feature_columns,
            types,
            run.split_ratios,
            run.write_format,
            paths,
            run.target_dir,
            run.batches_per_file,
            run.merge_output,
            run.allow_sequence_splitting,
            run.split_method,
            run.seed,
            assignments,
            pool,
            normalize_on_all_data=run.normalize_on_all_data,
            alignment=run.alignment,
            split_values=run.split_values,
        )
        if run.merge_output:
            files = create_file_paths_for_single_file(
                run.project_root,
                run.target_dir,
                run.n_splits,
                n_batches,
                name,
                run.write_format,
            )
            combine_multiprocessing_outputs(
                run.project_root,
                run.target_dir,
                run.n_splits,
                files,
                name,
                run.write_format,
                in_target_dir=fragment,
            )
            delete_files(files)
            if fragment:
                for split in range(run.n_splits):
                    path = create_split_file_path(
                        run.project_root,
                        name,
                        split,
                        run.write_format,
                        True,
                        run.target_dir,
                        None,
                        None,
                    )
                    self.fragments.setdefault(split, []).append(path)

    def write_static(self, static, fields):
        run = self.run
        path = Path("data") / f"{run.data_name_root}-static.parquet"
        static.write_parquet(Path(run.project_root) / path)
        files = [self.metadata_path]
        if not run.merge_output:
            files.extend(
                Path(split).with_suffix("") / "metadata.json"
                for split in run.split_paths
            )
        columns = [name for name in static.columns if name != "sequenceId"]
        for file in files:
            metadata = json.loads(file.read_text())
            _add_static_metadata(metadata, path, columns, fields)
            file.write_text(json.dumps(metadata, indent=4))

    def finish(self):
        run = self.run
        if self.fragments:
            combine_multiprocessing_outputs(
                run.project_root,
                run.target_dir,
                run.n_splits,
                self.fragments,
                run.data_name_root,
                run.write_format,
                in_target_dir=False,
            )
            delete_files(self.fragments)
        run._cleanup(run.write_format)
        if run._static_source is not None:
            run._finalize_static_source()
            shutil.rmtree(Path(run.project_root) / "data" / run.target_dir)


class IndexedOutput:
    indexed: Literal[True] = True
    time_index_name = "item-times.parquet"

    def __init__(self, run, writer, resume):
        if resume:
            raise ValueError("Indexed preparation can be restarted from raw input")
        self.run = run
        self.writer = writer
        self.metadata = {}
        self.video_index_dir = str(writer.root / "video-indexes")
        run.target_dir = str(writer.root / "scratch")

    def training_sequence_ids(self):
        return self.writer.split_sequence_ids(0)

    def time_index_path(self):
        path = self.writer.root / self.time_index_name
        self.writer.files.append(path)
        return path

    def bind_media(self, manifest):
        run = self.run
        for name, source in manifest["image_sources"].items():
            source["path"] = str(Path(run.project_root) / run.image_sources[name].path)
            paths = source.get("video_index_paths", {})
            for sequence, path in paths.items():
                index_path = Path(path)
                self.writer.files.append(index_path)
                paths[sequence] = str(index_path.relative_to(self.writer.root))
        if prefix := manifest.get("preconditioning_image"):
            assert run.preconditioning_image is not None
            prefix["paths"] = {
                name: str(Path(run.project_root) / path)
                for name, path in run.preconditioning_image.paths.items()
            }

    def write_static(self, static, fields):
        self.writer.write_static(static)
        columns = [name for name in static.columns if name != "sequenceId"]
        _add_static_metadata(self.metadata, "static.parquet", columns, fields)

    def write_metadata(self, metadata, columns):
        run = self.run
        self.feature_columns = columns
        names = (
            ["predict"]
            if run.indexed_prediction
            else [str(index) for index in range(run.n_splits)]
        )
        metadata.update(
            split_paths=[f"{name}.split.json" for name in names],
            indexed_views=names,
            indexed_passthrough=run.indexed_passthrough,
            fit_split_indices=(
                []
                if run.metadata_config_path
                else (list(range(run.n_splits)) if run.normalize_on_all_data else [0])
            ),
        )
        self.metadata.update(metadata)

    def write_sequences(self, data, assignments, *, bucket=None, pool=None):
        run = self.run
        arrays = _batch_to_arrays(data, [])
        self.metadata["indexed_curriculum_columns"] = list(arrays.curriculum_values)
        for start, stop in zip(arrays.run_starts.tolist(), arrays.run_stops.tolist()):
            sequence = int(arrays.sequence_ids[start])
            views = (
                {}
                if run.indexed_prediction
                else _place_sequence_windows_for_splits(
                    arrays,
                    start,
                    stop,
                    sequence,
                    run.split_layouts,
                    run.window_stride,
                    run.split_ratios,
                    run.split_method,
                    run.seed,
                    assignments,
                    alignment=run.alignment,
                    split_values=run.split_values,
                )
            )
            self.writer.add(
                data.slice(start, stop - start).drop(SPLIT_VALUE_COLUMN, strict=False),
                views,
            )

    def finish(self):
        run = self.run

        if run._static_source is not None:
            run._finalize_static_source()
        shutil.rmtree(Path(run.project_root) / "data" / run.target_dir)
        original = (
            Path(run.project_root) / run.metadata_config_path
            if run.metadata_config_path
            else None
        )
        input_path = Path(run.preprocessing_data_path)
        paths = (
            [input_path]
            if input_path.is_file()
            else [
                Path(path)
                for path in _folder_input_files(
                    str(input_path), run.read_format, run.static_data_path
                )
            ]
        )
        if run.static_data_path:
            paths.append(Path(run.static_data_path))
        self.writer.publish(
            self.metadata,
            feature_columns=self.feature_columns,
            original_metadata=original,
            provenance={
                "source": {
                    "preprocessing_data_path": str(input_path.absolute()),
                    "static_data_path": str(Path(run.static_data_path).absolute())
                    if run.static_data_path
                    else None,
                    "indexed_passthrough": run.indexed_passthrough,
                },
                "files": [
                    {"path": str(path.absolute()), "sha256": file_checksum(path)}
                    for path in paths
                ],
            },
        )
