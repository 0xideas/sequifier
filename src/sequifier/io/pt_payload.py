"""Central versioned PT tensor payload and validation at host boundaries."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from sequifier.config.depth_layout import DepthLayoutRegistryModel
from sequifier.contract_versions import ContractVersions, require_contract_version


def valid_categorical_ids(values: Tensor, n_classes: int) -> bool:
    if values.is_floating_point() or values.is_complex() or values.dtype == torch.bool:
        return False
    ids = values.to(torch.int64)
    return not (ids < 0).any().item() and not (ids >= n_classes).any().item()


def validate_tensor_inputs(
    sequences: dict[str, Tensor],
    masks: dict[str, Tensor],
    layouts: DepthLayoutRegistryModel,
    *,
    n_classes=None,
    attention_valid_mask: Tensor | None = None,
):
    if not sequences:
        raise ValueError("Tensor inputs must contain features")
    shape = next(iter(sequences.values())).shape[:2]
    if len(shape) != 2:
        raise ValueError("Features require batch and temporal axes")
    for column, values in sequences.items():
        layout = layouts.layout_for_column(column)
        expected = (*shape, layout.max_depth_length) if layout else shape
        if tuple(values.shape) != tuple(expected):
            raise ValueError(
                f"{column}: expected {tuple(expected)}, got {tuple(values.shape)}"
            )
        if values.dtype == torch.bool or values.is_complex():
            raise ValueError(f"{column}: unsupported feature dtype {values.dtype}")
        if values.is_floating_point() and not torch.isfinite(values).all().item():
            raise ValueError(
                f"{column}: all values, including masked slots, must be finite"
            )
        if n_classes and column in n_classes:
            if not valid_categorical_ids(values, n_classes[column]):
                raise ValueError(
                    f"{column}: categorical IDs must be integers in [0, {n_classes[column]})"
                )
    if set(masks) != set(layouts.root):
        raise ValueError(
            f"Depth masks must match layouts: expected {sorted(layouts.root)}, got {sorted(masks)}"
        )
    if attention_valid_mask is not None:
        if attention_valid_mask.dtype != torch.bool or tuple(
            attention_valid_mask.shape
        ) != tuple(shape):
            raise ValueError("Outer validity mask must be bool [batch, time]")
    for name, layout in layouts.items():
        mask = masks[name]
        if mask.dtype != torch.bool or tuple(mask.shape) != (
            *shape,
            layout.max_depth_length,
        ):
            raise ValueError(
                f"{name}: depth mask must be bool [batch, time, {layout.max_depth_length}]"
            )
        if not layout.allow_gaps and (mask[..., 1:] & ~mask[..., :-1]).any().item():
            raise ValueError(
                f"{name}: allow_gaps=false requires a true prefix followed by a false tail"
            )
        if (
            attention_valid_mask is not None
            and (mask & ~attention_valid_mask[..., None]).any().item()
        ):
            raise ValueError(
                f"{name}: temporally padded slots must have false depth masks"
            )


@dataclass
class StoredTensorBatch:
    sequences: dict[str, Tensor]
    sequence_ids: Tensor
    subsequence_ids: Tensor
    start_item_positions: Tensor
    left_pad_lengths: Tensor
    split_start_item_positions: Tensor
    split_end_item_positions: Tensor
    depth_valid_masks: dict[str, Tensor] = field(default_factory=dict)
    sample_positions: Tensor | None = None
    curriculum_columns: tuple[str, ...] = ()

    def slice_rows(self, start: int, stop: int) -> "StoredTensorBatch":
        # Own just the selected storage so saving a shard cannot serialize its parent.
        rows = slice(start, stop)
        return StoredTensorBatch(
            sequences={
                name: value[rows].clone() for name, value in self.sequences.items()
            },
            depth_valid_masks={
                name: value[rows].clone()
                for name, value in self.depth_valid_masks.items()
            },
            sample_positions=(
                self.sample_positions[rows].clone()
                if self.sample_positions is not None
                else None
            ),
            curriculum_columns=self.curriculum_columns,
            sequence_ids=self.sequence_ids[rows].clone(),
            subsequence_ids=self.subsequence_ids[rows].clone(),
            start_item_positions=self.start_item_positions[rows].clone(),
            left_pad_lengths=self.left_pad_lengths[rows].clone(),
            split_start_item_positions=self.split_start_item_positions[rows].clone(),
            split_end_item_positions=self.split_end_item_positions[rows].clone(),
        )

    def validate(self, layouts=None, *, n_classes=None):
        layouts = DepthLayoutRegistryModel.model_validate(layouts or {})
        missing = set(layouts.deep_columns) - set(self.sequences)
        if missing:
            raise ValueError(
                f"Stored payload is missing layout features: {sorted(missing)}"
            )
        if not self.sequences:
            raise ValueError("Stored tensor batch has no features")
        first = next(iter(self.sequences.values()))
        if first.ndim < 2:
            raise ValueError("Stored tensors require batch and temporal axes")
        n, width = first.shape[:2]
        for key, value in (
            ("sequence_ids", self.sequence_ids),
            ("subsequence_ids", self.subsequence_ids),
            ("start_item_positions", self.start_item_positions),
            ("left_pad_lengths", self.left_pad_lengths),
            ("split_start_item_positions", self.split_start_item_positions),
            ("split_end_item_positions", self.split_end_item_positions),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.int64
                or tuple(value.shape) != (n,)
            ):
                raise ValueError(f"{key} must be int64 [{n}]")
        if (
            (self.split_start_item_positions >= self.split_end_item_positions)
            .any()
            .item()
        ):
            raise ValueError("Split target bounds must be non-empty")
        if self.sample_positions is not None and (
            self.sample_positions.dtype != torch.int64
            or self.sample_positions.ndim not in {1, 2}
            or self.sample_positions.shape[0] != n
            or (self.sample_positions.ndim == 2 and self.sample_positions.shape[1] == 0)
        ):
            raise ValueError(
                f"sample_positions must be int64 [{n}] or [{n}, n_columns]"
            )
        if self.sample_positions is None and self.curriculum_columns:
            raise ValueError("curriculum_columns require sample_positions")
        if self.curriculum_columns and (
            any(
                not isinstance(column, str) or not column
                for column in self.curriculum_columns
            )
            or len(set(self.curriculum_columns)) != len(self.curriculum_columns)
        ):
            raise ValueError("curriculum_columns must be unique, non-empty strings")
        if self.sample_positions is not None and self.curriculum_columns:
            curriculum_width = (
                1 if self.sample_positions.ndim == 1 else self.sample_positions.shape[1]
            )
            if len(self.curriculum_columns) != curriculum_width:
                raise ValueError(
                    "curriculum_columns must match the sample_positions width"
                )
        if ((self.left_pad_lengths < 0) | (self.left_pad_lengths > width)).any().item():
            raise ValueError("Invalid stored left padding lengths")
        outer = (
            torch.arange(width, device=self.left_pad_lengths.device)[None, :]
            >= self.left_pad_lengths[:, None]
        )
        validate_tensor_inputs(
            self.sequences,
            self.depth_valid_masks,
            layouts,
            n_classes=n_classes,
            attention_valid_mask=outer,
        )
        return self


def tabular_tensor_batch(
    data, column_data_types, columns, window_length, *, curriculum=(), sort_rows=True
) -> StoredTensorBatch:
    """Decode features, coordinates and curriculum values in the same window order."""
    import polars as pl

    from sequifier.helpers import columns_from_slice
    from sequifier.io.sample_order import (
        SAMPLE_KEY_COLUMNS,
        curriculum_storage_column,
        curriculum_storage_columns,
        source_curriculum_column,
    )

    keys = SAMPLE_KEY_COLUMNS
    bounds = (
        "startItemPosition",
        "leftPadLength",
        "splitStartItemPosition",
        "splitEndItemPosition",
    )
    if not set(bounds) <= set(data.columns):
        raise ValueError(
            "Prepared windows require padding, start positions and split bounds"
        )
    if curriculum is None:
        curriculum = [
            source_curriculum_column(col)
            for col in curriculum_storage_columns(data.columns)
        ]
    stored = [curriculum_storage_column(name) for name in curriculum]
    if any(not data.schema[key].is_integer() or data[key].null_count() for key in keys):
        raise ValueError("Stored sequence and subsequence IDs must be integers")
    values = []
    for name, alias in zip(curriculum, stored):
        source = alias if alias in data.columns else name
        if source not in data.columns:
            raise ValueError(f"Missing curriculum column {name!r} in stored windows")
        column = pl.col(source)
        values.append(pl.when(column.n_unique() == 1).then(column.first()).alias(alias))
    positions = data.group_by(keys, maintain_order=True).agg(
        pl.col(*bounds).first(), *values
    )
    if sort_rows:
        positions = positions.sort(keys)
    samples = None
    if stored:
        selected = positions.select(stored)
        if any(selected.null_count().row(0)):
            raise ValueError(
                "Curriculum values must be non-null and identical per subsequence"
            )
        samples = torch.tensor(selected.to_numpy(), dtype=torch.int64)

    identities = positions.select(keys)
    steps = columns_from_slice(slice(0, window_length), window_length)
    partitions = data.partition_by("inputCol", as_dict=True) if columns else {}
    sequences = {}
    for name in columns:
        rows = partitions.get((name,))
        if rows is None:
            raise ValueError(f"Column not found in preprocessed data: {name}")
        if not rows.select(keys).equals(identities):
            rows = identities.join(rows, on=keys, validate="1:1", maintain_order="left")
            if rows.height != identities.height:
                raise ValueError(f"{name}: expected one feature row per stored window")
        sequences[name] = torch.tensor(
            rows.select(steps).to_numpy(), dtype=column_data_types[name]
        )

    def coordinate(name):
        return torch.tensor(
            positions[name].cast(pl.Int64).to_numpy(), dtype=torch.int64
        )

    return StoredTensorBatch(
        sequences=sequences,
        sequence_ids=coordinate("sequenceId"),
        subsequence_ids=coordinate("subsequenceId"),
        start_item_positions=coordinate("startItemPosition"),
        left_pad_lengths=coordinate("leftPadLength"),
        split_start_item_positions=coordinate("splitStartItemPosition"),
        split_end_item_positions=coordinate("splitEndItemPosition"),
        sample_positions=samples,
        curriculum_columns=tuple(curriculum),
    )


def load_pt_payload(path, *, layouts=None, n_classes=None) -> StoredTensorBatch:
    value = torch.load(path, map_location="cpu", weights_only=True)
    version = value.get("version") if isinstance(value, dict) else None
    require_contract_version(
        "PT tensor-batch envelope", version, ContractVersions.PT_TENSOR_BATCH
    )
    if value.get("format") != "sequifier_tensor_batch":
        raise ValueError(f"Unsupported PT tensor payload format in {path}")
    expected = {
        "format",
        "version",
        "sequences",
        "metadata",
        "sequence_ids",
        "subsequence_ids",
        "start_item_positions",
        "left_pad_lengths",
        "split_start_item_positions",
        "split_end_item_positions",
    }
    expected_metadata = {"depth_valid_masks", "sample_positions", "curriculum_columns"}
    metadata = value.get("metadata")
    if (
        set(value) != expected
        or not isinstance(metadata, dict)
        or set(metadata) != expected_metadata
    ):
        raise ValueError("Inconsistent tensor payload schema")
    batch = StoredTensorBatch(
        **{key: value[key] for key in expected - {"format", "version", "metadata"}},
        depth_valid_masks=metadata["depth_valid_masks"],
        sample_positions=metadata["sample_positions"],
        curriculum_columns=tuple(metadata["curriculum_columns"]),
    )
    return batch.validate(layouts, n_classes=n_classes)


def save_pt_payload(batch: StoredTensorBatch, path, *, layouts=None, n_classes=None):
    batch.validate(layouts, n_classes=n_classes)
    torch.save(
        {
            "format": "sequifier_tensor_batch",
            "version": ContractVersions.PT_TENSOR_BATCH,
            "sequences": batch.sequences,
            "metadata": {
                "depth_valid_masks": batch.depth_valid_masks,
                "sample_positions": batch.sample_positions,
                "curriculum_columns": list(batch.curriculum_columns),
            },
            "sequence_ids": batch.sequence_ids,
            "subsequence_ids": batch.subsequence_ids,
            "start_item_positions": batch.start_item_positions,
            "left_pad_lengths": batch.left_pad_lengths,
            "split_start_item_positions": batch.split_start_item_positions,
            "split_end_item_positions": batch.split_end_item_positions,
        },
        path,
    )


def concatenate_pt_batches(batches: list[StoredTensorBatch]) -> StoredTensorBatch:
    if not batches:
        raise ValueError("Cannot concatenate an empty batch collection")
    first = batches[0]
    for batch in batches:
        if set(batch.sequences) != set(first.sequences) or set(
            batch.depth_valid_masks
        ) != set(first.depth_valid_masks):
            raise ValueError("Cannot concatenate different tensor payload schemas")
        if (batch.sample_positions is None) != (first.sample_positions is None):
            raise ValueError("Cannot concatenate mixed curriculum payloads")
        if batch.curriculum_columns != first.curriculum_columns:
            raise ValueError("Cannot concatenate different curriculum columns")
    return StoredTensorBatch(
        sequences={
            key: torch.cat([b.sequences[key] for b in batches])
            for key in first.sequences
        },
        depth_valid_masks={
            key: torch.cat([b.depth_valid_masks[key] for b in batches])
            for key in first.depth_valid_masks
        },
        sample_positions=(
            torch.cat([b.sample_positions for b in batches])
            if first.sample_positions is not None
            else None
        ),
        curriculum_columns=first.curriculum_columns,
        split_start_item_positions=torch.cat(
            [b.split_start_item_positions for b in batches]
        ),
        split_end_item_positions=torch.cat(
            [b.split_end_item_positions for b in batches]
        ),
        sequence_ids=torch.cat([batch.sequence_ids for batch in batches]),
        subsequence_ids=torch.cat([batch.subsequence_ids for batch in batches]),
        start_item_positions=torch.cat(
            [batch.start_item_positions for batch in batches]
        ),
        left_pad_lengths=torch.cat([batch.left_pad_lengths for batch in batches]),
    )
