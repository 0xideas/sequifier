"""Central versioned PT tensor payload and validation at host boundaries."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from sequifier.config.depth_layout import DepthLayoutRegistryModel


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
        expected = (*shape, layout.context_length) if layout else shape
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
            if (
                values.is_floating_point()
                or (values.to(torch.int64) < 0).any().item()
                or (values.to(torch.int64) >= n_classes[column]).any().item()
            ):
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
            layout.context_length,
        ):
            raise ValueError(
                f"{name}: depth mask must be bool [batch, time, {layout.context_length}]"
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
    depth_valid_masks: dict[str, Tensor] = field(default_factory=dict)
    sample_positions: Tensor | None = None

    def __iter__(self):
        # Preserve the historical unpacking contract for coordinate-only readers.
        return iter(
            (
                self.sequences,
                self.sequence_ids,
                self.subsequence_ids,
                self.start_item_positions,
                self.left_pad_lengths,
            )
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
        for key in (
            "sequence_ids",
            "subsequence_ids",
            "start_item_positions",
            "left_pad_lengths",
        ):
            value = getattr(self, key)
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.int64
                or tuple(value.shape) != (n,)
            ):
                raise ValueError(f"{key} must be int64 [{n}]")
        if self.sample_positions is not None and (
            self.sample_positions.dtype != torch.int64
            or tuple(self.sample_positions.shape) != (n,)
        ):
            raise ValueError(f"sample_positions must be int64 [{n}]")
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


def load_pt_payload(path, *, layouts=None, n_classes=None) -> StoredTensorBatch:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, (tuple, list)) and len(value) == 5:
        batch = StoredTensorBatch(*value)
    elif isinstance(value, dict):
        version = value.get("version")
        if value.get("format") != "sequifier_tensor_batch" or version not in {2, 3}:
            raise ValueError(f"Unsupported PT tensor payload format/version in {path}")
        expected = {
            "format",
            "version",
            "sequences",
            "metadata",
            "sequence_ids",
            "subsequence_ids",
            "start_item_positions",
            "left_pad_lengths",
        }
        expected_metadata = (
            {"depth_valid_masks"}
            if version == 2
            else {"depth_valid_masks", "sample_positions"}
        )
        if set(value) != expected or set(value["metadata"]) != expected_metadata:
            raise ValueError(f"Inconsistent version {version} tensor payload schema")
        batch = StoredTensorBatch(
            **{key: value[key] for key in expected - {"format", "version", "metadata"}},
            depth_valid_masks=value["metadata"]["depth_valid_masks"],
            sample_positions=value["metadata"].get("sample_positions"),
        )
    else:
        raise ValueError(f"Unsupported PT tensor payload in {path}")
    return batch.validate(layouts, n_classes=n_classes)


def save_pt_payload(batch: StoredTensorBatch, path, *, layouts=None, n_classes=None):
    batch.validate(layouts, n_classes=n_classes)
    if not batch.depth_valid_masks and batch.sample_positions is None:
        torch.save(tuple(batch), path)
    else:
        torch.save(
            {
                "format": "sequifier_tensor_batch",
                "version": 3 if batch.sample_positions is not None else 2,
                "sequences": batch.sequences,
                "metadata": {
                    "depth_valid_masks": batch.depth_valid_masks,
                    **(
                        {"sample_positions": batch.sample_positions}
                        if batch.sample_positions is not None
                        else {}
                    ),
                },
                **{
                    key: getattr(batch, key)
                    for key in (
                        "sequence_ids",
                        "subsequence_ids",
                        "start_item_positions",
                        "left_pad_lengths",
                    )
                },
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
            raise ValueError("Cannot concatenate mixed samplePosition payloads")
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
        **{
            key: torch.cat([getattr(b, key) for b in batches])
            for key in (
                "sequence_ids",
                "subsequence_ids",
                "start_item_positions",
                "left_pad_lengths",
            )
        },
    )
