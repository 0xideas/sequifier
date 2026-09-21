"""Deterministic sample ordering for ordinary and curriculum datasets."""

from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from sequifier.helpers import WindowSampleIndex

# Private storage names; user-facing curriculum columns are explicitly configured.
CURRICULUM_COLUMN_PREFIX = "__sequifier_curriculum_value__"
SAMPLE_POSITION_COLUMN = f"{CURRICULUM_COLUMN_PREFIX}samplePosition"
LEGACY_SAMPLE_POSITION_COLUMN = "samplePosition"
SAMPLE_KEY_COLUMNS = ["sequenceId", "subsequenceId"]


def curriculum_columns(value: object) -> tuple[str, ...]:
    """Normalize a configured curriculum key to ordered column names."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(
        isinstance(column, str) for column in value
    ):
        return tuple(value)
    return ()


def curriculum_storage_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return private curriculum columns in stored schema order."""
    return [column for column in columns if column.startswith(CURRICULUM_COLUMN_PREFIX)]


def curriculum_storage_column(column: str) -> str:
    return f"{CURRICULUM_COLUMN_PREFIX}{column}"


def source_curriculum_column(column: str) -> str:
    return column.removeprefix(CURRICULUM_COLUMN_PREFIX)


def sample_positions_from_parquet(data: pl.DataFrame, column: str) -> Tensor | None:
    """Return one validated curriculum value per stored subsequence."""
    candidates = (column, curriculum_storage_column(column))
    stored_column = next((name for name in candidates if name in data.columns), None)
    if stored_column is None and column == LEGACY_SAMPLE_POSITION_COLUMN:
        stored_column = (
            LEGACY_SAMPLE_POSITION_COLUMN
            if LEGACY_SAMPLE_POSITION_COLUMN in data.columns
            else None
        )
    if stored_column is None:
        return None
    positions = (
        data.group_by(SAMPLE_KEY_COLUMNS)
        .agg(
            pl.col(stored_column).first(),
            pl.col(stored_column).n_unique().alias("__position_count"),
        )
        .sort(SAMPLE_KEY_COLUMNS)
    )
    if positions.get_column("__position_count").max() != 1:
        raise ValueError(
            f"Curriculum column {column!r} must be identical per subsequence"
        )
    return torch.tensor(
        positions.get_column(stored_column).to_numpy(), dtype=torch.int64
    )


def logical_sample_positions(
    stored_positions: Tensor | None, sample_index: WindowSampleIndex
) -> Tensor | None:
    """Expand stored-subsequence positions to logical model-window positions."""
    if stored_positions is None:
        return None
    if stored_positions.shape[0] != sample_index.counts.shape[0]:
        raise ValueError("Curriculum value count does not match the subsequence count")
    return torch.repeat_interleave(
        stored_positions.to(dtype=torch.int64, device="cpu"),
        sample_index.counts,
        dim=0,
    )


@dataclass(frozen=True)
class SampleOrderPlan:
    """A reusable position-sorted index with deterministic epoch tie shuffling."""

    base_indices: Tensor
    group_bounds: tuple[tuple[int, int], ...]
    curriculum: bool

    @classmethod
    def build(
        cls, sample_count: int, sample_positions: Tensor | None
    ) -> "SampleOrderPlan":
        if sample_positions is None:
            return cls(torch.arange(sample_count), (), False)
        positions = sample_positions.to(dtype=torch.int64, device="cpu")
        if tuple(positions.shape) != (sample_count,):
            raise ValueError(f"Curriculum values must have shape [{sample_count}]")
        base_indices = torch.argsort(positions, stable=True)
        sorted_positions = positions[base_indices]
        if sample_count == 0:
            bounds: tuple[tuple[int, int], ...] = ()
        else:
            boundaries = (
                torch.nonzero(sorted_positions[1:] != sorted_positions[:-1])
                .flatten()
                .add(1)
                .tolist()
            )
            starts = [0, *boundaries]
            stops = [*boundaries, sample_count]
            bounds = tuple(zip(starts, stops))
        return cls(base_indices, bounds, True)

    def indices_for_epoch(self, *, seed: int, epoch: int, shuffle: bool) -> Tensor:
        """Return this epoch's logical sample order."""
        if not shuffle:
            return torch.arange(len(self.base_indices))
        generator = torch.Generator().manual_seed(seed + epoch)
        if not self.curriculum:
            return self.base_indices[
                torch.randperm(len(self.base_indices), generator=generator)
            ]
        indices = self.base_indices.clone()
        for start, stop in self.group_bounds:
            size = stop - start
            if size > 1:
                permutation = torch.randperm(size, generator=generator)
                indices[start:stop] = indices[start:stop][permutation]
        return indices


def configured_file_order(config: object) -> str:
    """Read the validated per-part file-order policy, tolerating test doubles."""
    value = getattr(getattr(config, "part", None), "file_order", "shuffled")
    return value if value in {"shuffled", "name"} else "shuffled"


def curriculum_training_enabled(config: object) -> bool:
    """Return the validated training-plan curriculum switch."""
    value = getattr(config, "curriculum_training", False)
    return value if isinstance(value, bool) else False


def configured_curriculum_column(config: object) -> str | None:
    """Read the validated training-plan curriculum column."""
    value = getattr(config, "curriculum_column", None)
    return value if isinstance(value, str) else None


def curriculum_sample_positions(
    config: object,
    sample_positions: Tensor | None,
    source: str,
    stored_columns: tuple[str, ...] = (),
) -> Tensor | None:
    """Select curriculum metadata or fail clearly when the enabled data lacks it."""
    if not curriculum_training_enabled(config):
        return None
    if sample_positions is None:
        column = configured_curriculum_column(config)
        raise ValueError(
            f"curriculum_training is enabled for {source}, but curriculum column "
            f"{column!r} is unavailable in the preprocessed data."
        )
    column = configured_curriculum_column(config)
    if sample_positions.ndim == 2:
        if column not in stored_columns:
            raise ValueError(
                f"Curriculum column {column!r} is unavailable in {source}; "
                f"available columns are {list(stored_columns)!r}."
            )
        sample_positions = sample_positions[:, stored_columns.index(column)]
    return sample_positions


def curriculum_positions_from_parquet(
    config: object, data: pl.DataFrame, source: str
) -> Tensor | None:
    """Read curriculum values only when curriculum training is enabled."""
    if not curriculum_training_enabled(config):
        return None
    column = configured_curriculum_column(config)
    return curriculum_sample_positions(
        config, sample_positions_from_parquet(data, column or ""), source
    )


def validate_folder_curriculum(config: object, metadata: dict, source: str) -> None:
    """Fail before iteration when a folder lacks curriculum-capable metadata."""
    if not curriculum_training_enabled(config):
        return
    column = configured_curriculum_column(config)
    stored_columns = curriculum_columns(metadata.get("curriculum_column"))
    if not stored_columns and metadata.get("sample_positions"):
        stored_columns = (LEGACY_SAMPLE_POSITION_COLUMN,)
    if column not in stored_columns:
        raise ValueError(
            f"curriculum_training is enabled for {source}, but its metadata.json "
            f"does not declare curriculum column {column!r}. Reprocess the dataset "
            "with that curriculum column."
        )


def epoch_file_order(
    file_count: int, *, seed: int, epoch: int, shuffle: bool, policy: str
) -> list[int]:
    """Return deterministic physical-file order for one epoch."""
    if not shuffle or policy == "name":
        return list(range(file_count))
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(file_count, generator=generator).tolist()


def concatenate_file_orders(
    plans: list[tuple[int, SampleOrderPlan]],
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    file_order: str,
) -> Tensor:
    """Compose per-file orders without imposing a folder-global curriculum."""
    curriculum_files = [plan.curriculum for _, plan in plans]
    if any(curriculum_files) and not all(curriculum_files):
        raise ValueError("Curriculum values must be present in every data file or none")
    if file_order == "shuffled" and not any(curriculum_files):
        sample_count = sum(len(plan.base_indices) for _, plan in plans)
        return SampleOrderPlan.build(sample_count, None).indices_for_epoch(
            seed=seed, epoch=epoch, shuffle=shuffle
        )
    order = epoch_file_order(
        len(plans),
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
        policy=file_order,
    )
    return torch.cat(
        [
            plans[file_id][1].indices_for_epoch(
                seed=seed + file_id + 1, epoch=epoch, shuffle=shuffle
            )
            + plans[file_id][0]
            for file_id in order
        ]
    )
