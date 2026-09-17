"""Deterministic sample ordering for ordinary and curriculum datasets."""

from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from sequifier.helpers import WindowSampleIndex

SAMPLE_POSITION_COLUMN = "samplePosition"
SAMPLE_KEY_COLUMNS = ["sequenceId", "subsequenceId"]


def sample_positions_from_parquet(data: pl.DataFrame) -> Tensor | None:
    """Return one validated sample position per stored subsequence."""
    if SAMPLE_POSITION_COLUMN not in data.columns:
        return None
    positions = (
        data.group_by(SAMPLE_KEY_COLUMNS)
        .agg(
            pl.col(SAMPLE_POSITION_COLUMN).first(),
            pl.col(SAMPLE_POSITION_COLUMN).n_unique().alias("__position_count"),
        )
        .sort(SAMPLE_KEY_COLUMNS)
    )
    if positions.get_column("__position_count").max() != 1:
        raise ValueError("samplePosition must be identical across each subsequence")
    return torch.tensor(
        positions.get_column(SAMPLE_POSITION_COLUMN).to_numpy(),
        dtype=torch.int64,
    )


def logical_sample_positions(
    stored_positions: Tensor | None, sample_index: WindowSampleIndex
) -> Tensor | None:
    """Expand stored-subsequence positions to logical model-window positions."""
    if stored_positions is None:
        return None
    if tuple(stored_positions.shape) != tuple(sample_index.counts.shape):
        raise ValueError(
            "samplePosition count does not match the stored subsequence count"
        )
    return torch.repeat_interleave(
        stored_positions.to(dtype=torch.int64, device="cpu"), sample_index.counts
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
            raise ValueError(f"samplePosition must have shape [{sample_count}]")
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
        raise ValueError("samplePosition must be present in every data file or none")
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
