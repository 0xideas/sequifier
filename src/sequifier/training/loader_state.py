"""Aggregate exact loader-generator and iterator state."""

from __future__ import annotations

from sequifier.artifacts.run_checkpoint import LoaderState
from sequifier.training.runtime import DatasetRuntimeRegistry


class LoaderStateService:
    def state_dict(self, datasets: DatasetRuntimeRegistry) -> LoaderState:
        return LoaderState(
            parts={
                dataset_name: {
                    part_name: part.loader_state_dict()
                    for part_name, part in dataset.parts.items()
                }
                for dataset_name, dataset in datasets.items()
            }
        )

    def load_state_dict(
        self,
        datasets: DatasetRuntimeRegistry,
        state: LoaderState,
        *,
        resume_from_iterator_start: bool,
    ) -> None:
        unknown_datasets = set(state.parts).difference(iter(datasets))
        if unknown_datasets:
            raise ValueError(
                "Checkpoint contains unknown loader datasets: "
                f"{sorted(unknown_datasets)!r}."
            )
        for dataset_name, part_states in state.parts.items():
            dataset = datasets.resolve(dataset_name)
            unknown_parts = set(part_states).difference(dataset.parts)
            if unknown_parts:
                raise ValueError(
                    f"Checkpoint contains unknown loader parts for {dataset_name!r}: "
                    f"{sorted(unknown_parts)!r}."
                )
            for part_name, part_state in part_states.items():
                dataset.parts[part_name].load_loader_state_dict(
                    part_state,
                    resume_from_iterator_start=resume_from_iterator_start,
                )
