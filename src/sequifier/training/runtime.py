"""Dataset, part, source, and dataset-specific freezing runtime structures."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from sequifier.config.train_config import (
    ResolvedDatasetTrainingSpec,
    ResolvedModelInterface,
    ResolvedSequifierConfig,
    ResolvedTrainingPhase,
    ResolvedTrainingSource,
    SelectedObjectiveConfig,
    dataset_part_view,
    interface_build_view,
)
from sequifier.helpers import normalize_path
from sequifier.io.batch import SequifierBatch
from sequifier.io.memory_budget import DataLoadingBudget
from sequifier.io.sequifier_dataset_from_file import SequifierDatasetFromFile
from sequifier.io.sequifier_dataset_from_folder_parquet import (
    SequifierDatasetFromFolderParquet,
)
from sequifier.io.sequifier_dataset_from_folder_parquet_lazy import (
    SequifierDatasetFromFolderParquetLazy,
)
from sequifier.io.sequifier_dataset_from_folder_pt import SequifierDatasetFromFolderPt
from sequifier.io.sequifier_dataset_from_folder_pt_lazy import (
    SequifierDatasetFromFolderPtLazy,
)
from sequifier.model.network import ComposableTransformerNetwork
from sequifier.model.parameter_catalog import ParameterCatalog
from sequifier.objectives import CausalObjective, Objective, create_objective
from sequifier.training.accurate_cross_entropy import AccurateCrossEntropyLoss
from sequifier.training.plan import (
    PhaseLoading,
    SupervisedPhaseView,
    SupervisedRunPlan,
    dataset_learning_view,
)
from sequifier.typechecking import beartype


@dataclass(frozen=True)
class RuntimeBatch:
    dataset: str
    part: str
    batch: SequifierBatch


@dataclass
class PartLoaderFactory:
    config: ResolvedSequifierConfig
    dataset_name: str
    part_name: str
    memory_budget: DataLoadingBudget
    curriculum_eligible: bool
    _loading: PhaseLoading | None = field(default=None, init=False)
    loaders: dict[Literal["training", "validation"], DataLoader] = field(
        default_factory=dict,
        init=False,
    )
    generators: dict[Literal["training", "validation"], torch.Generator] = field(
        default_factory=dict,
        init=False,
    )

    def _build_dataset(
        self, loading: PhaseLoading, split: Literal["training", "validation"]
    ):
        global_spec = self.config.runtime_settings
        curriculum_column = (
            loading.curriculum_column
            if split == "training" and self.curriculum_eligible
            else None
        )
        dataset_config = dataset_part_view(
            self.config,
            self.dataset_name,
            self.part_name,
            batch_size=loading.batch_size(split),
            window_stride=loading.window_stride,
            curriculum_column=curriculum_column,
        )
        part = dataset_config.part
        path = (
            part.training_data_path
            if split == "training"
            else part.validation_data_path
        )
        if path is None:
            raise ValueError(
                f"Part {self.dataset_name}.{self.part_name} has no {split} path"
            )
        shuffle = split == "training"
        if part.storage_form == "file":
            if global_spec.distributed:
                raise ValueError(
                    "Distributed training is not supported with single-file parts"
                )
            split_paths = [
                normalize_path(split_path, self.config.project_root)
                for split_path in part.metadata.split_paths
            ]
            if path in split_paths:
                dataset_config = replace(
                    dataset_config,
                    storage_layout=part.metadata.layout_for_split(
                        split_paths.index(path)
                    ),
                )
            dataset = SequifierDatasetFromFile(path, dataset_config, shuffle=shuffle)
        elif dataset_config.read_format == "pt":
            cls = (
                SequifierDatasetFromFolderPt
                if global_spec.load_full_data_to_ram
                else SequifierDatasetFromFolderPtLazy
            )
            dataset = cls(path, dataset_config, shuffle=shuffle)
        elif dataset_config.read_format == "parquet":
            cls = (
                SequifierDatasetFromFolderParquet
                if global_spec.load_full_data_to_ram
                else SequifierDatasetFromFolderParquetLazy
            )
            dataset = cls(path, dataset_config, shuffle=shuffle)
        else:
            raise ValueError(
                f"Folder parts do not support read_format={dataset_config.read_format!r}"
            )
        self.memory_budget.validate_dataset(
            f"{self.dataset_name}.{self.part_name}.{split}",
            dataset,
            global_spec.num_workers,
        )
        return dataset

    def count_training_batches(self, phase: SupervisedPhaseView) -> int:
        # The dataset emits pre-batched items and the loader uses batch_size=None.
        return len(self._build_dataset(phase.loading, "training"))

    @beartype
    def build(
        self, phase: SupervisedPhaseView, split: Literal["training", "validation"]
    ) -> DataLoader:
        if self._loading != phase.loading:
            self.loaders.clear()
            self._loading = phase.loading
        if split in self.loaders:
            return self.loaders[split]
        dataset = self._build_dataset(phase.loading, split)
        global_spec = self.config.runtime_settings
        generator = self.generators.get(split)
        if generator is None:
            generator = torch.Generator().manual_seed(
                self.config.seed
                + (10_001 if split == "training" else 10_002)
                + list(self.config.dataset_training).index(self.dataset_name) * 101
                + list(self.config.dataset_training[self.dataset_name].parts).index(
                    self.part_name
                )
            )
            self.generators[split] = generator
        loader = DataLoader(
            dataset,
            batch_size=None,
            sampler=None,
            num_workers=global_spec.num_workers,
            pin_memory=self.config.device not in {"mps", "cpu"},
            prefetch_factor=4 if global_spec.num_workers > 0 else None,
            persistent_workers=global_spec.num_workers > 0,
            generator=generator,
        )
        self.loaders[split] = loader
        return loader


@dataclass
class DatasetPartRuntime:
    name: str
    factory: PartLoaderFactory
    iterator_positions: dict[str, int] = field(default_factory=dict)
    iterator_start_states: dict[str, torch.Tensor] = field(default_factory=dict)
    _restore_from_iterator_start: set[str] = field(default_factory=set)

    @property
    def loaders(self) -> dict[Literal["training", "validation"], DataLoader]:
        return self.factory.loaders

    @property
    def loader_generators(
        self,
    ) -> dict[Literal["training", "validation"], torch.Generator]:
        return self.factory.generators

    @property
    def source_metadata(self) -> dict[str, Any]:
        part = self.factory.config.dataset_training[self.factory.dataset_name].parts[
            self.name
        ]
        return {
            "storage_form": part.storage_form,
            "training_data_path": part.training_data_path,
            "validation_data_path": part.validation_data_path,
        }

    def loader(
        self, phase: SupervisedPhaseView, split: Literal["training", "validation"]
    ) -> DataLoader:
        return self.factory.build(phase, split)

    def iter_loader(
        self, phase: SupervisedPhaseView, split: Literal["training", "validation"]
    ) -> Iterator[Any]:
        loader = self.loader(phase, split)
        generator = self.factory.generators[split]
        if split in self._restore_from_iterator_start:
            generator.set_state(self.iterator_start_states[split])
            self._restore_from_iterator_start.remove(split)
        else:
            self.iterator_start_states[split] = generator.get_state().clone()
        yield from loader

    def loader_state_dict(self) -> dict[str, Any]:
        return {
            "generators": {
                split: generator.get_state()
                for split, generator in self.factory.generators.items()
            },
            "iterator_positions": dict(self.iterator_positions),
            "iterator_start_states": dict(self.iterator_start_states),
        }

    def load_loader_state_dict(
        self,
        state: dict[str, Any],
        *,
        resume_from_iterator_start: bool,
    ) -> None:
        for split, generator_state in state.get("generators", {}).items():
            generator = self.factory.generators.get(split)
            if generator is None:
                generator = torch.Generator()
                self.factory.generators[split] = generator
            generator.set_state(generator_state)
        self.iterator_positions = {
            str(key): int(value)
            for key, value in state.get("iterator_positions", {}).items()
        }
        self.iterator_start_states = {
            str(key): value
            for key, value in state.get("iterator_start_states", {}).items()
        }
        self._restore_from_iterator_start = (
            set(self.iterator_start_states) if resume_from_iterator_start else set()
        )


@dataclass
class DatasetMetrics:
    training_batches: int = 0
    evaluation_batches: int = 0


@dataclass
class DatasetRuntime:
    name: str
    config: ResolvedDatasetTrainingSpec
    interface_name: str
    interface: ResolvedModelInterface
    objective: Objective
    parts: dict[str, DatasetPartRuntime]
    criteria: dict[str, nn.Module]
    loss_weights: Optional[dict[str, float]]
    class_weights: Optional[dict[str, list[float]]]
    frozen_parameter_ids: frozenset[int]
    runtime_metadata: Any
    metrics: DatasetMetrics = field(default_factory=DatasetMetrics)


class DatasetRuntimeRegistry:
    def __init__(self, datasets: dict[str, DatasetRuntime]) -> None:
        if not datasets:
            raise ValueError("A training run requires at least one dataset.")
        self._datasets = dict(datasets)

    def resolve(self, dataset_name: str) -> DatasetRuntime:
        try:
            return self._datasets[dataset_name]
        except KeyError as error:
            raise ValueError(f"Unknown dataset runtime {dataset_name!r}.") from error

    def __iter__(self):
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    def values(self):
        return self._datasets.values()

    def items(self):
        return self._datasets.items()


@dataclass(frozen=True)
class PhaseTrainabilityPlan:
    """Phase-owned update masks expressed as stable parameter catalog IDs."""

    parameter_ids_by_dataset: dict[str, frozenset[str]]

    @classmethod
    @beartype
    def resolve(
        cls,
        config: ResolvedSequifierConfig,
        network: ComposableTransformerNetwork,
        phase: ResolvedTrainingPhase,
    ) -> PhaseTrainabilityPlan:
        catalog = ParameterCatalog(network)
        parameter_id_by_identity = {
            id(catalog.parameter(descriptor.parameter_id)): descriptor.parameter_id
            for descriptor in catalog.descriptors()
        }
        parameter_ids_by_dataset = {}
        active_datasets = {ref.dataset for ref in phase.sources}
        phase_parameters = {id(p) for p in network.backbone.parameters()}
        for name in active_datasets:
            phase_parameters.update(
                id(p)
                for p in network.interfaces[
                    config.dataset_training[name].model_interface
                ].parameters()
            )
        default_mask = phase.trainability.policy("model").resolve(
            network, eligible=phase_parameters
        )
        for name, dataset in config.dataset_training.items():
            route = network.interfaces[dataset.model_interface]
            active_parameter_ids = {
                id(parameter)
                for module in (network.backbone, route)
                for parameter in module.parameters()
            }
            override = phase.trainability.by_dataset.get(name)
            if name not in active_datasets:
                frozen = frozenset()
            elif override is not None and "model" in override.roles:
                frozen = override.roles["model"].resolve(
                    network, eligible=active_parameter_ids
                )
            else:
                frozen = default_mask & active_parameter_ids
            trainable = {
                id(p)
                for module in (network.backbone, route)
                for p in module.parameters()
                if p.requires_grad
            }
            if name in active_datasets and not trainable - frozen:
                raise ValueError(f"Dataset {name!r} leaves no trainable parameters")
            missing = frozen.difference(parameter_id_by_identity)
            if missing:
                raise RuntimeError(
                    "Freezing selected parameters outside the model parameter catalog"
                )
            parameter_ids_by_dataset[name] = frozenset(
                parameter_id_by_identity[identity] for identity in frozen
            )
        return cls(parameter_ids_by_dataset)

    @beartype
    def bind(self, network: ComposableTransformerNetwork) -> dict[str, frozenset[int]]:
        catalog = ParameterCatalog(network)
        return {
            dataset_name: frozenset(
                id(catalog.parameter(parameter_id)) for parameter_id in parameter_ids
            )
            for dataset_name, parameter_ids in self.parameter_ids_by_dataset.items()
        }


@dataclass
class TrainingSourceRuntime:
    config: ResolvedTrainingSource
    dataset: DatasetRuntime
    part_names: tuple[str, ...]
    phase: SupervisedPhaseView

    @property
    def dataset_config(self) -> ResolvedDatasetTrainingSpec:
        return self.dataset.config

    @beartype
    def _loaders(self, split: Literal["training", "validation"]):
        return [
            (part_name, self.dataset.parts[part_name].loader(self.phase, split))
            for part_name in self.part_names
        ]

    @beartype
    def set_epoch(self, epoch: int, split: Literal["training", "validation"]):
        for _, loader in self._loaders(split):
            set_epoch = getattr(loader.dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

    @beartype
    def iter_batches(
        self, split: Literal["training", "validation"] = "training"
    ) -> Iterator[RuntimeBatch]:
        for part_name, _ in self._loaders(split):
            for batch in self.dataset.parts[part_name].iter_loader(self.phase, split):
                if not isinstance(batch, SequifierBatch):
                    raise TypeError(
                        "Sequifier loaders must yield SequifierBatch, got "
                        f"{type(batch).__name__}"
                    )
                yield RuntimeBatch(self.dataset.config.name, part_name, batch)

    @beartype
    def num_batches(self, split: Literal["training", "validation"] = "training"):
        return sum(len(loader) for _, loader in self._loaders(split))


@runtime_checkable
class ScheduledSource(Protocol):
    """Structural source contract consumed by the deterministic scheduler."""

    config: ResolvedTrainingSource

    @property
    def dataset_config(self) -> ResolvedDatasetTrainingSpec: ...

    @beartype
    def set_epoch(
        self, epoch: int, split: Literal["training", "validation"]
    ) -> None: ...

    @beartype
    def iter_batches(
        self, split: Literal["training", "validation"] = "training"
    ) -> Iterator[Any]: ...


@beartype
def _criterion_modules(
    dataset: ResolvedDatasetTrainingSpec,
    interface: ResolvedModelInterface,
    device: torch.device,
) -> dict[str, nn.Module]:
    modules = {}
    for target in interface.target_columns:
        if target in interface.categorical_hashing:
            if dataset.class_weights and target in dataset.class_weights:
                raise ValueError(
                    f"class_weights[{target!r}] describes canonical classes, but hash heads "
                    "predict buckets; no unambiguous class-weight conversion is available."
                )
            if dataset.criterion[target] not in {
                "CrossEntropyLoss",
                "AccurateCrossEntropyLoss",
            }:
                raise ValueError(
                    f"Hashed target {target!r} requires CrossEntropyLoss or "
                    "AccurateCrossEntropyLoss with class-index labels."
                )
        criterion_class = (
            AccurateCrossEntropyLoss
            if dataset.criterion[target] == "AccurateCrossEntropyLoss"
            else getattr(torch.nn, dataset.criterion[target])
        )
        kwargs: dict[str, Any] = {"reduction": "none"}
        if dataset.class_weights is not None and target in dataset.class_weights:
            weights = torch.tensor(dataset.class_weights[target])
            if interface.target_column_types[target] == "categorical":
                if weights.numel() == interface.n_classes[target]:
                    weights = weights[interface.target_decoder_ids[target]]
                elif weights.numel() != interface.target_n_classes[target]:
                    raise ValueError(
                        f"class_weights[{target!r}] has incompatible length"
                    )
            kwargs["weight"] = weights
        modules[target] = criterion_class(**kwargs).to(device)
    return modules


def build_dataset_parts(
    config: ResolvedSequifierConfig,
    plan: SupervisedRunPlan,
    memory_budget: DataLoadingBudget,
) -> dict[str, dict[str, DatasetPartRuntime]]:
    training_parts = {
        (source.dataset, part)
        for phase in plan.phases
        for source in phase.spec().sources
        for part in source.parts
    }
    return {
        name: {
            part: DatasetPartRuntime(
                part,
                PartLoaderFactory(
                    config, name, part, memory_budget, (name, part) in training_parts
                ),
            )
            for part in dataset.parts
        }
        for name, dataset in config.dataset_training.items()
    }


def build_dataset_runtimes(
    config: ResolvedSequifierConfig,
    phase: SupervisedPhaseView,
    device: torch.device,
    *,
    parts: dict[str, dict[str, DatasetPartRuntime]],
    metrics: dict[str, DatasetMetrics],
    runtime_metadata: Any,
    masks: dict[str, frozenset[int]],
) -> DatasetRuntimeRegistry:
    spec = phase.spec()
    settings = phase.objective_settings(config.runtime_settings)
    runtimes = {}
    for name, base in config.dataset_training.items():
        dataset = dataset_learning_view(base, spec)
        interface = config.interface(dataset.model_interface)
        objective_view = SelectedObjectiveConfig(
            **vars(interface_build_view(config, interface)), runtime_settings=settings
        )
        objective = (
            CausalObjective(objective_view)
            if name not in spec.learning.datasets
            and objective_view.training_objective == "next_occurrence"
            else create_objective(objective_view)
        )
        runtimes[name] = DatasetRuntime(
            name=name,
            config=dataset,
            interface_name=dataset.model_interface,
            interface=interface,
            objective=objective,
            parts=parts[name],
            criteria=_criterion_modules(dataset, interface, device),
            loss_weights=dataset.loss_weights,
            class_weights=dataset.class_weights,
            frozen_parameter_ids=masks[name],
            runtime_metadata=runtime_metadata.interfaces[dataset.model_interface],
            metrics=metrics[name],
        )
    return DatasetRuntimeRegistry(runtimes)


def build_source_runtime(
    source: ResolvedTrainingSource,
    datasets: DatasetRuntimeRegistry,
    phase: SupervisedPhaseView,
) -> TrainingSourceRuntime:
    dataset = datasets.resolve(source.dataset)
    return TrainingSourceRuntime(source, dataset, source.parts, phase)


class SourceScheduler:
    """Deterministic sequential/interleaved selection for one phase epoch."""

    @beartype
    def __init__(
        self,
        phase: SupervisedPhaseView,
        sources: Sequence[ScheduledSource],
    ) -> None:
        self.phase = phase.spec()
        self.sources = sources
        self.phase_index = phase.index
        self.rng = random.Random(phase.seed)
        self.round_robin_cursor = 0
        self.epoch = 0
        self.consumed_by_source = [0 for _ in sources]
        self.exhausted_sources: set[int] = set()
        self.pending_source: int | None = None
        self.pending_batches = 0
        self.sequential_source = 0

    @beartype
    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "round_robin_cursor": self.round_robin_cursor,
            "epoch": self.epoch,
            "consumed_by_source": list(self.consumed_by_source),
            "exhausted_sources": sorted(self.exhausted_sources),
            "pending_source": self.pending_source,
            "pending_batches": self.pending_batches,
            "sequential_source": self.sequential_source,
        }

    @beartype
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["rng_state"])
        self.round_robin_cursor = int(state["round_robin_cursor"])
        self.epoch = int(state.get("epoch", 0))
        self.consumed_by_source = [
            int(value) for value in state.get("consumed_by_source", [])
        ] or [0 for _ in self.sources]
        self.exhausted_sources = {
            int(value) for value in state.get("exhausted_sources", [])
        }
        pending = state.get("pending_source")
        self.pending_source = None if pending is None else int(pending)
        self.pending_batches = int(state.get("pending_batches", 0))
        self.sequential_source = int(state.get("sequential_source", 0))

    @beartype
    def _choose(self, active: list[int]) -> int:
        if self.phase.selection == "weighted_random":
            return self.rng.choices(
                active,
                weights=[
                    self.sources[index].config.selection_weight for index in active
                ],
                k=1,
            )[0]
        for _ in range(len(self.sources)):
            selected = self.round_robin_cursor % len(self.sources)
            self.round_robin_cursor += 1
            if selected in active:
                return selected
        raise RuntimeError("No active source remains")

    @beartype
    def iter_epoch(self, epoch: int) -> Iterator[RuntimeBatch]:
        for source in self.sources:
            source.set_epoch(epoch, "training")
        if self.epoch != epoch:
            self.epoch = epoch
            self.consumed_by_source = [0 for _ in self.sources]
            self.exhausted_sources = set()
            self.pending_source = None
            self.pending_batches = 0
            self.sequential_source = 0
        iterators = [iter(source.iter_batches("training")) for source in self.sources]
        for source_index, consumed in enumerate(self.consumed_by_source):
            for _ in range(consumed):
                try:
                    next(iterators[source_index])
                except StopIteration as exc:
                    raise RuntimeError(
                        "Saved source position exceeds the reconstructed loader"
                    ) from exc
        if self.phase.mode == "sequential":
            for source_index in range(self.sequential_source, len(self.sources)):
                self.sequential_source = source_index
                for batch in iterators[source_index]:
                    self.consumed_by_source[source_index] += 1
                    yield batch
                self.exhausted_sources.add(source_index)
                self.sequential_source = source_index + 1
            return

        active = [
            index
            for index in range(len(iterators))
            if index not in self.exhausted_sources
        ]
        dataset_indices = {
            name: index
            for index, name in enumerate(
                dict.fromkeys(source.dataset_config.name for source in self.sources)
            )
        }
        while active:
            if self.pending_source is None or self.pending_batches == 0:
                selected = (
                    self._choose(active)
                    if not dist.is_initialized() or dist.get_rank() == 0
                    else 0
                )
                source = self.sources[selected]
                part_index = (
                    -1
                    if source.config.part is None
                    else list(source.dataset_config.parts).index(source.config.part)
                )
                record = (
                    self.phase_index,
                    selected,
                    dataset_indices[source.dataset_config.name],
                    part_index,
                    source.config.batches_per_selection,
                )
                if dist.is_initialized():
                    from sequifier.training.distributed import (
                        broadcast_source_selection,
                    )

                    record = broadcast_source_selection(
                        record if dist.get_rank() == 0 else None
                    )
                    selected = record[1]
                    if record[0] != self.phase_index or selected not in active:
                        raise RuntimeError(
                            "Distributed source selection does not match the "
                            "active phase/source state"
                        )
                    selected_source = self.sources[selected]
                    expected_part = (
                        -1
                        if selected_source.config.part is None
                        else list(selected_source.dataset_config.parts).index(
                            selected_source.config.part
                        )
                    )
                    if (
                        record[2]
                        != dataset_indices[selected_source.dataset_config.name]
                        or record[3] != expected_part
                    ):
                        raise RuntimeError(
                            "Distributed dataset/part selection is inconsistent"
                        )
                self.pending_source = selected
                self.pending_batches = record[4]
            selected = self.pending_source
            iterator = iterators[selected]
            while self.pending_batches > 0:
                try:
                    batch = next(iterator)
                except StopIteration:
                    active.remove(selected)
                    self.exhausted_sources.add(selected)
                    self.pending_source = None
                    self.pending_batches = 0
                    break
                self.consumed_by_source[selected] += 1
                self.pending_batches -= 1
                yield batch
            if self.pending_batches == 0:
                self.pending_source = None


def validate_phase_learning(config, plan: SupervisedRunPlan, device):
    """Validate every phase's losses before native outputs are created."""
    for view in plan.phases:
        phase = view.spec()
        for name, learning in phase.learning.datasets.items():
            dataset = dataset_learning_view(config.dataset_training[name], phase)
            interface = config.interface(dataset.model_interface)
            learning.validate_targets(interface.target_columns, f"{phase.name}: {name}")
            for column in dataset.class_share_log_columns:
                if interface.target_column_types.get(column) != "categorical":
                    raise ValueError(
                        f"{phase.name}: class share columns must be categorical targets"
                    )
            _criterion_modules(dataset, interface, device)
