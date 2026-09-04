"""Dataset, part, source, and dataset-specific freezing runtime structures."""

from __future__ import annotations

import random
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from sequifier.config.train_config import (
    DatasetFreezingSpecModel,
    ResolvedDatasetTrainingSpec,
    ResolvedSequifierConfig,
    ResolvedTrainingPhase,
    ResolvedTrainingSource,
    dataset_part_view,
)
from sequifier.io.batch import SequifierBatch
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
from sequifier.model.parameter_groups import semantic_parameter_groups
from sequifier.objectives import Objective
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
    loaders: dict[Literal["training", "validation"], DataLoader] = field(
        default_factory=dict,
        init=False,
    )
    generators: dict[Literal["training", "validation"], torch.Generator] = field(
        default_factory=dict,
        init=False,
    )

    @beartype
    def build(self, split: Literal["training", "validation"]) -> DataLoader:
        global_spec = self.config.global_training
        if split in self.loaders:
            return self.loaders[split]

        dataset_config: Any = dataset_part_view(
            self.config, self.dataset_name, self.part_name
        )
        part = self.config.dataset_training[self.dataset_name].parts[self.part_name]
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
            dataset = SequifierDatasetFromFile(path, dataset_config, shuffle=shuffle)
        elif global_spec.read_format == "pt":
            cls = (
                SequifierDatasetFromFolderPt
                if global_spec.load_full_data_to_ram
                else SequifierDatasetFromFolderPtLazy
            )
            dataset = cls(path, dataset_config, shuffle=shuffle)
        elif global_spec.read_format == "parquet":
            cls = (
                SequifierDatasetFromFolderParquet
                if global_spec.load_full_data_to_ram
                else SequifierDatasetFromFolderParquetLazy
            )
            dataset = cls(path, dataset_config, shuffle=shuffle)
        else:
            raise ValueError(
                f"Folder parts do not support read_format={global_spec.read_format!r}"
            )
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

    def loader(self, split: Literal["training", "validation"]) -> DataLoader:
        return self.factory.build(split)

    def iter_loader(self, split: Literal["training", "validation"]) -> Iterator[Any]:
        loader = self.loader(split)
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
    objective: Objective
    parts: dict[str, DatasetPartRuntime]
    criteria: dict[str, nn.Module]
    loss_weights: Optional[dict[str, float]]
    class_weights: Optional[dict[str, list[float]]]
    frozen_parameter_ids: frozenset[int]
    runtime_metadata: Any
    metrics: DatasetMetrics = field(default_factory=DatasetMetrics)

    @property
    def freeze_policy(self) -> DatasetFreezingSpecModel:
        return self.config.freeze


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


@dataclass
class TrainingSourceRuntime:
    config: ResolvedTrainingSource
    dataset: DatasetRuntime
    part_names: tuple[str, ...]
    loaders: dict[str, list[tuple[str, DataLoader]]] = field(default_factory=dict)

    @beartype
    def _loaders(self, split: Literal["training", "validation"]):
        if split not in self.loaders:
            self.loaders[split] = [
                (
                    part_name,
                    self.dataset.parts[part_name].loader(split),
                )
                for part_name in self.part_names
            ]
        return self.loaders[split]

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
            for batch in self.dataset.parts[part_name].iter_loader(split):
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
    dataset: DatasetRuntime

    @beartype
    def set_epoch(
        self, epoch: int, split: Literal["training", "validation"]
    ) -> None: ...

    @beartype
    def iter_batches(
        self, split: Literal["training", "validation"] = "training"
    ) -> Iterator[Any]: ...


@beartype
def _policy_parameter_ids(module: nn.Module, policy: Any, usage: str) -> set[int]:
    if not policy.has_freezing_policy:
        return set()
    groups = semantic_parameter_groups(module)
    configured = set(
        policy.freeze if policy.freeze is not None else policy.freezing_except or []
    )
    matched = {group for group in configured if groups.get(group)}
    unmatched = configured - matched
    if unmatched and policy.freezing_except is not None:
        raise ValueError(
            f"{usage} freezing_except groups matched no parameters: "
            f"{', '.join(sorted(unmatched))}"
        )
    if unmatched:
        warnings.warn(
            f"{usage} freezing groups matched no parameters: {sorted(unmatched)}",
            stacklevel=2,
        )
    selected = {
        id(parameter) for group in configured for parameter in groups.get(group, ())
    }
    if policy.freeze is not None:
        return selected
    return {id(parameter) for parameter in module.parameters()} - selected


@beartype
def frozen_parameter_ids(
    network: ComposableTransformerNetwork,
    interface_name: str,
    freeze: DatasetFreezingSpecModel,
) -> frozenset[int]:
    route = network.interfaces[interface_name]
    frozen = _policy_parameter_ids(network.backbone, freeze.backbone, "backbone")
    frozen.update(_policy_parameter_ids(route.ingestion, freeze.ingestion, "ingestion"))
    if freeze.ingestion_adapter:
        frozen.update(
            id(parameter) for parameter in route.ingestion_adapter.parameters()
        )
    frozen.update(_policy_parameter_ids(route.decoder, freeze.decoder, "decoder"))
    return frozenset(frozen)


@beartype
def _criterion_modules(
    dataset: ResolvedDatasetTrainingSpec, device: torch.device
) -> dict[str, nn.Module]:
    modules = {}
    interface = dataset.interface
    for target in interface.target_columns:
        criterion_class = getattr(torch.nn, dataset.criterion[target])
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


@beartype
def build_dataset_runtimes(
    config: ResolvedSequifierConfig,
    network: ComposableTransformerNetwork,
    device: torch.device,
    *,
    objectives: dict[str, Objective],
    runtime_metadata: Any,
) -> DatasetRuntimeRegistry:
    runtimes = {}
    for name, dataset in config.dataset_training.items():
        frozen = frozen_parameter_ids(network, dataset.model_interface, dataset.freeze)
        route = network.interfaces[dataset.model_interface]
        active_parameter_ids = {
            id(parameter)
            for module in (network.backbone, route)
            for parameter in module.parameters()
        }
        if frozen >= active_parameter_ids:
            raise ValueError(f"Dataset {name!r} leaves no trainable parameters")
        runtimes[name] = DatasetRuntime(
            name=name,
            config=dataset,
            interface_name=dataset.model_interface,
            objective=objectives[dataset.model_interface],
            parts={
                part_name: DatasetPartRuntime(
                    part_name, PartLoaderFactory(config, name, part_name)
                )
                for part_name in dataset.parts
            },
            criteria=_criterion_modules(dataset, device),
            loss_weights=dataset.loss_weights,
            class_weights=dataset.class_weights,
            frozen_parameter_ids=frozen,
            runtime_metadata=runtime_metadata.interfaces[dataset.model_interface],
        )

    permanently_frozen = set.intersection(
        *(set(runtime.frozen_parameter_ids) for runtime in runtimes.values())
    )
    for parameter in network.parameters():
        if id(parameter) in permanently_frozen:
            parameter.requires_grad_(False)
    return DatasetRuntimeRegistry(runtimes)


@beartype
def build_source_runtime(
    source: ResolvedTrainingSource,
    datasets: DatasetRuntimeRegistry,
) -> TrainingSourceRuntime:
    dataset = datasets.resolve(source.dataset)
    part_names = (
        (source.part,) if source.part is not None else tuple(dataset.config.parts)
    )
    return TrainingSourceRuntime(source, dataset, part_names)


class SourceScheduler:
    """Deterministic sequential/interleaved selection for one phase epoch."""

    @beartype
    def __init__(
        self,
        phase: ResolvedTrainingPhase,
        sources: Sequence[ScheduledSource],
        *,
        seed: int,
        phase_index: int = 0,
    ) -> None:
        self.phase = phase
        self.sources = sources
        self.phase_index = phase_index
        self.rng = random.Random(seed)
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
                weights=[self.phase.sources[index].weight for index in active],
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
                dict.fromkeys(source.dataset.config.name for source in self.sources)
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
                    else list(source.dataset.config.parts).index(source.config.part)
                )
                record = (
                    self.phase_index,
                    selected,
                    dataset_indices[source.dataset.config.name],
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
                        else list(selected_source.dataset.config.parts).index(
                            selected_source.config.part
                        )
                    )
                    if (
                        record[2]
                        != dataset_indices[selected_source.dataset.config.name]
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
