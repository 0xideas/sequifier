"""Model execution facts, independent of dataset ownership and learning phases."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelExecution:
    model: Any
    interfaces: dict
    runtime_settings: Any
    device: str = "cpu"
    project_root: str = "."
    model_name: str = "loaded-model"
    seed: int = 0
    embedding_layer_names: list[str] = field(
        default_factory=lambda: ["backbone.final_norm"]
    )

    @classmethod
    def from_config(cls, config):
        if isinstance(config, cls):
            return config
        interfaces = {}
        for dataset in config.dataset_training.values():
            interfaces.setdefault(dataset.model_interface, dataset.interface)
        execution = cls(config.model, interfaces, config.runtime_settings)
        for name in (
            "device",
            "project_root",
            "model_name",
            "seed",
            "embedding_layer_names",
        ):
            setattr(execution, name, getattr(config, name, getattr(execution, name)))
        return execution

    def interface(self, name):
        return self.interfaces[name]

    def validate(self):
        from sequifier.config.train_config import interface_build_view
        from sequifier.model.decoders import resolve_decoding_plan
        from sequifier.model.ingestion_compiler import resolve_ingestion_plan

        for interface in self.interfaces.values():
            view = interface_build_view(self, interface)
            resolve_ingestion_plan(view)
            resolve_decoding_plan(view)
        return self

    def dataset_interface(self, name, spec, metadata, storage_layout, runtime_settings):
        """Resolve a storage view while retaining model-owned task parameters."""
        from sequifier.config.train_config import _resolve_interface

        interface = _resolve_interface(
            name, spec, metadata, runtime_settings, storage_layout
        )
        previous = self.interface(name)
        interface.policy = previous.policy
        interface.prediction_distributions = dict(previous.prediction_distributions)
        return interface

    def dataset(self, name):
        """Construct the resolved loss and interface view needed by execution."""
        from sequifier.config.train_config import ResolvedDatasetTrainingSpec

        interface = self.interfaces[name]
        return ResolvedDatasetTrainingSpec(
            name=name,
            model_interface=name,
            interface=interface,
            parts={},
            criterion={
                target: "CrossEntropyLoss" if kind == "categorical" else "MSELoss"
                for target, kind in interface.target_column_types.items()
            },
            class_share_log_columns=[],
        )

    def training_config(self, *, device=None, **options):
        """Adapt execution to the public, serialized training schema at its boundary."""
        from sequifier.config.train_config import ResolvedSequifierConfig

        return ResolvedSequifierConfig(
            runtime_settings=self.runtime_settings,
            model=self.model,
            dataset_training={name: self.dataset(name) for name in self.interfaces},
            **{
                "project_root": self.project_root,
                "model_name": self.model_name,
                "seed": self.seed,
                "device": self.device if device is None else device,
                "training_plan": [],
                "export_generative_model": True,
                "export_embedding_model": False,
                "export_onnx": False,
                "export_pt": False,
                "embedding_layer_names": list(self.embedding_layer_names),
                **options,
            },
        )

    def retain_interfaces(self, config):
        """Keep inactive interfaces when selecting supervised datasets."""
        existing = {
            dataset.model_interface for dataset in config.dataset_training.values()
        }
        for name in self.model.interfaces:
            if name in existing:
                continue
            key = name
            while key in config.dataset_training:
                key += "_unused"
            config.dataset_training[key] = self.dataset(name).model_copy(
                update={"name": key}
            )
