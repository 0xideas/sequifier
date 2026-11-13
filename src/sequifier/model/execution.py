"""Model execution facts, independent of dataset ownership and learning phases."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelExecution:
    model: Any
    interfaces: dict
    runtime_settings: Any

    @classmethod
    def from_config(cls, config):
        interfaces = {}
        for dataset in config.dataset_training.values():
            interfaces.setdefault(dataset.model_interface, dataset.interface)
        return cls(config.model, interfaces, config.runtime_settings)

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

    def training_config(self, *, device="cpu", **options):
        """Construct the resolved execution configuration."""
        from sequifier.config.train_config import ResolvedSequifierConfig

        return ResolvedSequifierConfig(
            runtime_settings=self.runtime_settings,
            model=self.model,
            dataset_training={name: self.dataset(name) for name in self.interfaces},
            **{
                "project_root": ".",
                "model_name": "loaded-model",
                "seed": 0,
                "device": device,
                "training_plan": [],
                "export_generative_model": True,
                "export_embedding_model": False,
                "export_onnx": False,
                "export_pt": False,
                "embedding_layer_names": ["backbone.final_norm"],
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
            selected = type(self)(
                self.model,
                {name: self.interfaces[name]},
                config.runtime_settings,
            )
            config.dataset_training[key] = (
                selected.training_config()
                .dataset_training[name]
                .model_copy(update={"name": key})
            )
