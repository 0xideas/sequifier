"""Coordinated, non-mutating model export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from sequifier.artifacts.model_export import model_execution_config
from sequifier.export.embedding import EmbeddingModelExporter
from sequifier.export.onnx import OnnxModelExporter
from sequifier.export.pytorch import PyTorchModelExporter
from sequifier.logging_paths import model_artifact_path
from sequifier.model.embedding import ONNX_EMBEDDING_LAYER_NAMES_KEY
from sequifier.model.execution import ModelExecution
from sequifier.model.execution_schema import MODEL_CONFIG_KEY
from sequifier.model.factory import build_transformer_network
from sequifier.runtime.random_state import RandomStateManager
from sequifier.special_tokens import ONNX_CATEGORICAL_TARGET_CODECS_KEY


@dataclass(frozen=True)
class ExportOptions:
    suffix: str
    epoch: int


@dataclass(frozen=True)
class ExportResult:
    paths: tuple[Path, ...] = ()
    network: Any | None = field(default=None, compare=False)


class Exporter(Protocol):
    def export(
        self, network: Any, config: Any, destination: Path, options: ExportOptions
    ) -> Path: ...


class ExportService:
    def __init__(self, config: Any, rank: int) -> None:
        self.config = config
        self.rank = rank
        self.pytorch = PyTorchModelExporter()
        self.onnx = OnnxModelExporter()
        self.embedding = EmbeddingModelExporter()

    def export(self, network, state_dict, options):
        manager = RandomStateManager(torch.device("cpu"))
        state = manager.capture_local()
        try:
            return self._export(network, state_dict, options)
        finally:
            manager.restore(state)

    def _export(
        self,
        network: Any,
        state_dict: dict[str, Any],
        options: ExportOptions,
    ) -> ExportResult:
        if self.rank != 0:
            return ExportResult()
        del network
        execution = ModelExecution.from_config(self.config)
        export_network = build_transformer_network(
            execution, device=torch.device("cpu"), initialize=False
        )
        export_network.load_state_dict(state_dict)
        from sequifier.model.factory import validate_hash_contracts

        validate_hash_contracts(export_network, execution)
        export_network.eval()
        paths: list[Path] = []
        export = self.config.run.export
        if export.prediction and export.prediction.pt:
            path = model_artifact_path(
                self.config.project_root,
                self.config.model_name,
                f"{options.suffix}-{options.epoch}",
                "pt",
            )
            exported_path = self.pytorch.export(
                export_network, self.config, path, options
            )
            paths.append(exported_path)
        if export.embedding and export.embedding.pt:
            embedding_options = ExportOptions(
                f"{options.suffix}-embedding", options.epoch
            )
            path = model_artifact_path(
                self.config.project_root,
                self.config.model_name,
                f"{embedding_options.suffix}-{embedding_options.epoch}",
                "pt",
            )
            paths.append(
                self.pytorch.export(
                    export_network, self.config, path, embedding_options
                )
            )
        prediction_graph = export.prediction.onnx if export.prediction else None
        embedding_graph = export.embedding.onnx if export.embedding else None
        if prediction_graph or embedding_graph:
            contract = model_execution_config(execution)
            settings = execution.options
            datasets = self.config.dataset_training
            dataset_count = len(datasets)
            for dataset_name, dataset in datasets.items():
                name = dataset.model_interface
                interface = execution.interface(name)
                metadata = {
                    MODEL_CONFIG_KEY: json.dumps(
                        {**contract, "interfaces": {name: contract["interfaces"][name]}}
                    ),
                    ONNX_CATEGORICAL_TARGET_CODECS_KEY: json.dumps(
                        interface.target_decoder_ids
                    ),
                }
                if prediction_graph:
                    path = model_artifact_path(
                        self.config.project_root,
                        self.config.model_name,
                        f"{options.suffix}-{options.epoch}",
                        "onnx",
                        dataset_name=dataset_name,
                        dataset_count=dataset_count,
                    )
                    paths.append(
                        self.onnx.export(
                            export_network,
                            interface,
                            path,
                            interface_name=name,
                            batch_size=prediction_graph.batch_size,
                            context_length=settings.context_length,
                            training=prediction_graph.with_dropout,
                            metadata=metadata,
                        )
                    )
                if embedding_graph:
                    embedding = self.embedding.build(export_network, name, execution)
                    path = model_artifact_path(
                        self.config.project_root,
                        self.config.model_name,
                        f"{options.suffix}-embedding-{options.epoch}",
                        "onnx",
                        dataset_name=dataset_name,
                        dataset_count=dataset_count,
                    )
                    paths.append(
                        self.onnx.export(
                            embedding,
                            interface,
                            path,
                            interface_name=None,
                            batch_size=embedding_graph.batch_size,
                            context_length=settings.context_length,
                            training=embedding_graph.with_dropout,
                            metadata={
                                **metadata,
                                ONNX_EMBEDDING_LAYER_NAMES_KEY: json.dumps(
                                    execution.embedding_layer_names
                                ),
                            },
                        )
                    )
        return ExportResult(
            paths=tuple(paths),
            network=export_network,
        )
