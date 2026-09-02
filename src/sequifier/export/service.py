"""Coordinated, non-mutating model export."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from sequifier.export.embedding import EmbeddingModelExporter
from sequifier.export.onnx import OnnxModelExporter
from sequifier.export.pytorch import PyTorchModelExporter
from sequifier.logging_paths import model_artifact_path
from sequifier.model.embedding import ONNX_EMBEDDING_LAYER_NAMES_KEY
from sequifier.model.factory import build_transformer_network
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

    def export(
        self,
        network: Any,
        state_dict: dict[str, Any],
        options: ExportOptions,
    ) -> ExportResult:
        if self.rank != 0:
            return ExportResult()
        del network
        export_config = self.config.model_copy(deep=True)
        export_config.device = "cpu"
        export_config.global_training.distributed = False
        export_config.global_training.data_parallelism = None
        export_config.global_training.torch_compile = "none"
        export_network = build_transformer_network(
            export_config, device=torch.device("cpu"), initialize=False
        ).network
        export_network.load_state_dict(state_dict)
        export_network.eval()
        paths: list[Path] = []
        if self.config.export_pt and self.config.export_generative_model:
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
        if self.config.export_pt and self.config.export_embedding_model:
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
        if self.config.export_onnx:
            onnx_network = export_network
            if any(
                parameter.dtype in (torch.float16, torch.bfloat16, torch.float64)
                for parameter in export_network.parameters()
            ):
                onnx_network = copy.deepcopy(export_network).float()
            count = len(self.config.dataset_training)
            for dataset_name, dataset in self.config.dataset_training.items():
                if self.config.export_generative_model:
                    path = model_artifact_path(
                        self.config.project_root,
                        self.config.model_name,
                        f"{options.suffix}-{options.epoch}",
                        "onnx",
                        dataset_name=dataset_name,
                        dataset_count=count,
                    )
                    paths.append(
                        self.onnx.export(
                            onnx_network,
                            dataset.interface,
                            path,
                            interface_name=dataset.model_interface,
                            batch_size=self.config.global_training.inference_batch_size,
                            context_length=self.config.global_training.context_length,
                            training=self.config.export_with_dropout,
                            metadata={
                                ONNX_CATEGORICAL_TARGET_CODECS_KEY: json.dumps(
                                    dataset.interface.target_decoder_ids
                                )
                            },
                        )
                    )
                if self.config.export_embedding_model:
                    embedding = self.embedding.build(
                        onnx_network, dataset.model_interface, self.config
                    )
                    path = model_artifact_path(
                        self.config.project_root,
                        self.config.model_name,
                        f"{options.suffix}-embedding-{options.epoch}",
                        "onnx",
                        dataset_name=dataset_name,
                        dataset_count=count,
                    )
                    paths.append(
                        self.onnx.export(
                            embedding,
                            dataset.interface,
                            path,
                            interface_name=None,
                            batch_size=self.config.global_training.inference_batch_size,
                            context_length=self.config.global_training.context_length,
                            training=self.config.export_with_dropout,
                            metadata={
                                ONNX_CATEGORICAL_TARGET_CODECS_KEY: json.dumps(
                                    dataset.interface.target_decoder_ids
                                ),
                                ONNX_EMBEDDING_LAYER_NAMES_KEY: json.dumps(
                                    self.config.embedding_layer_names
                                ),
                            },
                        )
                    )
        return ExportResult(
            paths=tuple(paths),
            network=export_network,
        )
