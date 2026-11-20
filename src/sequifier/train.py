"""Training command composition and PT inference entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sequifier.artifacts.model_artifact import restore_execution_network
from sequifier.config.inference_common import ResolvedInferenceModel
from sequifier.config.train_config import load_train_config
from sequifier.helpers import get_torch_dtype
from sequifier.model.embedding import embedding_layer_trace_site
from sequifier.model.tracing import CaptureRequest
from sequifier.training.coordinator import run_training as run_training
from sequifier.typechecking import beartype


@dataclass
class LoadedInferenceModel:
    """Execution metadata alongside the sole weight-owning network."""

    network: Any
    config: Any
    interface_name: str

    @property
    def interface(self) -> Any:
        return self.config.interface(self.interface_name)

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.interface.categorical_columns)

    @property
    def target_decoder_ids(self) -> dict[str, list[int]]:
        return dict(self.interface.target_decoder_ids)


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    config_path = args.config_path or "configs/train.yaml"
    run_training(load_train_config(config_path, args_config, args.skip_metadata))


@beartype
def load_inference_model(
    model: ResolvedInferenceModel,
    device: str,
    infer_with_dropout: bool,
) -> LoadedInferenceModel:
    execution, name = model.execution, model.interface_name
    assert model.artifact is not None and execution is not None and name is not None
    network = restore_execution_network(model.artifact, execution, device)
    network.eval()
    if infer_with_dropout:
        for module in network.modules():
            from sequifier.model.layers import SelfAttention

            if isinstance(module, (nn.Dropout, nn.MultiheadAttention, SelfAttention)):
                module.train()
    if not device.startswith("mps"):
        network = torch.compile(network)
    return LoadedInferenceModel(network, execution, name)


def _tensor_batches(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    metadata: list[dict[str, np.ndarray]],
    device: str,
    column_data_types: dict[str, torch.dtype],
):
    categorical = set(model.categorical_columns)
    layer_types = model.config.options.layer_type_dtypes or {}
    reference_dtype = get_torch_dtype(layer_types.get("linear", "float32"))
    for index, features in enumerate(x):
        values = {
            column: torch.from_numpy(array).to(
                device,
                dtype=(
                    torch.uint8
                    if column.startswith("__image:")
                    or column == "__preconditioning_image"
                    else torch.int64
                    if column in categorical
                    else column_data_types.get(column, reference_dtype)
                ),
            )
            for column, array in features.items()
        }
        metadata_values = {
            column: torch.from_numpy(array).to(device)
            for column, array in (metadata[index] if metadata else {}).items()
        }
        yield values, metadata_values


@beartype
def infer_with_embedding_model(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> np.ndarray:
    del size, target_columns
    sites = tuple(
        embedding_layer_trace_site(name) for name in model.config.embedding_layer_names
    )
    outputs = []
    with torch.no_grad():
        for features, metadata_values in _tensor_batches(
            model, x, metadata, device, column_data_types
        ):
            traced = model.network.trace(
                features,
                metadata_values,
                CaptureRequest(sites=sites),
                interface_name=model.interface_name,
            )
            route = model.network.resolve_interface(model.interface_name)
            embedding = torch.cat(
                [
                    traced.captures[site][:, -route.prediction_length :]
                    for site in sites
                ],
                dim=-1,
            )
            array = embedding.cpu().float().numpy()
            outputs.append(array.reshape(-1, array.shape[-1]))
    return np.concatenate(outputs, axis=0)


@beartype
def infer_with_generative_model(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> dict[str, np.ndarray]:
    outputs: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for features, metadata_values in _tensor_batches(
            model, x, metadata, device, column_data_types
        ):
            result = model.network(
                features,
                metadata_values,
                interface_name=model.interface_name,
            )
            route = model.network.resolve_interface(model.interface_name)
            outputs.append(
                {
                    target: (
                        torch.log_softmax(
                            value[:, result.prediction_positions].float(), dim=-1
                        )
                        if route.target_column_types[target] == "categorical"
                        else value[:, result.prediction_positions].float()
                    ).cpu()
                    for target, value in result.logits.items()
                }
            )
    return {
        target: np.concatenate(
            [
                batch[target].numpy().reshape(-1, batch[target].shape[-1])
                for batch in outputs
            ],
            axis=0,
        )[:size]
        for target in target_columns
    }
