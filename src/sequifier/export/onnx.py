from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor, nn


class _OnnxWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        feature_columns: tuple[str, ...],
        interface_name: str | None,
    ):
        super().__init__()
        self.model = model
        self.feature_columns = feature_columns
        self.interface_name = interface_name

    def forward(self, *values: Tensor):
        features = dict(zip(self.feature_columns, values[:-1]))
        metadata = {"attention_valid_mask": values[-1]}
        if self.interface_name is None:
            return self.model(features, metadata)
        output = self.model(features, metadata, interface_name=self.interface_name)
        route = self.model.resolve_interface(self.interface_name)
        return tuple(
            (
                torch.log_softmax(
                    output.logits[name][:, output.prediction_positions].float(),
                    dim=-1,
                )
                if route.target_column_types[name] == "categorical"
                else output.logits[name][:, output.prediction_positions]
            ).transpose(0, 1)
            for name in sorted(output.logits)
        )


class OnnxModelExporter:
    def export(
        self,
        model: nn.Module,
        interface: Any,
        destination: Path,
        *,
        interface_name: str | None,
        batch_size: int,
        context_length: int,
        training: bool,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        feature_columns = tuple(interface.input_columns)
        inputs = []
        for column in feature_columns:
            dtype = (
                torch.int64
                if column in interface.categorical_columns
                else torch.float32
            )
            inputs.append(torch.ones((batch_size, context_length), dtype=dtype))
        inputs.append(torch.ones((batch_size, context_length), dtype=torch.bool))
        destination.parent.mkdir(parents=True, exist_ok=True)
        wrapper = _OnnxWrapper(model, feature_columns, interface_name)
        wrapper.train(training)
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            destination,
            export_params=True,
            opset_version=18,
            dynamo=True,
            input_names=[
                *(f"{name}_in" for name in feature_columns),
                "attention_valid_mask",
            ],
            output_names=[f"{name}_out" for name in sorted(interface.target_columns)]
            if interface_name is not None
            else ["embedding_out"],
        )
        if metadata:
            model = onnx.load(destination)  # pyright: ignore[reportAttributeAccessIssue]
            for key, value in metadata.items():
                entry = model.metadata_props.add()
                entry.key = key
                entry.value = value
            onnx.save(model, destination)  # pyright: ignore[reportAttributeAccessIssue]
        return destination
