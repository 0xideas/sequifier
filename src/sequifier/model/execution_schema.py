"""Versioned execution descriptors shared by PT, ONNX, and warm-up."""

from dataclasses import asdict, dataclass

import torch

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)

EXECUTION_SCHEMA_KEY = "sequifier.execution_schema"
MODEL_CONFIG_KEY = "sequifier.model_config"
DROPOUT_MODE_KEY = "sequifier.dropout_mode"


@dataclass(frozen=True)
class InputDescriptor:
    name: str
    key: str
    role: str
    dtype: str
    shape: tuple


@dataclass(frozen=True)
class ExecutionSchema:
    inputs: tuple[InputDescriptor, ...]
    outputs: tuple[dict, ...]
    depth_layouts: dict
    n_classes: dict
    version: int = 1
    batch_policy: str = "symbolic"

    @classmethod
    def from_interface(
        cls, interface, context_length, *, embedding=False, embedding_width=None
    ):
        layouts: DepthLayoutRegistryModel | None = getattr(
            interface, "depth_layouts", None
        )
        if layouts is None:
            layouts = DepthLayoutRegistryModel()
        inputs = []
        for index, column in enumerate(interface.input_columns):
            layout = layouts.layout_for_column(column)
            shape = (
                ("batch", context_length, layout.context_length)
                if layout
                else ("batch", context_length)
            )
            inputs.append(
                InputDescriptor(
                    f"feature_{index}",
                    column,
                    "feature",
                    "int64" if column in interface.categorical_columns else "float32",
                    shape,
                )
            )
        inputs.append(
            InputDescriptor(
                "attention_valid_mask",
                "attention_valid_mask",
                "metadata",
                "bool",
                ("batch", context_length),
            )
        )
        for index, (name, layout) in enumerate(sorted(layouts.items())):
            inputs.append(
                InputDescriptor(
                    f"depth_mask_{index}",
                    depth_mask_metadata_key(name),
                    "metadata",
                    "bool",
                    ("batch", context_length, layout.context_length),
                )
            )
        outputs = (
            (
                {
                    "name": "embedding_out",
                    "key": "embedding",
                    "dtype": "float32",
                    "shape": (
                        interface.decoder.prediction_length,
                        "batch",
                        embedding_width,
                    ),
                },
            )
            if embedding
            else tuple(
                {
                    "name": f"output_{index}",
                    "key": column,
                    "dtype": "float32",
                    "shape": (
                        interface.decoder.prediction_length,
                        "batch",
                        interface.target_n_classes[column]
                        if interface.target_column_types[column] == "categorical"
                        else 1,
                    ),
                }
                for index, column in enumerate(sorted(interface.target_columns))
            )
        )
        return cls(
            tuple(inputs),
            outputs,
            layouts.model_dump(mode="json"),
            dict(interface.n_classes),
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if values.get("version") != 1 or values.get("batch_policy") != "symbolic":
            raise ValueError("Unsupported execution schema version or batch policy")
        descriptors = tuple(
            InputDescriptor(**{**value, "shape": tuple(value["shape"])})
            for value in values["inputs"]
        )
        if any(
            d.role not in {"feature", "metadata"}
            or d.dtype not in {"float32", "int64", "bool"}
            or not d.shape
            or d.shape[0] != "batch"
            for d in descriptors
        ):
            raise ValueError("Invalid execution input descriptor")
        if len({d.name for d in descriptors}) != len(descriptors):
            raise ValueError("Duplicate execution input names")
        return cls(
            descriptors,
            tuple(values["outputs"]),
            values["depth_layouts"],
            values["n_classes"],
        )

    def example_inputs(self, batch_size=2, device="cpu"):
        values = []
        for item in self.inputs:
            shape = tuple(
                batch_size if size == "batch" else size for size in item.shape
            )
            if item.role == "metadata":
                value = torch.ones(shape, dtype=torch.bool, device=device)
                if len(shape) == 3 and batch_size > 1:
                    value[1, :, 1:] = False
                if shape[1] > 1:
                    value[0, 0] = False
            elif item.dtype == "int64":
                value = torch.full(
                    shape,
                    3 if self.n_classes.get(item.key, 0) > 3 else 0,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                value = (
                    torch.linspace(0.1, 0.9, steps=shape[1], device=device)
                    .reshape(1, shape[1], *([1] if len(shape) == 3 else []))
                    .expand(shape)
                    .clone()
                )
            values.append(value)
        return tuple(values)

    def bind(self, values):
        features, metadata = {}, {}
        for item, value in zip(self.inputs, values):
            (features if item.role == "feature" else metadata)[item.key] = value
        return features, metadata

    def validate(self, features, metadata):
        from sequifier.io.pt_payload import validate_tensor_inputs

        layouts = DepthLayoutRegistryModel.model_validate(self.depth_layouts)
        selected = {
            item.key: features[item.key]
            for item in self.inputs
            if item.role == "feature"
        }
        for item in self.inputs:
            value = (features if item.role == "feature" else metadata)[item.key]
            if value.ndim != len(item.shape) or any(
                isinstance(size, int) and value.shape[axis] != size
                for axis, size in enumerate(item.shape)
            ):
                raise ValueError(
                    f"{item.key}: expected shape {item.shape}, got {tuple(value.shape)}"
                )
        validate_tensor_inputs(
            selected,
            {name: metadata[depth_mask_metadata_key(name)] for name in layouts.root},
            layouts,
            n_classes=self.n_classes,
            attention_valid_mask=metadata.get("attention_valid_mask"),
        )
