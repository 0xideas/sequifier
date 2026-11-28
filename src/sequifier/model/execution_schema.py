"""Versioned execution descriptors shared by PT, ONNX, and warm-up."""

from dataclasses import asdict, dataclass, replace

import torch

from sequifier.config.depth_layout import (
    DepthLayoutRegistryModel,
    depth_mask_metadata_key,
)
from sequifier.config.image_config import image_views
from sequifier.contract_versions import ContractVersions, require_contract_version

EXECUTION_SCHEMA_KEY = "sequifier.execution_schema"
MODEL_CONFIG_KEY = "sequifier.model_config"
POLICY_GRAPH_KEY = "sequifier.policy_graph"
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
    version: int = ContractVersions.MODEL_EXECUTION_SCHEMA
    batch_policy: str = "symbolic"

    def __post_init__(self):
        require_contract_version(
            "model execution schema",
            self.version,
            ContractVersions.MODEL_EXECUTION_SCHEMA,
        )

    @classmethod
    def from_interface(
        cls, interface, context_length, *, embedding=False, embedding_width=None
    ):
        layouts: DepthLayoutRegistryModel | None = interface.depth_layouts
        if layouts is None:
            layouts = DepthLayoutRegistryModel()
        inputs = []
        for index, column in enumerate(interface.input_columns):
            layout = layouts.layout_for_column(column)
            shape = (
                ("batch", context_length, layout.max_depth_length)
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
        static_columns = (
            interface.static_conditioning.columns
            if interface.static_conditioning is not None
            else []
        )
        for index, column in enumerate(static_columns):
            inputs.append(
                InputDescriptor(
                    f"static_feature_{index}",
                    column,
                    "static_feature",
                    (
                        "int64"
                        if column in interface.static_categorical_columns
                        else "float32"
                    ),
                    ("batch",),
                )
            )

        for index, name in enumerate(sorted(image_views(interface.ingestion))):
            source = interface.image_sources[name]
            capacity = source.frame_capacity
            inputs.append(
                InputDescriptor(
                    f"image_{index}",
                    f"__image:{name}",
                    "image",
                    "uint8",
                    (
                        "batch",
                        context_length,
                        capacity,
                        3,
                        source.crop_size,
                        source.crop_size,
                    ),
                )
            )
            inputs.append(
                InputDescriptor(
                    f"image_mask_{index}",
                    f"__image_mask:{name}",
                    "metadata",
                    "bool",
                    ("batch", context_length, capacity),
                )
            )
        if interface.preconditioning_image_encoder is not None:
            source = interface.preconditioning_image
            inputs.append(
                InputDescriptor(
                    "preconditioning_image",
                    "__preconditioning_image",
                    "image",
                    "uint8",
                    ("batch", len(source.views), 3, source.crop_size, source.crop_size),
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
        for index, (name, layout) in enumerate(
            sorted(layouts.relevant_layouts(interface.input_columns).items())
        ):
            inputs.append(
                InputDescriptor(
                    f"depth_mask_{index}",
                    depth_mask_metadata_key(name),
                    "metadata",
                    "bool",
                    ("batch", context_length, layout.max_depth_length),
                )
            )
        prediction_length = interface.decoder.prediction_length
        outputs = []
        if embedding:
            outputs.append(
                {
                    "name": "embedding_out",
                    "key": "embedding",
                    "dtype": "float32",
                    "shape": (prediction_length, "batch", embedding_width),
                }
            )
        else:
            static_targets = set(interface.static_target_columns)
            for column in sorted(interface.target_columns):
                length = 1 if column in static_targets else prediction_length
                layout = layouts.layout_for_column(column)
                depth_axis = (layout.max_depth_length,) if layout else ()
                width = interface.target_n_classes.get(column, 1)
                outputs.append(
                    {
                        "name": f"output_{len(outputs)}",
                        "key": column,
                        "dtype": "float32",
                        "shape": (length, "batch", *depth_axis, width),
                    }
                )
            for name, layout in layouts.relevant_layouts(
                interface.target_columns
            ).items():
                outputs.append(
                    {
                        "name": f"output_{len(outputs)}",
                        "key": f"depth_occupancy:{name}",
                        "dtype": "float32",
                        "shape": (
                            prediction_length,
                            "batch",
                            layout.max_depth_length,
                            1,
                        ),
                    }
                )
        return cls(
            tuple(inputs),
            tuple(outputs),
            layouts.relevant_layouts(interface.input_columns).model_dump(mode="json"),
            dict(interface.n_classes),
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        require_contract_version(
            "model execution schema",
            values.get("version") if isinstance(values, dict) else None,
            ContractVersions.MODEL_EXECUTION_SCHEMA,
        )
        if values.get("batch_policy") != "symbolic":
            raise ValueError("Unsupported execution schema batch policy")
        descriptors = tuple(
            InputDescriptor(**{**value, "shape": tuple(value["shape"])})
            for value in values["inputs"]
        )
        if any(
            d.role
            not in {"feature", "static_feature", "image", "metadata", "policy_control"}
            or (
                d.dtype != "float64"
                if d.role == "policy_control"
                else d.dtype not in {"float32", "int64", "bool", "uint8"}
            )
            or not d.shape
            or d.shape[0] != "batch"
            for d in descriptors
        ):
            raise ValueError("Invalid execution input descriptor")
        if len({d.name for d in descriptors}) != len(descriptors):
            raise ValueError("Duplicate execution input names")
        return cls(
            descriptors,
            tuple(
                {**value, "shape": tuple(value["shape"])} for value in values["outputs"]
            ),
            values["depth_layouts"],
            values["n_classes"],
        )

    def example_inputs(self, batch_size=2, device="cpu"):
        values = []
        for item in self.inputs:
            shape = tuple(
                batch_size if size == "batch" else size for size in item.shape
            )
            if item.role == "policy_control":
                value = torch.full(shape, 0.5, dtype=torch.float64, device=device)
                value[0, 0] = 0.0
            elif item.role == "metadata":
                value = torch.ones(shape, dtype=torch.bool, device=device)
                if len(shape) == 3 and batch_size > 1:
                    value[1, :, 1:] = False
                if shape[1] > 1:
                    value[0, 0] = False
            elif item.dtype == "uint8":
                value = torch.full(shape, 127, dtype=torch.uint8, device=device)
            elif item.dtype == "int64":
                value = torch.full(
                    shape,
                    3 if self.n_classes.get(item.key, 0) > 3 else 0,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                value = (
                    torch.full(shape, 0.5, device=device)
                    if len(shape) == 1
                    else torch.linspace(0.1, 0.9, steps=shape[1], device=device)
                    .reshape(1, shape[1], *([1] if len(shape) == 3 else []))
                    .expand(shape)
                    .clone()
                )
            values.append(value)
        return tuple(values)

    def bind(self, values):
        features, metadata = {}, {}
        for item, value in zip(self.inputs, values):
            destination = (
                features
                if item.role in {"feature", "static_feature", "image"}
                else metadata
            )
            destination[item.key] = value
        return features, metadata

    def validate(self, features, metadata):
        from sequifier.io.pt_payload import (
            valid_categorical_ids,
            validate_tensor_inputs,
        )

        layouts = DepthLayoutRegistryModel.model_validate(self.depth_layouts)
        selected = {
            item.key: features[item.key]
            for item in self.inputs
            if item.role == "feature"
        }
        for item in self.inputs:
            source = (
                features
                if item.role in {"feature", "static_feature", "image"}
                else metadata
            )
            value = source[item.key]
            if value.ndim != len(item.shape) or any(
                isinstance(size, int) and value.shape[axis] != size
                for axis, size in enumerate(item.shape)
            ):
                raise ValueError(
                    f"{item.key}: expected shape {item.shape}, got {tuple(value.shape)}"
                )
            if item.role == "static_feature":
                if item.dtype == "int64" and not valid_categorical_ids(
                    value, self.n_classes[item.key]
                ):
                    raise ValueError(f"{item.key}: invalid static categorical IDs")
                if item.dtype == "float32" and not torch.isfinite(value).all().item():
                    raise ValueError(f"{item.key}: static values must be finite")
            if item.role == "image" and value.dtype != torch.uint8:
                raise ValueError(f"{item.key}: image tensors must have uint8 dtype")
        if selected or layouts.root:
            validate_tensor_inputs(
                selected,
                {
                    name: metadata[depth_mask_metadata_key(name)]
                    for name in layouts.root
                },
                layouts,
                n_classes=self.n_classes,
                attention_valid_mask=metadata.get("attention_valid_mask"),
            )


def policy_graph_schema(interface, binding):
    inputs = ExecutionSchema.from_interface(interface, binding.context_length)
    names = tuple(binding.actions)
    outputs = []

    def output(name, key, dtype, *width):
        outputs.append(dict(name=name, key=key, dtype=dtype, shape=("batch", *width)))

    for index, name in enumerate(names):
        categorical = binding.actions[name].type == "categorical"
        output(f"action_{index}", name, "int64" if categorical else "float64")
    output("joint_log_probability", "joint_log_probability", "float32")
    for index, name in enumerate(names):
        field = binding.actions[name]
        if field.type == "categorical":
            output(f"probabilities_{index}", name, "float32", len(field.values))
    for role, dtype in (("latent", "float64"), ("component", "int64")):
        for index, name in enumerate(names):
            if binding.actions[name].type == "real":
                output(f"{role}_{index}", name, dtype)
    draws = InputDescriptor(
        "policy_draws",
        "policy_draws",
        "policy_control",
        "float64",
        ("batch", 1 + 2 * len(names)),
    )
    return inputs, replace(
        inputs, inputs=(*inputs.inputs, draws), outputs=tuple(outputs)
    )
