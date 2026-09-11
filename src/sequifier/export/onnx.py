from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from sequifier.model.execution_schema import (
    DROPOUT_MODE_KEY,
    EXECUTION_SCHEMA_KEY,
    ExecutionSchema,
)
from sequifier.model.layers import SelfAttention
from sequifier.runtime.random_state import RandomStateManager


class _OnnxWrapper(nn.Module):
    def __init__(self, model, schema, interface_name):
        super().__init__()
        self.model = model
        self.schema = schema
        self.interface_name = interface_name

    def forward(self, *values: Tensor):
        features, metadata = self.schema.bind(values)
        if self.interface_name is None:
            return self.model(features, metadata)
        route = self.model.resolve_interface(self.interface_name)
        representation = self.model.encode(
            features, metadata, interface_name=self.interface_name
        )
        logits = route.decode(representation)
        prediction_positions = slice(-route.prediction_length, None)
        return tuple(
            (
                torch.log_softmax(
                    logits[item["key"]][:, prediction_positions].float(),
                    dim=-1,
                )
                if route.target_column_types[item["key"]] == "categorical"
                else logits[item["key"]][:, prediction_positions].float()
            ).transpose(0, 1)
            for item in self.schema.outputs
        )


def validate_graph_contract(graph, schema):
    expected = {item.name: item for item in schema.inputs}
    for value in graph.graph.input:
        if value.name not in expected:
            raise ValueError(f"Unexpected ONNX graph input {value.name!r}")
        descriptor = expected[value.name]
        tensor = value.type.tensor_type
        expected_dtype = {
            "float32": onnx.TensorProto.FLOAT,
            "int64": onnx.TensorProto.INT64,
            "bool": onnx.TensorProto.BOOL,
        }[descriptor.dtype]
        if tensor.elem_type != expected_dtype or len(tensor.shape.dim) != len(
            descriptor.shape
        ):
            raise ValueError(
                f"ONNX graph input {value.name!r} contradicts execution dtype/rank"
            )
        for dimension, wanted in zip(tensor.shape.dim, descriptor.shape):
            if wanted == "batch":
                if not dimension.dim_param:
                    raise ValueError(
                        f"ONNX graph input {value.name!r} lost symbolic batch"
                    )
            elif dimension.dim_value != wanted:
                raise ValueError(
                    f"ONNX graph input {value.name!r} contradicts fixed capacity"
                )
    actual = {value.name for value in graph.graph.input}
    # Layout masks with consumed deep features are semantically required.
    for item in schema.inputs:
        if item.key.startswith("depth_valid_mask:") and item.name not in actual:
            layout = schema.depth_layouts[item.key.split(":", 1)[1]]
            feature_names = {
                d.name
                for d in schema.inputs
                if d.role == "feature" and d.key in layout["columns"]
            }
            if actual.intersection(feature_names):
                raise ValueError(
                    f"Required depth mask {item.key!r} was pruned during export"
                )
    if [value.name for value in graph.graph.output] != [
        item["name"] for item in schema.outputs
    ]:
        raise ValueError("ONNX output order contradicts execution schema")
    for value, descriptor in zip(graph.graph.output, schema.outputs):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Portable ONNX outputs must be FP32")
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 3:
            raise ValueError("ONNX outputs must be sequence-major rank three")
        for axis, size in enumerate(descriptor["shape"]):
            if (
                isinstance(size, int)
                and dims[axis].HasField("dim_value")
                and dims[axis].dim_value != size
            ):
                raise ValueError("ONNX output dimensions contradict execution schema")


def dropout_graph_inventory(graph):
    """Expand local function calls while retaining each call site's provenance."""
    result = []
    functions = {
        (function.domain, function.name): function for function in graph.functions
    }
    constants = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in graph.graph.initializer
        if tensor.data_location != onnx.TensorProto.EXTERNAL
    }

    def visit(nodes, prefix, inherited_scope, environment, active_functions=()):
        environment = dict(environment)
        for node in nodes:
            scope = {
                **inherited_scope,
                **{entry.key: entry.value for entry in node.metadata_props},
            }
            if node.op_type == "Constant":
                for attribute in node.attribute:
                    if attribute.name == "value":
                        environment[node.output[0]] = onnx.numpy_helper.to_array(
                            attribute.t
                        )
                    elif attribute.name in {"value_int", "value_float"}:
                        environment[node.output[0]] = (
                            attribute.i
                            if attribute.name == "value_int"
                            else attribute.f
                        )
            elif (
                node.op_type in {"Cast", "CastLike", "Identity"}
                and node.input[0] in environment
            ):
                environment[node.output[0]] = environment[node.input[0]]
            if node.op_type == "Dropout":
                if len(node.input) < 3 or node.input[2] not in environment:
                    raise ValueError(
                        f"Dropout node {node.name!r} lacks an attributable fixed training-mode operand"
                    )
                if not bool(np.asarray(environment[node.input[2]]).item()):
                    raise ValueError(
                        f"Stochastic export disabled dropout at {node.name!r}"
                    )
                if node.input[1] not in environment:
                    raise ValueError(
                        f"Dropout ratio is not statically attributable at {node.name!r}"
                    )
                result.append(
                    {
                        "node": prefix + node.name,
                        "inputs": list(node.input),
                        "ratio": float(np.asarray(environment[node.input[1]]).item()),
                        "scope": scope,
                    }
                )
            function_key = (node.domain, node.op_type)
            if function_key in functions:
                if function_key in active_functions:
                    raise ValueError("Recursive ONNX local functions are unsupported")
                function = functions[function_key]
                arguments = {
                    formal: environment[actual]
                    for formal, actual in zip(function.input, node.input)
                    if actual in environment
                }
                visit(
                    function.node,
                    prefix + node.name + "/",
                    scope,
                    arguments,
                    active_functions + (function_key,),
                )
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(
                        attribute.g.node,
                        prefix + node.name + "/",
                        scope,
                        environment,
                        active_functions,
                    )

    visit(graph.graph.node, "", {}, constants)
    return result


def _expected_dropout_sites(wrapper, inputs):
    """Observe execution sites on the disposable export reference, including aliases."""
    names = {id(module): name for name, module in wrapper.named_modules()}
    aliases = {
        name: names[id(module)]
        for name, module in wrapper.named_modules(remove_duplicate=False)
    }
    counts = Counter()
    handles = []
    try:
        for name, module in wrapper.named_modules():
            if isinstance(module, nn.Dropout) and module.p > 0 and module.training:
                handles.append(
                    module.register_forward_hook(
                        lambda module, args, output, name=name: counts.update([name])
                    )
                )
        with torch.no_grad():
            wrapper(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    probabilities = {
        name: module.p for name, module in wrapper.named_modules() if name in counts
    }
    return counts, aliases, probabilities


def _validate_dropout_sites(graph, expected, aliases, probabilities):
    actual = Counter()
    inventory = dropout_graph_inventory(graph)
    for item in inventory:
        scopes = item["scope"].get("pkg.torch.onnx.name_scopes")
        if scopes is None:
            continue
        names = ast.literal_eval(scopes)
        for name in reversed(names):
            canonical = aliases.get(name)
            if canonical in expected:
                if not np.isclose(
                    item["ratio"], probabilities[canonical], rtol=1e-6, atol=1e-8
                ):
                    raise ValueError(
                        f"Exporter changed dropout probability at {canonical!r}"
                    )
                actual[canonical] += 1
                item["module"] = canonical
                break
    missing = {
        name: {"expected": count, "retained": actual[name]}
        for name, count in expected.items()
        if actual[name] != count
    }
    if missing:
        raise ValueError(
            f"Exporter did not retain individually attributable dropout sites: {missing}"
        )
    return inventory


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
        import re
        from importlib.metadata import version

        minimums = {
            "torch": (2, 6),
            "onnx": (1, 17),
            "onnxscript": (0, 5, 4),
            "onnxruntime": (1, 20),
        }
        for package, minimum in minimums.items():
            installed_version = version(package)
            match = re.match(r"[0-9.]+", installed_version)
            if match is None:
                raise RuntimeError(
                    f"Cannot determine ONNX export compatibility for {package}: "
                    f"unrecognized version {installed_version!r}"
                )
            found = tuple(int(part) for part in match.group().rstrip(".").split("."))
            if found < minimum:
                raise RuntimeError(
                    f"ONNX export requires {package}>={'.'.join(map(str, minimum))}; installed {installed_version}"
                )
        manager = RandomStateManager(torch.device("cpu"))
        state = manager.capture_local()
        try:
            # All graph lowering and dtype conversion happen on a disposable copy.
            model = copy.deepcopy(model).cpu().float()
            for module in model.modules():
                if isinstance(module, SelfAttention):
                    module._sequifier_export_attention = True
            schema = ExecutionSchema.from_interface(
                interface, context_length, embedding=interface_name is None
            )
            inputs = schema.example_inputs(min(4, max(2, batch_size)))
            schema.validate(*schema.bind(inputs))
            wrapper = _OnnxWrapper(model, schema, interface_name).train(training)
            expected_dropout, aliases, probabilities = (
                _expected_dropout_sites(wrapper, inputs) if training else ({}, {}, {})
            )
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".onnx-staging-", dir=destination.parent
            ) as directory:
                staged = Path(directory) / destination.name
                batch = torch.export.Dim("batch", min=1)
                torch.onnx.export(
                    wrapper,
                    inputs,
                    staged,
                    export_params=True,
                    opset_version=18,
                    dynamo=True,
                    external_data=True,
                    optimize=False,
                    dynamic_shapes={"values": tuple({0: batch} for _ in inputs)},
                    input_names=[item.name for item in schema.inputs],
                    output_names=[item["name"] for item in schema.outputs],
                )
                graph = onnx.load(staged, load_external_data=False)
                # Materialize resolved output widths in the portable metadata.
                for value, descriptor in zip(graph.graph.output, schema.outputs):
                    dims = value.type.tensor_type.shape.dim
                    if (
                        descriptor["shape"][-1] is None
                        and len(dims) == 3
                        and dims[2].HasField("dim_value")
                    ):
                        descriptor["shape"] = (
                            *descriptor["shape"][:2],
                            dims[2].dim_value,
                        )
                validate_graph_contract(graph, schema)
                dropout_sites = (
                    _validate_dropout_sites(
                        graph, expected_dropout, aliases, probabilities
                    )
                    if training
                    else []
                )
                combined_metadata = {
                    **(metadata or {}),
                    EXECUTION_SCHEMA_KEY: json.dumps(schema.to_dict()),
                    DROPOUT_MODE_KEY: "stochastic" if training else "evaluation",
                    "sequifier.dropout_graph_inventory": json.dumps(dropout_sites),
                }
                for key, value in combined_metadata.items():
                    entry = graph.metadata_props.add()
                    entry.key, entry.value = key, value
                external_files = {}
                publication_id = uuid.uuid4().hex
                for tensor in graph.graph.initializer:
                    if tensor.data_location != onnx.TensorProto.EXTERNAL:
                        continue
                    for item in tensor.external_data:
                        if item.key != "location":
                            continue
                        original = item.value
                        if original not in external_files:
                            source = (Path(directory) / original).resolve()
                            if not source.is_relative_to(Path(directory).resolve()):
                                raise ValueError(
                                    "ONNX exporter emitted external data outside staging"
                                )
                            unique = f"{destination.name}.{publication_id}.{len(external_files)}.data"
                            target = Path(directory) / unique
                            os.replace(source, target)
                            external_files[original] = target
                        item.value = external_files[original].name
                onnx.save(graph, staged)
                onnx.checker.check_model(str(staged), full_check=True)
                options = ort.SessionOptions()
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    if training
                    else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session = ort.InferenceSession(
                    str(staged),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                actual_names = {item.name for item in session.get_inputs()}
                feeds = {
                    item.name: value.numpy()
                    for item, value in zip(schema.inputs, inputs)
                    if item.name in actual_names
                }
                outputs = session.run([item["name"] for item in schema.outputs], feeds)
                if any(not np.isfinite(value).all() for value in outputs):
                    raise ValueError(
                        "ONNX artifact produced non-finite outputs during publication validation"
                    )
                if not training:
                    with torch.no_grad():
                        reference = wrapper(*inputs)
                    reference = (
                        (reference,) if isinstance(reference, Tensor) else reference
                    )
                    for expected, actual in zip(reference, outputs):
                        if not np.allclose(
                            expected.detach().numpy(), actual, rtol=2e-4, atol=2e-5
                        ):
                            raise ValueError(
                                "ONNX artifact differs from its FP32 export reference"
                            )
                published = []
                try:
                    for external_file in external_files.values():
                        final_file = destination.parent / external_file.name
                        os.replace(external_file, final_file)
                        published.append(final_file)
                    # Unique sidecars are published first; the graph is the atomic
                    # commit point, leaving an existing artifact usable on failure.
                    os.replace(staged, destination)
                except BaseException:
                    for final_file in published:
                        final_file.unlink(missing_ok=True)
                    raise
            return destination
        except Exception as error:
            raise RuntimeError(
                f"ONNX export failed for interface {interface_name or interface.name!r} at {destination}: {error}"
            ) from error
        finally:
            manager.restore(state)
