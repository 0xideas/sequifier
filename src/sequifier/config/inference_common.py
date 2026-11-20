"""Execution options consumed by every inference route."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.command_ownership import InferenceRunSettings
from sequifier.config.naming import ConfigModel
from sequifier.helpers import normalize_path
from sequifier.io.output_paths import validate_inference_destination


class InferenceExecutionConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run: InferenceRunSettings = Field(default_factory=InferenceRunSettings)
    model_paths: list[str] = Field(min_length=1)
    model_interface: str | None = None
    read_format: str = "parquet"
    write_format: str = "csv"
    output_path: str | None = None
    metadata_config_path: str | None = None
    input_layout: Literal["auto", "raw", "indexed"] = "auto"
    static_data_path: str | None = None
    mask_column: str | None = None

    if TYPE_CHECKING:
        data_path: Any
        task: Any
        rollout: Any

    @field_validator("model_paths")
    @classmethod
    def validate_paths(cls, value):
        if any(not path.strip() for path in value):
            raise ValueError("model_paths requires at least one nonempty artifact path")
        return value

    @field_validator("model_interface")
    @classmethod
    def validate_interface(cls, value):
        if value is not None and not value.isidentifier():
            raise ValueError("model_interface must be a valid identifier without '.'")
        return value

    @property
    def project_root(self):
        return self.run.project_root

    @property
    def device(self):
        return self.run.execution.device

    @property
    def seed(self):
        return self.run.reproducibility.seed

    @property
    def enforce_determinism(self):
        return self.run.reproducibility.enforce_determinism

    @property
    def inference_batch_size(self):
        return self.run.execution.batch_size

    @model_validator(mode="after")
    def validate_resources(self):
        if self.output_path is not None and len(self.model_paths) != 1:
            raise ValueError("output_path requires exactly one model path")
        rollout = self.rollout
        if rollout is not None:
            self.validate_output_destination(
                normalize_path(rollout.trajectory_path, self.project_root),
                label="trajectory_path",
            )
        elif self.output_path is not None:
            destination = (
                normalize_path(self.output_path, self.project_root)
                if self.task == "policy"
                else Path(self.project_root) / self.output_path
            )
            self.validate_output_destination(destination, label="output_path")
        return self

    def validate_output_destination(
        self, destination: str | Path, *, label: str
    ) -> None:
        inputs = self.input_paths()
        validate_inference_destination(
            destination,
            inputs,
            label=label,
        )

    def input_paths(self) -> list[str]:
        inputs = list(self.model_paths)
        inputs.extend(
            path
            for path in (
                self.data_path,
                self.metadata_config_path,
                self.static_data_path,
            )
            if path
        )
        normalized = [normalize_path(path, self.project_root) for path in inputs]
        indexed = getattr(self, "_indexed_view", None)
        if indexed is not None:
            normalized.extend(str(path) for path in indexed.dependencies())
        return normalized


def resolve_inference_model(values, *, interface_name, task):
    """Apply task-specific interface selection before resolving portable execution."""
    from sequifier.artifacts.model_config import model_execution_from_model_config

    interfaces = values.get("interfaces")
    if (
        interface_name is None
        and task != "forecast"
        and isinstance(interfaces, dict)
        and len(interfaces) > 1
        and any(
            interface.get("policy") is not None for interface in interfaces.values()
        )
    ):
        raise ValueError(
            "Select model_interface for a multi-interface artifact with a policy binding"
        )
    return model_execution_from_model_config(values, interface_name=interface_name)


def inference_model_config(path, project_root, artifacts):
    """Read the interface contract once for PT artifacts, or from ONNX metadata."""
    from sequifier.artifacts.model_artifact import read_execution_artifact

    normalized = normalize_path(path, project_root)
    if path.lower().endswith(".pt"):
        if normalized not in artifacts:
            artifacts[normalized] = artifacts.get(path) or read_execution_artifact(
                normalized
            )
        artifacts[path] = artifacts[normalized]
        return artifacts[path].model_config.values
    if not path.lower().endswith(".onnx"):
        return None
    import onnx

    from sequifier.model.execution_schema import (
        DROPOUT_MODE_KEY,
        EXECUTION_SCHEMA_KEY,
        MODEL_CONFIG_KEY,
        ExecutionSchema,
    )

    graph = onnx.load(normalized, load_external_data=False)
    properties = {item.key: item.value for item in graph.metadata_props}
    schema = properties.get(EXECUTION_SCHEMA_KEY)
    if (
        schema is not None
        or MODEL_CONFIG_KEY in properties
        or DROPOUT_MODE_KEY in properties
        or any(
            len(value.type.tensor_type.shape.dim) == 3 for value in graph.graph.input
        )
    ):
        ExecutionSchema.from_dict(json.loads(schema) if schema is not None else None)
    return (
        json.loads(properties[MODEL_CONFIG_KEY])
        if MODEL_CONFIG_KEY in properties
        else None
    )
