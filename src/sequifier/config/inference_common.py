"""Execution options consumed by every inference route."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.command_ownership import InferenceRunSettings
from sequifier.config.naming import ConfigModel
from sequifier.helpers import normalize_path
from sequifier.io.output_paths import validate_inference_destination

if TYPE_CHECKING:
    from sequifier.artifacts.model_artifact import ModelArtifact
    from sequifier.model.execution import ModelExecution
    from sequifier.model.execution_schema import ExecutionSchema


class InferenceExecutionConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run: InferenceRunSettings = Field(default_factory=InferenceRunSettings)
    model_paths: list[str] = Field(min_length=1)
    model_interface: str | None = None
    read_format: str = "parquet"
    write_format: str = "csv"
    output_path: str | None = None
    metadata_config_path: str | None = None
    split: str | None = None
    input_layout: Literal["auto", "raw", "indexed"] = "auto"
    static_data_path: str | None = None
    mask_column: str | None = None

    if TYPE_CHECKING:
        data_path: Any
        task: Any
        rollout: Any

    @field_validator("read_format")
    @classmethod
    def validate_read_format(cls, v: str) -> str:
        if v not in ["csv", "parquet", "pt"]:
            raise ValueError(
                "Currently only 'csv', 'parquet' and 'pt' are supported for "
                "inference input"
            )
        return v

    @field_validator("write_format")
    @classmethod
    def validate_write_format(cls, v: str) -> str:
        if v not in ["csv", "parquet"]:
            raise ValueError(
                "Currently only 'csv' and 'parquet' are supported for inference output"
            )
        return v

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
            paths = [rollout.trajectory_path]
            if rollout.output:
                manifest = rollout.output.manifest_path
                paths.extend([manifest, manifest + ".parts"])
            from sequifier.io.output_paths import validate_path_conflicts

            destinations = [normalize_path(p, self.project_root) for p in paths if p]
            validate_path_conflicts(destinations, (), label="rollout output")
            for destination in destinations:
                self.validate_output_destination(destination, label="rollout output")
        elif self.output_path is not None:
            destination = Path(self.project_root) / self.output_path
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


@dataclass(frozen=True)
class ResolvedInferenceModel:
    path: str
    execution: ModelExecution | None
    interface_name: str | None
    artifact: ModelArtifact | None = None
    onnx_schema: ExecutionSchema | None = None

    @classmethod
    def load(cls, path, *, interface_name, task):
        from sequifier.artifacts.model_artifact import read_execution_artifact

        artifact = schema = None
        if path.lower().endswith(".pt"):
            artifact = read_execution_artifact(path)
            values = artifact.model_config.values
        elif path.lower().endswith(".onnx"):
            import onnx

            from sequifier.export.onnx import validate_graph_contract
            from sequifier.model.execution_schema import (
                EXECUTION_SCHEMA_KEY,
                MODEL_CONFIG_KEY,
                ExecutionSchema,
            )

            graph = onnx.load(path, load_external_data=False)
            properties = {item.key: item.value for item in graph.metadata_props}
            schema = ExecutionSchema.from_dict(
                json.loads(properties.get(EXECUTION_SCHEMA_KEY, "null"))
            )
            validate_graph_contract(graph, schema)
            values = (
                json.loads(properties[MODEL_CONFIG_KEY])
                if MODEL_CONFIG_KEY in properties
                else None
            )
        else:
            raise ValueError(f"Unsupported inference model format: {path}")
        execution = None
        if values is not None:
            execution, interface_name = resolve_inference_model(
                values, interface_name=interface_name, task=task
            )
        return cls(path, execution, interface_name, artifact, schema)
