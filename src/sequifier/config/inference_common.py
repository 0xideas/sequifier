"""Execution options consumed by every inference route."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from sequifier.config.command_ownership import CommandRunSettings
from sequifier.config.naming import ConfigModel
from sequifier.helpers import normalize_path
from sequifier.io.output_paths import validate_inference_destination


class InferenceExecutionConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    model_interface: str | None = None
    read_format: str = "parquet"
    write_format: str = "csv"
    output_path: str | None = None

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
        self.run.validate_for("infer")
        if self.output_path is not None and len(getattr(self, "model_paths", ())) != 1:
            raise ValueError("output_path requires exactly one model path")
        rollout = getattr(self, "rollout", None)
        if rollout is not None:
            self.validate_output_destination(
                normalize_path(rollout.trajectory_path, self.project_root),
                label="trajectory_path",
            )
        elif self.output_path is not None:
            destination = (
                normalize_path(self.output_path, self.project_root)
                if getattr(self, "task", None) == "policy"
                else Path(self.project_root) / self.output_path
            )
            self.validate_output_destination(destination, label="output_path")
        return self

    def validate_output_destination(
        self, destination: str | Path, *, label: str
    ) -> None:
        inputs = list(getattr(self, "model_paths", ()))
        for field in ("data_path", "metadata_config_path", "static_features_path"):
            if path := getattr(self, field, None):
                inputs.append(path)
        inputs = [normalize_path(path, self.project_root) for path in inputs]
        if training_config_path := getattr(self, "training_config_path", None):
            inputs.append(training_config_path)
        validate_inference_destination(
            destination,
            inputs,
            label=label,
        )


class InferenceRouteConfig(ConfigModel):
    """Validate artifact and interface selection before route-specific options."""

    model_config = ConfigDict(extra="ignore")
    run: CommandRunSettings = Field(default_factory=CommandRunSettings)
    model_paths: list[str] = Field(min_length=1)
    model_interface: str | None = None
    task: Literal["forecast", "policy", "embedding"]

    @property
    def project_root(self):
        return self.run.project_root

    @field_validator("model_paths")
    @classmethod
    def validate_paths(cls, value):
        paths = value
        if not paths or any(not path.strip() for path in paths):
            raise ValueError("model_paths requires at least one nonempty artifact path")
        return value

    @field_validator("model_interface")
    @classmethod
    def validate_interface(cls, value):
        if value is not None and not value.isidentifier():
            raise ValueError("model_interface must be a valid identifier without '.'")
        return value


@dataclass(frozen=True)
class SelectedInferenceModel:
    path: str
    values: dict
    interface_name: str

    @classmethod
    def select(cls, path, values, *, interface_name=None, task=None):
        interfaces = values.get("interfaces")
        if not isinstance(interfaces, dict) or not interfaces:
            raise ValueError("model_config.interfaces must be a non-empty mapping")
        if interface_name is None:
            policies = [
                name
                for name, interface in interfaces.items()
                if interface.get("policy") is not None
            ]
            if len(interfaces) > 1 and policies and task != "forecast":
                raise ValueError(
                    "Artifact has multiple interfaces; select model_interface before choosing task"
                )
            else:
                interface_name = next(iter(interfaces))
        if interface_name not in interfaces:
            raise ValueError(f"Unknown model_interface {interface_name!r}")
        return cls(path, values, interface_name)

    @property
    def policy_bound(self):
        return self.values["interfaces"][self.interface_name].get("policy") is not None

    def resolve(self, device):
        from sequifier.artifacts.model_config import model_execution_from_model_config

        return model_execution_from_model_config(
            self.values, device=device, interface_name=self.interface_name
        )[0]


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
