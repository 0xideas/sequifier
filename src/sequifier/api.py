"""Stable sibling-facing, model-level Sequifier API."""

from sequifier.artifacts.model_artifact import (
    ModelArtifact,
    ModelArtifactMetadata,
    ModelExecutionConfig,
    load_model_artifact,
    load_weights_from_run_checkpoint,
)
from sequifier.artifacts.state_dict import (
    canonical_parameter_name,
    canonicalize_state_dict,
)
from sequifier.model.factory import build_transformer_network
from sequifier.model.network import (
    ComposableTransformerNetwork,
    DecodeRequest,
    EncodedOutput,
    EncodeRequest,
    ModelInterfaceModule,
    ModelOutput,
    TracedModelOutput,
)
from sequifier.model.parameter_catalog import ParameterCatalog, ParameterDescriptor
from sequifier.model.tracing import (
    CaptureRequest,
    Intervention,
    InterventionBinding,
    TraceSite,
)

__all__ = [
    "CaptureRequest",
    "ComposableTransformerNetwork",
    "DecodeRequest",
    "EncodedOutput",
    "EncodeRequest",
    "Intervention",
    "InterventionBinding",
    "ModelArtifact",
    "ModelArtifactMetadata",
    "ModelExecutionConfig",
    "ModelInterfaceModule",
    "ModelOutput",
    "ParameterCatalog",
    "ParameterDescriptor",
    "TraceSite",
    "TracedModelOutput",
    "build_transformer_network",
    "canonical_parameter_name",
    "canonicalize_state_dict",
    "load_model_artifact",
    "load_weights_from_run_checkpoint",
]
