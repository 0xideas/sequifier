"""Stable model and artifact API for Sequifier integrations.

Only the names re-exported here are part of the integration contract. They
cover construction, tracing, parameter inspection, artifact loading, and
training controls without exposing training-command implementation details.
"""

from sequifier.artifacts.loading import (
    ExecutionOptions,
    LoadedModel,
    load_model_for_analysis,
    normalize_model_state_dict,
)
from sequifier.artifacts.model_export import model_execution_config, pt_bundle
from sequifier.artifacts.run_checkpoint import RunCheckpointStore
from sequifier.config.train_config import (
    BackboneComponentConfig,
    DecoderComponentConfig,
    FeatureLayoutRegistryModel,
    IngestionComponentConfig,
    ResolvedModelInterface,
    ResolvedSequifierConfig,
    SelectedDatasetPartConfig,
    SelectedInterfaceConfig,
)
from sequifier.integration.contexts import StepIdentity
from sequifier.integration.controls import TrainingDirective
from sequifier.model.factory import BuiltModel, build_transformer_network
from sequifier.model.network import (
    ComposableTransformerNetwork,
    DecodeRequest,
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
    trace_sites,
)

__all__ = [
    "BackboneComponentConfig",
    "BuiltModel",
    "CaptureRequest",
    "ComposableTransformerNetwork",
    "DecodeRequest",
    "DecoderComponentConfig",
    "ExecutionOptions",
    "FeatureLayoutRegistryModel",
    "IngestionComponentConfig",
    "Intervention",
    "InterventionBinding",
    "LoadedModel",
    "ModelInterfaceModule",
    "ModelOutput",
    "ParameterCatalog",
    "ParameterDescriptor",
    "ResolvedModelInterface",
    "ResolvedSequifierConfig",
    "SelectedDatasetPartConfig",
    "SelectedInterfaceConfig",
    "StepIdentity",
    "RunCheckpointStore",
    "TraceSite",
    "TracedModelOutput",
    "TrainingDirective",
    "build_transformer_network",
    "load_model_for_analysis",
    "model_execution_config",
    "normalize_model_state_dict",
    "pt_bundle",
    "trace_sites",
]
