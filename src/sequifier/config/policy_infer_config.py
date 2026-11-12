"""Inference options for policy interfaces in ordinary portable model artifacts."""

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from sequifier.artifacts.model_artifact import ModelArtifact
from sequifier.config.inference_common import (
    InferenceExecutionConfig,
    InferenceRouteConfig,
    SelectedInferenceModel,
    inference_model_config,
)
from sequifier.config.policy_config import PolicyConfigBase
from sequifier.config.policy_plan import OnlineEnvironmentConfig
from sequifier.helpers import normalize_path


class PolicyRolloutConfig(PolicyConfigBase):
    environment: OnlineEnvironmentConfig
    max_steps: int = Field(gt=0)
    trajectory_path: str = Field(min_length=1)
    seed_row_index: int = Field(ge=0)
    max_episode_steps: int | None = Field(default=None, gt=0)


class PolicyInferenceConfig(InferenceExecutionConfig):
    model_paths: list[str] = Field(min_length=1, max_length=1)
    task: Literal["policy"]

    data_path: str
    output_path: str | None = None
    output_probabilities: bool = False
    action_selection: Literal["deterministic", "stochastic"] = "deterministic"
    desired_return: float | None = Field(default=None, allow_inf_nan=False)
    desired_return_column: str = "desired_return"
    previous_action_columns: dict[str, str] = Field(default_factory=dict)
    sequence_id_column: str = "sequenceId"
    item_position_column: str = "itemPosition"
    rollout: PolicyRolloutConfig | None = None
    # Optional assertions against the portable artifact contract.
    metadata_config_path: str | None = None
    input_columns: list[str] | None = None
    static_columns: list[str] | None = None
    target_columns: list[str] | None = None
    column_data_types: dict[str, str] | None = None
    target_column_types: dict[str, str] | None = None
    training_objective: str | None = None
    context_length: int | None = Field(default=None, gt=0)
    target_offset: int | None = Field(default=None, ge=0)
    prediction_length: int | None = Field(default=None, gt=0)
    decoder_support_length: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_execution(self):
        if self.read_format not in {"csv", "parquet"} or self.write_format not in {
            "csv",
            "parquet",
        }:
            raise ValueError("policy inference formats must be csv or parquet")
        if self.rollout is not None and self.output_probabilities:
            raise ValueError(
                "rollout writes a trajectory including action log probabilities"
            )
        return self


class ResolvedPolicyInferenceConfig(PolicyInferenceConfig):
    """Validated policy request with selected artifact/interface and assertions."""

    _artifacts: dict[str, ModelArtifact] = PrivateAttr(default_factory=dict)


def _policy_inference_route(
    route: InferenceRouteConfig, artifacts: dict[str, ModelArtifact]
) -> SelectedInferenceModel | None:
    """Inspect artifact interfaces before applying either inference schema."""
    if route.task != "policy":
        return None
    paths = route.model_paths
    bindings = []
    selection = None
    for path in paths:
        model_config = inference_model_config(path, route.project_root, artifacts)
        if model_config is None:
            bindings.append(False)
            continue
        selection = SelectedInferenceModel.select(
            path,
            model_config,
            interface_name=route.model_interface,
            task=route.task,
        )
        bindings.append(selection.policy_bound)
    if len(paths) != 1 or not all(bindings):
        raise ValueError(
            "task: policy requires one artifact with a selected policy-bound interface"
        )
    if not paths[0].lower().endswith(".pt"):
        raise ValueError(
            "task: policy requires a portable .pt artifact; "
            "use task: forecast for forecast inference"
        )
    assert selection is not None
    return selection


def policy_inference_config(
    values: dict, artifacts: dict[str, ModelArtifact] | None = None
) -> ResolvedPolicyInferenceConfig | None:
    """Dispatch from the persisted interface contract, without training YAML."""
    route = InferenceRouteConfig.model_validate(values)
    from sequifier.config.infer_config import (
        _assert_metadata_matches_source,
        _execution_source,
    )
    from sequifier.config.metadata import load_dataset_metadata

    artifacts = {} if artifacts is None else artifacts
    selection = _policy_inference_route(route, artifacts)
    if selection is None:
        return None
    model_path, selected = selection.path, selection.interface_name
    request = PolicyInferenceConfig.model_validate(values)
    execution = selection.resolve(request.device)
    policy = execution.interface(selected).policy
    if policy is None:
        raise ValueError("selected interface has no policy")
    if (
        request.rollout is not None
        and request.rollout.environment.units != policy.units
    ):
        raise ValueError("rollout environment units differ from policy artifact")
    contract, metadata = _execution_source(execution, execution.dataset(selected))
    for key, actual in contract.items():
        if (
            key in request.model_fields_set
            and getattr(request, key) is not None
            and getattr(request, key) != actual
        ):
            raise ValueError(f"{key} differs from policy artifact")
    if request.metadata_config_path:
        explicit = load_dataset_metadata(
            normalize_path(request.metadata_config_path, request.project_root)
        )
        _assert_metadata_matches_source(
            explicit,
            metadata,
            "policy metadata",
            "storage_layout"
            in artifacts[model_path].model_config.values["interfaces"][selected],
        )
    authored = request.model_dump()
    authored["run"] = request.run.model_dump(mode="python", exclude_unset=True)
    authored["model_paths"] = [model_path]
    authored["model_interface"] = selected
    config = ResolvedPolicyInferenceConfig.model_validate(authored)
    config._artifacts = artifacts
    return config
