"""Runner and publication capabilities checked when compiling a training plan."""

from typing import Literal

from sequifier.config.ownership import RunSettings

RunMode = Literal["supervised", "policy", "coordinated"]


def validate_runner_settings(
    run: RunSettings, *, supervised: bool, policy: bool
) -> None:
    execution = run.execution
    if not policy and execution.policy is not None:
        raise ValueError("run.execution.policy requires a policy phase")
    if not supervised and execution.supervised is not None:
        raise ValueError("run.execution.supervised requires a supervised phase")
    if policy and execution.precision.layer_type_dtypes:
        raise ValueError(
            "Plans with policy phases do not support manual layer pre-casting"
        )
    if not supervised and run.logging.log_interval_batches is not None:
        raise ValueError("Batch logging requires a supervised phase")
    if not policy and run.logging.log_interval_iterations is not None:
        raise ValueError("Iteration logging requires a policy phase")
    export = run.export
    if policy and (export.embedding or not export.prediction):
        raise ValueError("Policy plans publish a portable prediction artifact")
    if (
        policy
        and export.prediction
        and export.prediction.onnx
        and export.prediction.onnx.with_dropout
    ):
        raise ValueError(
            "Policy ONNX export uses evaluation mode; stochastic actions use policy draws"
        )
    if not (export.prediction or export.embedding):
        raise ValueError("Select at least one export artifact")


def training_run_mode(phases) -> RunMode:
    policy = [phase for phase in phases if phase.algorithm != "supervised"]
    if not policy:
        return "supervised"
    if len(policy) == len(phases) and len({p.model_interface for p in policy}) == 1:
        return "policy"
    return "coordinated"
