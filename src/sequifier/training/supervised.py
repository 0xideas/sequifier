"""Launch supervised workers and execute a native supervised run."""

from sequifier.helpers import configure_determinism, configure_logger
from sequifier.integration import IntegrationManager
from sequifier.runtime.builder import RunBuilder
from sequifier.training.engine import TrainingEngine
from sequifier.training.launch import launch_training as launch_supervised
from sequifier.training.launch import train_worker
from sequifier.training.runtime import PreparedSupervisedGroup

__all__ = ["execute_supervised", "launch_supervised", "train_worker"]


def execute_supervised(
    group,
    execution,
    *,
    integration_specs=(),
    integration_instances=(),
    semantic_optimizer_grouping=False,
    log_model_name=None,
    **build_options,
):
    """Build and finalize one native runtime in the supplied worker environment."""
    if isinstance(group, PreparedSupervisedGroup):
        config, dataset_names = group.config, ()
    else:
        config, dataset_names = group, tuple(group.dataset_training)
    settings = config.run.execution.supervised_settings
    logger = configure_logger(
        config.project_root,
        log_model_name or config.model_name,
        execution.rank,
        dataset_names=dataset_names,
        rank_specific=execution.distributed,
    )
    configure_determinism(config.seed, config.run.reproducibility.enforce_determinism)
    integrations = IntegrationManager(
        specs=integration_specs,
        instances=integration_instances,
        rank=execution.rank,
        world_size=execution.world_size,
        distributed=execution.distributed,
    )
    integrations.validate_execution(
        torch_compile=settings.compilation.torch_compile,
        data_parallelism=settings.distribution.strategy,
    )
    run = RunBuilder(
        semantic_optimizer_grouping=semantic_optimizer_grouping
    ).build_if_needed(
        group, execution, integrations, log_model_name=log_model_name, **build_options
    )
    if run is None:
        return
    logger.info(
        f"Starting training for {run.context.model_name}, run {run.state.run_id}"
    )
    result = TrainingEngine().run(run)
    logger.info(f"Training complete ({result.completion_reason})")
