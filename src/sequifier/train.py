"""Training command composition and PT inference entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from loguru import logger as loguru_logger  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from sequifier.artifacts.model_artifact import (  # noqa: E402
    load_model_artifact,
    load_weights_from_run_checkpoint,
)
from sequifier.config.train_config import (  # noqa: E402
    ResolvedSequifierConfig as TrainModel,
)
from sequifier.config.train_config import load_train_config  # noqa: E402
from sequifier.distributed.env import setup_distributed_env  # noqa: E402
from sequifier.helpers import (  # noqa: E402
    configure_determinism,
    configure_logger,
    get_torch_dtype,
)
from sequifier.integration import IntegrationManager, IntegrationSpec  # noqa: E402
from sequifier.model.embedding import embedding_layer_trace_site  # noqa: E402
from sequifier.model.tracing import CaptureRequest  # noqa: E402
from sequifier.runtime.builder import RunBuilder  # noqa: E402
from sequifier.runtime.context import ExecutionEnvironment  # noqa: E402
from sequifier.training.engine import TrainingEngine  # noqa: E402
from sequifier.typechecking import beartype  # noqa: E402


@dataclass
class LoadedInferenceModel:
    """Execution metadata alongside the sole weight-owning network."""

    network: Any
    config: Any
    interface_name: str
    embedding: bool

    @property
    def interface(self) -> Any:
        return next(
            dataset.interface
            for dataset in self.config.dataset_training.values()
            if dataset.model_interface == self.interface_name
        )

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.interface.categorical_columns)

    @property
    def target_decoder_ids(self) -> dict[str, list[int]]:
        return dict(self.interface.target_decoder_ids)

    @property
    def hparams(self) -> Any:
        return self.config


def _execution_device(config: TrainModel, local_rank: int) -> torch.device:
    configured = torch.device(config.device)
    if configured.type == "cuda":
        return torch.device("cuda", local_rank)
    return configured


@beartype
def train_worker(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    global_rank: int,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
) -> None:
    """Initialize one execution environment and run the composed runtime."""

    configure_logger(
        config.project_root,
        config.model_name,
        global_rank,
        dataset_names=tuple(config.dataset_training),
        rank_specific=config.global_training.distributed,
    )
    initialized_distributed = False
    if config.global_training.distributed:
        if config.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        setup_distributed_env(
            global_rank,
            local_rank,
            world_size,
            config.global_training.backend,
        )
        initialized_distributed = True
    configure_determinism(config.seed, config.global_training.enforce_determinism)
    execution = ExecutionEnvironment(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=_execution_device(config, local_rank),
        distributed=config.global_training.distributed,
    )
    integrations = IntegrationManager(
        specs=integration_specs,
        instances=integration_instances,
        rank=global_rank,
        world_size=world_size,
        distributed=config.global_training.distributed,
    )
    integrations.validate_execution(
        torch_compile=config.global_training.torch_compile,
        data_parallelism=config.global_training.data_parallelism,
    )
    run = None
    try:
        run = RunBuilder(semantic_optimizer_grouping=semantic_optimizer_grouping).build(
            config, execution, integrations
        )
        loguru_logger.info(
            f"--- Starting Training for model: {run.context.model_name} | "
            f"run: {run.state.run_id} | session: {run.state.session_id} ---"
        )
        result = TrainingEngine().run(run)
        loguru_logger.info(f"--- Training Complete ({result.completion_reason}) ---")
    finally:
        if run is not None:
            run.distributed.finalize()
        elif initialized_distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@beartype
def _mp_train_worker_wrapper(
    local_rank: int,
    world_size: int,
    config: TrainModel,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    semantic_optimizer_grouping: bool = False,
) -> None:
    train_worker(
        local_rank,
        world_size,
        config,
        global_rank=local_rank,
        integration_specs=integration_specs,
        semantic_optimizer_grouping=semantic_optimizer_grouping,
    )


@beartype
def run_training(
    config: TrainModel,
    *,
    integration_specs: tuple[IntegrationSpec, ...] = (),
    integration_instances: tuple[Any, ...] = (),
    semantic_optimizer_grouping: bool = False,
) -> None:
    """Launch canonical service-composed training locally or across workers."""

    if not isinstance(config, TrainModel):
        raise TypeError("Training requires a canonical resolved config.")
    spec = config.global_training
    if spec.distributed and integration_instances:
        raise ValueError(
            "Distributed runs require IntegrationSpec; direct instances cannot "
            "be transferred to workers."
        )
    torch.set_float32_matmul_precision(spec.float32_matmul_precision)
    if spec.distributed and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        train_worker(
            int(os.environ.get("LOCAL_RANK", 0)),
            int(os.environ["WORLD_SIZE"]),
            config,
            int(os.environ["RANK"]),
            integration_specs,
            (),
            semantic_optimizer_grouping,
        )
    elif spec.distributed:
        mp.spawn(
            _mp_train_worker_wrapper,
            args=(
                spec.world_size,
                config,
                integration_specs,
                semantic_optimizer_grouping,
            ),
            nprocs=spec.world_size,
            join=True,
        )
    else:
        train_worker(
            0,
            1,
            config,
            0,
            integration_specs,
            integration_instances,
            semantic_optimizer_grouping,
        )


@beartype
def train(args: Any, args_config: dict[str, Any]) -> None:
    config_path = args.config_path or "configs/train.yaml"
    run_training(load_train_config(config_path, args_config, args.skip_metadata))


@beartype
def load_inference_model(
    model_type: str,
    model_path: str,
    training_config_path: Optional[str],
    args_config: dict[str, Any],
    device: str,
    infer_with_dropout: bool,
) -> LoadedInferenceModel:
    """Load only the new portable-model or exact-run artifact formats."""

    del training_config_path
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    artifact_type = payload.get("artifact_type") if isinstance(payload, dict) else None
    interface_name = args_config.get("model_interface") or args_config.get("dataset")
    if artifact_type == "sequifier_model":
        network, config, _ = load_model_artifact(
            model_path, device=device, interface_name=interface_name
        )
    elif artifact_type == "sequifier_run_checkpoint":
        network, config, _ = load_weights_from_run_checkpoint(
            model_path, device=device, interface_name=interface_name
        )
    else:
        raise ValueError(
            "Unsupported PyTorch artifact. Sequifier accepts only the current "
            "portable model or run-checkpoint format."
        )
    if interface_name is None:
        if len(network.interfaces) != 1:
            raise ValueError(
                "model_interface is required for a multi-interface artifact."
            )
        interface_name = next(iter(network.interfaces))
    if interface_name not in network.interfaces:
        mapped = next(
            (
                dataset.model_interface
                for name, dataset in config.dataset_training.items()
                if name == interface_name
            ),
            None,
        )
        if mapped is None:
            raise ValueError(f"Unknown model interface {interface_name!r}.")
        interface_name = mapped
    network.eval()
    if infer_with_dropout:
        for module in network.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    if not device.startswith("mps"):
        network = torch.compile(network)
    if model_type not in {"generative", "embedding"}:
        raise ValueError(f"Unknown PT model type: {model_type!r}.")
    return LoadedInferenceModel(
        network=network,
        config=config,
        interface_name=interface_name,
        embedding=model_type == "embedding",
    )


def _tensor_batches(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    metadata: list[dict[str, np.ndarray]],
    device: str,
    column_data_types: dict[str, torch.dtype],
):
    categorical = set(model.categorical_columns)
    layer_types = model.config.global_training.layer_type_dtypes or {}
    reference_dtype = get_torch_dtype(layer_types.get("linear", "float32"))
    for index, features in enumerate(x):
        values = {
            column: torch.from_numpy(array).to(
                device,
                dtype=(
                    torch.int64
                    if column in categorical
                    else column_data_types.get(column, reference_dtype)
                ),
            )
            for column, array in features.items()
        }
        metadata_values = {
            column: torch.from_numpy(array).to(device)
            for column, array in (metadata[index] if metadata else {}).items()
        }
        yield values, metadata_values


@beartype
def infer_with_embedding_model(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> np.ndarray:
    del size, target_columns
    sites = tuple(
        embedding_layer_trace_site(name) for name in model.config.embedding_layer_names
    )
    outputs = []
    with torch.no_grad():
        for features, metadata_values in _tensor_batches(
            model, x, metadata, device, column_data_types
        ):
            traced = model.network.trace(
                features,
                metadata_values,
                CaptureRequest(sites=sites),
                interface_name=model.interface_name,
            )
            route = model.network.resolve_interface(model.interface_name)
            embedding = torch.cat(
                [
                    traced.captures[site][:, -route.prediction_length :]
                    for site in sites
                ],
                dim=-1,
            )
            array = embedding.cpu().float().numpy()
            outputs.append(array.reshape(-1, array.shape[-1]))
    return np.concatenate(outputs, axis=0)


@beartype
def infer_with_generative_model(
    model: LoadedInferenceModel,
    x: list[dict[str, np.ndarray]],
    device: str,
    size: int,
    target_columns: list[str],
    metadata: list[dict[str, np.ndarray]],
    column_data_types: dict[str, torch.dtype],
) -> dict[str, np.ndarray]:
    outputs: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for features, metadata_values in _tensor_batches(
            model, x, metadata, device, column_data_types
        ):
            result = model.network(
                features,
                metadata_values,
                interface_name=model.interface_name,
            )
            route = model.network.resolve_interface(model.interface_name)
            outputs.append(
                {
                    target: (
                        torch.log_softmax(
                            value[:, result.prediction_positions].float(), dim=-1
                        )
                        if route.target_column_types[target] == "categorical"
                        else value[:, result.prediction_positions].float()
                    ).cpu()
                    for target, value in result.logits.items()
                }
            )
    return {
        target: np.concatenate(
            [
                batch[target].numpy().reshape(-1, batch[target].shape[-1])
                for batch in outputs
            ],
            axis=0,
        )[:size]
        for target in target_columns
    }
