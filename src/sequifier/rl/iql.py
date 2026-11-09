"""Fixed-dataset categorical Implicit Q-Learning."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from sequifier.artifacts.run_checkpoint import checkpoint_path
from sequifier.io.iql_dataset import IQLBatch, IQLTransitionDataset
from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode
from sequifier.io.tensors import collate_tensor_records
from sequifier.model.factory import build_transformer_network
from sequifier.optimizers.optimizers import get_optimizer_class, get_scheduler_class
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.models import make_iql_models
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.runtime.random_state import RandomState, RandomStateManager


@dataclass(frozen=True)
class IQLTrainingResult:
    policy: CategoricalPolicy
    metrics: tuple[dict[str, float], ...]
    updates: int
    training_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]


def _select(inputs, mask):
    return type(inputs)(
        {name: value[mask] for name, value in inputs.inputs.items()},
        {name: value[mask] for name, value in inputs.metadata.items()},
    )


def _optimizer(module, spec, learning_rate):
    if "lr" in spec.arguments:
        raise ValueError("IQL optimizer learning rate belongs in the phase field")
    return get_optimizer_class(spec.name)(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=learning_rate,
        **spec.arguments,
    )


def _step(loss, optimizer, parameters, clip):
    if not torch.isfinite(loss):
        raise ValueError("non-finite IQL loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("IQL optimizer produced non-finite parameters")
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach())


def _polyak(target, source, rate):
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.lerp_(parameter, rate)
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(buffer)


def _checkpoint_payload(
    config,
    data,
    fitted,
    statistics,
    actor,
    value,
    q1,
    q2,
    targets,
    optimizers,
    schedulers,
    sampler,
    metrics,
    update,
    best_loss,
    best_actor,
    random,
):
    return {
        "artifact_type": "sequifier_iql_run",
        "format_version": 1,
        "config": _resume_contract(config),
        "data_contract": [part.data_contract for part in data],
        "fitted_policy": fitted.model_dump(mode="json"),
        "statistics": statistics,
        "update": update,
        "best_loss": best_loss,
        "best_actor": best_actor,
        "metrics": metrics,
        "networks": {
            name: module.state_dict()
            for name, module in (
                ("actor", actor.network),
                ("value", value),
                ("q1", q1),
                ("q2", q2),
                ("target_q1", targets[0]),
                ("target_q2", targets[1]),
            )
        },
        "optimizers": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "schedulers": {
            name: scheduler.state_dict() for name, scheduler in schedulers.items()
        },
        "sampler": sampler.get_state(),
        "random": random.capture_local().state_dict(),
    }


def _resume_contract(config):
    values = config.model_dump(mode="json")
    values["global_training"].pop("resume", None)
    return values


def _save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluate(actor, value, q1, q2, dataset, batch_size, gamma, device):
    totals = {"action_nll": 0.0, "bellman_residual": 0.0, "value": 0.0, "q_min": 0.0}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = cast(
                IQLBatch,
                collate_tensor_records(
                    [
                        dataset[index]
                        for index in range(start, min(start + batch_size, len(dataset)))
                    ]
                ),
            ).to(device)
            action = batch.action
            v = value(batch.current)
            q = torch.minimum(q1(batch.current, action), q2(batch.current, action))
            nll = -actor.evaluate_actions(batch.current, action)
            successor_value = torch.zeros_like(v)
            if batch.bootstrap.any():
                successor_value[batch.bootstrap] = value(
                    _select(batch.successor, batch.bootstrap)
                )
            totals["action_nll"] += float(nll.sum())
            totals["bellman_residual"] += float(
                (q - batch.reward - gamma * successor_value).abs().sum()
            )
            totals["value"] += float(v.sum())
            totals["q_min"] += float(q.sum())
    return {key: value / len(dataset) for key, value in totals.items()}


def run_iql_training(config) -> IQLTrainingResult:
    phase = config.training_plan.phases[0]
    spec = config.policy
    if spec.return_conditioned:
        raise ValueError("IQL requires an unconditioned policy")
    if config.global_training.distributed or config.global_training.data_parallelism:
        raise ValueError("IQL currently requires local execution")
    if config.global_training.layer_autocast or any(
        dtype != "float32"
        for dtype in (config.global_training.layer_type_dtypes or {}).values()
    ):
        raise ValueError("IQL requires float32 layers without autocast")
    if config.global_training.torch_compile != "none":
        raise ValueError("IQL requires torch_compile: none")
    if config.global_training.accumulation_steps not in (None, 1):
        raise ValueError("IQL does not support gradient accumulation")
    if config.global_training.num_workers:
        raise ValueError(
            "IQL samples transitions in the main process; num_workers must be zero"
        )
    for role in ("actor", "value", "critic"):
        optimizer = getattr(phase, f"{role}_optimizer")
        if optimizer.name not in {"Adam", "AdamW", "SGD", "RMSprop"}:
            raise ValueError(
                f"IQL {role} optimizer must be Adam, AdamW, SGD, or RMSprop"
            )
        if "lr" in optimizer.arguments:
            raise ValueError("IQL optimizer learning rates belong in phase fields")
        scheduler = getattr(phase, f"{role}_scheduler")
        if scheduler.name == "ReduceLROnPlateau":
            raise ValueError("IQL schedulers advance on optimizer updates, not metrics")
    checkpoint = checkpoint_path(config)
    resume = config.global_training.resume
    if resume is not None and resume.policy == "required" and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    will_resume = (
        resume is not None and resume.policy != "never" and checkpoint.is_file()
    )
    if (
        config.initialize_from is not None
        and not will_resume
        and not Path(config.initialize_from).is_file()
    ):
        raise FileNotFoundError(config.initialize_from)
    source_names = phase.trajectory_sources + (phase.validation_sources or [])
    from sequifier.trajectory.importers import source_paths

    sources = {name: source_paths(config.sources[name]) for name in source_names}
    paths = [path for values in sources.values() for path in values]
    protected = set(paths)
    for name in source_names:
        source = config.sources[name]
        protected.update(
            Path(value).resolve()
            for value in (
                source.manifest,
                source.metadata_config_path,
                source.static_data_path,
            )
            if value is not None
        )
    if config.initialize_from is not None:
        protected.add(Path(config.initialize_from).resolve())
    outputs = {Path(config.model_path), Path(config.report_path), checkpoint}
    if len(outputs) != 3 or outputs.intersection(protected):
        raise ValueError("IQL outputs must be distinct from trajectory inputs")
    if len(paths) != len(set(paths)):
        raise ValueError("IQL trajectory sources must not overlap")

    def corpus(names):
        return PolicySequenceDataset(
            [config.sources[name] for name in names],
            spec,
            episode_cache_size=config.global_training.episode_cache_size,
        )

    training = corpus(phase.trajectory_sources)
    if phase.validation_sources is None:
        training, validation = split_by_episode(
            training,
            config.global_training.validation_fraction,
            config.seed,
            group_identical=True,
        )
    else:
        validation = corpus(phase.validation_sources)
    if not len(training) or not len(validation):
        raise ValueError("IQL needs nonempty train and validation partitions")
    for attribute in ("episode_ids", "group_ids", "demonstration_keys"):
        if (set(getattr(training, attribute)) & set(getattr(validation, attribute))) - {
            None
        }:
            raise ValueError(f"IQL {attribute} overlaps training and validation")
    fitted = training.fit_categorical_config()
    for part in (training, validation):
        part.use_config(fitted)
    data = (IQLTransitionDataset(training), IQLTransitionDataset(validation))
    statistics = training.normalization_statistics()
    execution = policy_execution_config(
        fitted,
        config.model,
        config.device,
        global_training=config.global_training,
        statistics=statistics,
        project_root=config.project_root,
        model_name=config.model_name,
        seed=config.seed,
    )
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    actor = CategoricalPolicy.from_network(
        build_transformer_network(execution, device=device).network,
        execution,
        fitted.interface_name,
    )
    value, q1, q2, targets = make_iql_models(
        execution, fitted.interface_name, len(fitted.action_values)
    )
    if config.initialize_from is not None and not will_resume:
        source = load_policy(config.initialize_from, device=config.device)
        if (
            source.config != fitted
            or source.interface.selected_columns_statistics != statistics
            or source.execution_config.model != execution.model
        ):
            raise ValueError(
                "IQL warm-start policy schema, normalization, or model architecture differs"
            )
        actor.network.load_state_dict(source.network.state_dict(), strict=True)
    modules = {
        "actor": actor.network,
        "value": value,
        "critic": torch.nn.ModuleList([q1, q2]),
    }
    trainable_ids = [
        id(parameter)
        for module in modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if len(trainable_ids) != len(set(trainable_ids)):
        raise ValueError("IQL optimizer parameter assignments overlap")
    optimizers = {
        role: _optimizer(
            module,
            getattr(phase, f"{role}_optimizer"),
            getattr(phase, f"{role}_learning_rate"),
        )
        for role, module in modules.items()
    }
    schedulers = {
        role: get_scheduler_class(getattr(phase, f"{role}_scheduler").name)(
            optimizer, **getattr(phase, f"{role}_scheduler").arguments
        )
        for role, optimizer in optimizers.items()
    }
    for module in (actor, value, q1, q2, *targets):
        module.eval()
    sampler = torch.Generator(device="cpu").manual_seed(config.seed)
    random = RandomStateManager(device)
    metrics = []
    update_start = 0
    best_loss = math.inf
    best_actor = None
    if resume is not None and resume.policy != "never":
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError("IQL checkpoint must be a mapping")
            if (
                payload.get("artifact_type") != "sequifier_iql_run"
                or payload.get("format_version") != 1
            ):
                raise ValueError("IQL resume requires an IQL learner checkpoint")
            if (
                payload["config"] != _resume_contract(config)
                or payload["data_contract"] != [part.data_contract for part in data]
                or payload["fitted_policy"] != fitted.model_dump(mode="json")
                or payload["statistics"] != statistics
            ):
                raise ValueError("IQL resume config, data, or codec differs")
            for name, module in (
                ("actor", actor.network),
                ("value", value),
                ("q1", q1),
                ("q2", q2),
                ("target_q1", targets[0]),
                ("target_q2", targets[1]),
            ):
                module.load_state_dict(payload["networks"][name], strict=True)
            for name in optimizers:
                optimizers[name].load_state_dict(payload["optimizers"][name])
                schedulers[name].load_state_dict(payload["schedulers"][name])
            sampler.set_state(payload["sampler"])
            random.restore(RandomState.from_state_dict(payload["random"]))
            metrics = list(payload["metrics"])
            update_start = int(payload["update"])
            if not 0 <= update_start <= phase.updates:
                raise ValueError(
                    "IQL checkpoint update is outside the configured budget"
                )
            best_loss = float(payload["best_loss"])
            best_actor = payload["best_actor"]
            if not math.isfinite(best_loss) and best_actor is not None:
                raise ValueError("IQL checkpoint has an invalid best actor score")

    for update in range(update_start + 1, phase.updates + 1):
        indices = torch.randint(len(data[0]), (phase.batch_size,), generator=sampler)
        batch = cast(
            IQLBatch,
            collate_tensor_records([data[0][int(index)] for index in indices]),
        ).to(device)
        with torch.no_grad():
            qbar = torch.minimum(
                targets[0](batch.current, batch.action),
                targets[1](batch.current, batch.action),
            )
            old_v = value(batch.current)
            advantage = qbar - old_v
            weight = torch.exp(
                (phase.inverse_temperature * advantage).clamp(
                    max=math.log(phase.max_weight)
                )
            )
            successor_value = torch.zeros_like(old_v)
            if batch.bootstrap.any():
                successor_value[batch.bootstrap] = value(
                    _select(batch.successor, batch.bootstrap)
                )
            target = batch.reward + phase.gamma * successor_value
        delta = qbar - value(batch.current)
        v_loss = (
            torch.where(delta < 0, 1 - phase.expectile, phase.expectile)
            * delta.square()
        )
        v_loss = v_loss.mean()
        v_norm = _step(
            v_loss, optimizers["value"], list(value.parameters()), phase.gradient_clip
        )
        schedulers["value"].step()
        q1_value, q2_value = (
            q1(batch.current, batch.action),
            q2(batch.current, batch.action),
        )
        q_loss = ((q1_value - target).square() + (q2_value - target).square()).mean()
        q_norm = _step(
            q_loss,
            optimizers["critic"],
            list(q1.parameters()) + list(q2.parameters()),
            phase.gradient_clip,
        )
        schedulers["critic"].step()
        _polyak(targets[0], q1, phase.target_update_rate)
        _polyak(targets[1], q2, phase.target_update_rate)
        actor_loss = -(
            weight * actor.evaluate_actions(batch.current, batch.action)
        ).mean()
        actor_norm = _step(
            actor_loss,
            optimizers["actor"],
            list(actor.parameters()),
            phase.gradient_clip,
        )
        schedulers["actor"].step()
        if update % phase.evaluation_interval == 0 or update == phase.updates:
            held_out = _evaluate(
                actor, value, q1, q2, data[1], phase.batch_size, phase.gamma, device
            )
            record = {
                "update": update,
                "value_loss": float(v_loss.detach()),
                "critic_loss": float(q_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "advantage_mean": float(advantage.mean()),
                "weight_mean": float(weight.mean()),
                "weight_cap_fraction": float(
                    (weight >= phase.max_weight).float().mean()
                ),
                "advantage_p10": float(torch.quantile(advantage, 0.1)),
                "advantage_p50": float(torch.quantile(advantage, 0.5)),
                "advantage_p90": float(torch.quantile(advantage, 0.9)),
                "weight_p10": float(torch.quantile(weight, 0.1)),
                "weight_p50": float(torch.quantile(weight, 0.5)),
                "weight_p90": float(torch.quantile(weight, 0.9)),
                "value_min": float(old_v.min()),
                "value_max": float(old_v.max()),
                "q_min": float(qbar.min()),
                "q_max": float(qbar.max()),
                "actor_optimizer_step": update,
                "value_optimizer_step": update,
                "critic_optimizer_step": update,
                "value_grad_norm": v_norm,
                "critic_grad_norm": q_norm,
                "actor_grad_norm": actor_norm,
                **{f"validation_{name}": result for name, result in held_out.items()},
            }
            metrics.append(record)
            if held_out["action_nll"] < best_loss:
                best_loss = held_out["action_nll"]
                best_actor = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in actor.network.state_dict().items()
                }
        if update % phase.checkpoint_interval == 0 or update == phase.updates:
            _save_checkpoint(
                checkpoint,
                _checkpoint_payload(
                    config,
                    data,
                    fitted,
                    statistics,
                    actor,
                    value,
                    q1,
                    q2,
                    targets,
                    optimizers,
                    schedulers,
                    sampler,
                    metrics,
                    update,
                    best_loss,
                    best_actor,
                    random,
                ),
            )
    if best_actor is None:
        raise ValueError("IQL did not produce a finite held-out actor likelihood")
    actor.network.load_state_dict(best_actor, strict=True)
    report = {
        "format": "sequifier.iql-training-report",
        "version": 1,
        "algorithm": "iql",
        "config": config.model_dump(mode="json"),
        "training_episode_ids": training.episode_ids,
        "validation_episode_ids": validation.episode_ids,
        "sources": {
            name: [str(path) for path in values] for name, values in sources.items()
        },
        "initialize_from": config.initialize_from,
        "training_decisions": len(data[0]),
        "validation_decisions": len(data[1]),
        "best_validation_action_nll": best_loss,
        "updates": phase.updates,
        "metrics": metrics,
        "model_path": config.model_path,
        "checkpoint_path": str(checkpoint),
    }
    save_policy(actor, config.model_path, metadata=report)
    destination = Path(config.report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return IQLTrainingResult(
        actor,
        tuple(metrics),
        phase.updates,
        tuple(training.episode_ids),
        tuple(validation.episode_ids),
    )
