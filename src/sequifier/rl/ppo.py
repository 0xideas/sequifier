"""Synchronous single-slot categorical PPO, with portable actor-only export."""

import hashlib
import platform
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from sequifier.model.factory import build_transformer_network
from sequifier.optimizers.optimizers import get_optimizer_class, get_scheduler_class
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_exact_snapshot,
)
from sequifier.rl.models import HistoryCritic
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.rl.ppo_buffer import write_json_atomic
from sequifier.rl.ppo_checkpoint import (
    capture_rng_state,
    clone_weights,
    contract_fingerprint,
    load_ppo_checkpoint,
    restore_rng_state,
    resume_contract,
    save_ppo_checkpoint,
    snapshots_equal,
)
from sequifier.rl.ppo_collector import PPOCollector, stream_seed
from sequifier.rl.ppo_evaluation import evaluate_policy, require_evaluation_timer
from sequifier.rl.ppo_update import update_ppo


@dataclass(frozen=True)
class PPOTrainingResult:
    policy: CategoricalPolicy
    metrics: tuple[dict, ...]
    evaluations: tuple[dict, ...]
    env_steps: int
    iterations: int
    best_metric: float | None
    report: dict


def _make_optimizer(module, spec, learning_rate):
    return get_optimizer_class(spec.name)(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=learning_rate,
        **spec.arguments,
    )


def _initial_actor(config, execution, will_resume):
    actor = CategoricalPolicy.from_network(
        build_transformer_network(
            execution, device=torch.device(config.device)
        ).network,
        execution,
        config.policy.interface_name,
    )
    provenance = None
    if config.initialize_from is not None and not will_resume:
        source = load_policy(config.initialize_from, device=config.device)
        if (
            source.config != config.policy
            or source.interface.selected_columns_statistics
            != config.normalization_statistics
            or source.execution_config.model != execution.model
        ):
            raise ValueError(
                "PPO initialize_from must exactly match policy, normalization, model architecture and context"
            )
        actor.network.load_state_dict(source.network.state_dict(), strict=True)
        provenance = {
            "path": config.initialize_from,
            "sha256": hashlib.sha256(
                Path(config.initialize_from).read_bytes()
            ).hexdigest(),
        }
    return actor.eval(), provenance


def run_ppo_training(config) -> PPOTrainingResult:
    """Train one authored PPO phase; commits occur only after complete updates."""
    from sequifier.config.train_config import PPOTrainConfig
    from sequifier.helpers import configure_determinism

    if not isinstance(config, PPOTrainConfig):
        raise TypeError("run_ppo_training requires PPOTrainConfig")
    require_evaluation_timer()
    import resource

    configure_determinism(config.seed, config.enforce_determinism)
    phase = config.training_plan.phases[0]
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    checkpoint = Path(config.checkpoint_path)
    if phase.resume.policy == "required" and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    will_resume = phase.resume.policy != "never" and checkpoint.is_file()
    payload = load_ppo_checkpoint(checkpoint, config) if will_resume else None
    if (
        config.initialize_from is not None
        and not will_resume
        and not Path(config.initialize_from).is_file()
    ):
        raise FileNotFoundError(config.initialize_from)
    execution = policy_execution_config(
        config.policy,
        config.model,
        config.device,
        statistics=config.normalization_statistics,
        project_root=config.project_root,
        model_name=config.model_name,
        seed=config.seed,
    )
    actor, provenance = _initial_actor(config, execution, will_resume)
    actor.ppo_policy_version = 0
    actor.set_normalization_statistics(config.normalization_statistics)
    value = HistoryCritic(execution, config.policy.interface_name, None).eval()
    modules = {"actor": actor, "value": value}
    if any(
        not torch.isfinite(tensor).all()
        for module in modules.values()
        for tensor in module.state_dict().values()
    ):
        raise ValueError("PPO actor and V initialization must be finite")
    ids = [
        id(parameter)
        for module in modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("PPO actor and V must own disjoint trainable parameters")
    if (
        value.interface.selected_columns_statistics
        != actor.interface.selected_columns_statistics
    ):
        raise ValueError("PPO actor and V normalization differs")
    plans = {
        "actor": (
            actor,
            phase.actor_optimizer,
            phase.actor_learning_rate,
            phase.actor_scheduler,
        ),
        "value": (
            value,
            phase.value_optimizer,
            phase.value_learning_rate,
            phase.value_scheduler,
        ),
    }
    optimizers = {
        role: _make_optimizer(module, optimizer_spec, learning_rate)
        for role, (module, optimizer_spec, learning_rate, _) in plans.items()
    }
    schedulers = {
        role: get_scheduler_class(scheduler_spec.name)(
            optimizers[role], **scheduler_spec.arguments
        )
        for role, (_, _, _, scheduler_spec) in plans.items()
    }
    sampler = torch.Generator(device="cpu").manual_seed(
        stream_seed(config.seed, "minibatches", 0)
    )
    clocks = {"actor": 0, "value": 0}
    metrics, evaluations, continuations = [], [], []
    env_steps, iteration, policy_version = 0, 0, 0
    best_actor, best_metric, best_iteration = None, None, None
    trajectory_equivalent = True
    adapter = load_online_adapter(phase.environment)
    try:
        if phase.resume.mode == "exact":
            require_exact_snapshot(adapter)
        collector = PPOCollector(actor, value, adapter, phase, config.seed)
        if payload is not None:
            if payload["adapter_identity"] != adapter_identity(adapter):
                raise ValueError(
                    "PPO adapter implementation identity or snapshot capability differs"
                )
            actor.network.load_state_dict(payload["actor"], strict=True)
            value.load_state_dict(payload["value"], strict=True)
            for role in optimizers:
                optimizers[role].load_state_dict(payload["optimizers"][role])
                schedulers[role].load_state_dict(payload["schedulers"][role])
            sampler.set_state(payload["sampler"])
            clocks = dict(payload["optimizer_steps"])
            env_steps, iteration, policy_version = (
                payload["env_steps"],
                payload["iteration"],
                payload["policy_version"],
            )
            best_actor, best_metric, best_iteration = (
                payload["best_actor"],
                payload["best_metric"],
                payload["best_iteration"],
            )
            metrics, evaluations = (
                list(payload["metrics"]),
                list(payload["evaluations"]),
            )
            continuations = list(payload["continuations"])
            provenance = payload["initialization"]
            trajectory_equivalent = payload["trajectory_equivalent"]
            exact = phase.resume.mode == "exact"
            if exact:
                if payload["environment_state"] is None:
                    raise ValueError(
                        "checkpoint has no declared exact environment snapshot"
                    )
                adapter.restore_state(deepcopy(payload["environment_state"]))
                if not snapshots_equal(
                    adapter.snapshot_state(), payload["environment_state"]
                ):
                    raise ValueError(
                        "adapter did not restore its simulator and RNG snapshot exactly"
                    )
            else:
                trajectory_equivalent = False
            collector.load_state_dict(payload["collector"], exact=exact)
            continuations.append(
                {
                    "iteration": iteration,
                    "mode": phase.resume.mode,
                    "trajectory_equivalent": trajectory_equivalent,
                    "dropped_incomplete_episode": not exact
                    and not payload["collector"]["ended"],
                }
            )
            actor.ppo_policy_version = policy_version
            restore_rng_state(device, payload["random"])

        def evaluate():
            nonlocal best_actor, best_metric, best_iteration
            result = evaluate_policy(actor, phase.evaluation, forbidden_adapter=adapter)
            result.update(
                {
                    "env_steps": env_steps,
                    "iteration": iteration,
                    "policy_version": policy_version,
                }
            )
            evaluations.append(result)
            score = result["selection_value"]
            better = score is not None and (
                best_metric is None
                or (
                    score > best_metric
                    if phase.evaluation.direction == "max"
                    else score < best_metric
                )
            )
            if better:
                best_metric, best_iteration = score, iteration
                best_actor = clone_weights(actor.network)
            return result["seconds"]

        if not evaluations:
            evaluate()
        while env_steps < phase.total_env_steps:
            horizon = min(phase.rollout_length, phase.total_env_steps - env_steps)
            started = time.monotonic()
            buffer, episodes = collector.collect(horizon, policy_version)
            collection_seconds = time.monotonic() - started
            if phase.archive_directory is not None:
                buffer.archive(
                    Path(phase.archive_directory)
                    / f"collection-{iteration + 1:08d}.json",
                    include_scores=phase.archive_scores,
                )
            started = time.monotonic()
            record = update_ppo(
                actor,
                value,
                buffer,
                phase,
                optimizers,
                schedulers,
                sampler,
                policy_version,
                clocks,
            )
            update_seconds = time.monotonic() - started
            env_steps += horizon
            iteration += 1
            policy_version += 1
            evaluation_seconds = 0.0
            if (
                iteration % phase.evaluation_interval == 0
                or env_steps == phase.total_env_steps
            ):
                evaluation_seconds = evaluate()
            completed_returns = [
                episode["return"] for episode in episodes if episode["terminated"]
            ]
            completed_lengths = [
                episode["length"] for episode in episodes if episode["terminated"]
            ]
            record.update(
                {
                    "iteration": iteration,
                    "policy_version": policy_version,
                    "collected_policy_version": buffer.policy_version,
                    "env_steps": env_steps,
                    "collected_decisions": horizon,
                    "collection_seconds": collection_seconds,
                    "update_seconds": update_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "collection_steps_per_second": horizon
                    / max(collection_seconds, 1e-9),
                    "finished_episodes": episodes,
                    "training_mean_completed_return": statistics.fmean(
                        completed_returns
                    )
                    if completed_returns
                    else None,
                    "training_return_std": statistics.pstdev(completed_returns)
                    if completed_returns
                    else None,
                    "training_mean_completed_length": statistics.fmean(
                        completed_lengths
                    )
                    if completed_lengths
                    else None,
                    "completed_episode_count": sum(
                        episode["terminated"] for episode in episodes
                    ),
                    "censored_episode_count": sum(
                        episode["truncated"] and not episode["terminated"]
                        for episode in episodes
                    ),
                    "invalid_episode_count": 0,
                    "live_episode_length": 0
                    if collector.ended
                    else collector.episode_length,
                    "buffer_estimated_bytes": buffer.estimated_bytes,
                    "actor_learning_rate": optimizers["actor"].param_groups[0]["lr"],
                    "value_learning_rate": optimizers["value"].param_groups[0]["lr"],
                }
            )
            metrics.append(record)
            logger.info(
                "PPO iteration {}: {}/{} actions, policy loss {:.5f}, V loss {:.5f}",
                iteration,
                env_steps,
                phase.total_env_steps,
                record["policy_loss"],
                record["value_loss"],
            )
            if (
                iteration % phase.checkpoint_interval == 0
                or env_steps == phase.total_env_steps
            ):
                environment_state = None
                if getattr(adapter, "supports_exact_snapshot", None) is True:
                    require_exact_snapshot(adapter)
                    environment_state = adapter.snapshot_state()
                    if not isinstance(environment_state, dict) or not snapshots_equal(
                        environment_state, environment_state
                    ):
                        raise ValueError(
                            "adapter exact snapshot must be a finite, portable state dictionary"
                        )
                save_ppo_checkpoint(
                    checkpoint,
                    {
                        "artifact_type": "sequifier_ppo_run",
                        "format_version": 1,
                        "config": resume_contract(config),
                        "config_fingerprint": contract_fingerprint(config),
                        "policy": config.policy.model_dump(mode="json"),
                        "statistics": config.normalization_statistics,
                        "actor_spec": config.model.model_dump(mode="json"),
                        "value_spec": {
                            "type": "HistoryCritic",
                            "action_count": None,
                            "model": config.model.model_dump(mode="json"),
                        },
                        "actor": clone_weights(actor.network),
                        "value": clone_weights(value),
                        "optimizers": {
                            role: optimizer.state_dict()
                            for role, optimizer in optimizers.items()
                        },
                        "schedulers": {
                            role: scheduler.state_dict()
                            for role, scheduler in schedulers.items()
                        },
                        "optimizer_steps": clocks,
                        "sampler": sampler.get_state(),
                        "random": capture_rng_state(device),
                        "seed_streams": {
                            "version": 1,
                            "run_seed": config.seed,
                            "evaluation_seeds": phase.evaluation.seeds,
                        },
                        "collector": collector.state_dict(),
                        "environment_state": environment_state,
                        "adapter_identity": adapter_identity(adapter),
                        "env_steps": env_steps,
                        "iteration": iteration,
                        "policy_version": policy_version,
                        "best_actor": best_actor,
                        "best_metric": best_metric,
                        "best_iteration": best_iteration,
                        "initialization": provenance,
                        "trajectory_equivalent": trajectory_equivalent,
                        "continuations": continuations,
                        "metrics": metrics,
                        "evaluations": evaluations,
                    },
                )
    finally:
        adapter.close()
    if best_actor is not None:
        actor.network.load_state_dict(best_actor, strict=True)
    elapsed = sum(
        record["collection_seconds"] + record["update_seconds"] for record in metrics
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "format": "sequifier.ppo-training-report",
        "version": 1,
        "algorithm": "ppo",
        "config": config.model_dump(mode="json"),
        "config_fingerprint": contract_fingerprint(config),
        "env_steps": env_steps,
        "iterations": iteration,
        "policy_version": policy_version,
        "optimizer_steps": clocks,
        "metrics": metrics,
        "evaluations": evaluations,
        "monitor": {
            "metric": phase.evaluation.metric,
            "direction": phase.evaluation.direction,
            "value": best_metric,
            "available": best_metric is not None,
            "best_iteration": best_iteration,
        },
        "export_selection": "best_evaluated_actor"
        if best_actor is not None
        else "latest_actor_metric_unavailable",
        "initialization": provenance,
        "trajectory_equivalent": trajectory_equivalent,
        "continuations": continuations,
        "invalid_episode_count": 0,
        "evaluation_censored_count": sum(
            result["censored_count"] for result in evaluations
        ),
        "training_completed_count": sum(
            record["completed_episode_count"] for record in metrics
        ),
        "training_censored_count": sum(
            record["censored_episode_count"] for record in metrics
        ),
        "final_live_episode_censored": not collector.ended,
        "hardware": {
            "device": str(device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
        },
        "context_length": config.context_length,
        "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
        "value_parameters": sum(parameter.numel() for parameter in value.parameters()),
        "steps_per_second": env_steps / max(elapsed, 1e-9),
        "peak_process_memory_bytes": peak_rss
        if sys.platform == "darwin"
        else peak_rss * 1024,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "model_path": config.model_path,
        "checkpoint_path": config.checkpoint_path,
    }
    save_policy(actor, config.model_path, metadata=report)
    write_json_atomic(config.report_path, report)
    return PPOTrainingResult(
        actor,
        tuple(metrics),
        tuple(evaluations),
        env_steps,
        iteration,
        best_metric,
        report,
    )
