"""Supervised masked action likelihood using Sequifier's optimizer transaction."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader

from sequifier.config.policy_config import PolicyTrainingConfig
from sequifier.integration.callbacks import IntegrationManager
from sequifier.integration.contexts import StepIdentity
from sequifier.rl.distributions import masked_action_likelihood
from sequifier.rl.policy import CategoricalPolicy
from sequifier.training.optimization import OptimizationRuntime, UpdatePolicy
from sequifier.trajectory.dataset import TrajectoryDataset, collate_policy_examples


@dataclass(frozen=True)
class PolicyTrainingResult:
    policy: CategoricalPolicy
    metrics: tuple[dict[str, float], ...]
    best_epoch: int
    optimizer_steps: int
    training_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]


def train_policy(
    training: TrajectoryDataset,
    validation: TrajectoryDataset,
    config: PolicyTrainingConfig,
) -> PolicyTrainingResult:
    if not len(training) or not len(validation):
        raise ValueError("policy training and validation partitions must be nonempty")
    if (
        training.config != validation.config
        or training.action_semantics != validation.action_semantics
    ):
        raise ValueError(
            "training and validation must share the policy/return definition"
        )
    train_ids = tuple(episode.episode_id for episode in training.episodes)
    validation_ids = tuple(episode.episode_id for episode in validation.episodes)
    if set(train_ids) & set(validation_ids):
        raise ValueError(
            "training and validation partitions must not share episode IDs"
        )
    # Keep initialization and batch order equal across the RC and BC comparison.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = CategoricalPolicy(training.config)
        policy.set_normalization(*training.normalization_rows())
        optimization = OptimizationRuntime.create(
            config, "cpu", policy.parameters(), phase_epochs=config.epochs
        )
        integrations = IntegrationManager()
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            training,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_policy_examples,
        )
        validation_loader = DataLoader(
            validation, batch_size=config.batch_size, collate_fn=collate_policy_examples
        )
        metrics = []
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None
        global_batch = 0
        optimization.optimizer.zero_grad(set_to_none=True)
        for epoch in range(config.epochs):
            policy.train()
            total_loss = 0.0
            decisions = 0
            for batch_index, batch in enumerate(train_loader):
                distribution = policy.distribution(batch.inputs)
                loss = masked_action_likelihood(
                    distribution, batch.actions, batch.action_valid_mask
                )
                if not torch.isfinite(loss):
                    raise ValueError("policy action likelihood became non-finite")
                count = int(batch.action_valid_mask.sum())
                total_loss += float(loss.detach()) * count
                decisions += count
                identity = StepIdentity(
                    epoch,
                    batch_index,
                    global_batch,
                    optimization.optimizer_step,
                    0,
                    1,
                    0,
                    1,
                )
                optimization.accumulate(loss, identity, integrations, policy)
                result = optimization.complete_step(
                    policy, identity, integrations, UpdatePolicy()
                )
                if result.stop_requested or not result.applied:
                    raise RuntimeError(
                        "local policy optimizer did not apply its update"
                    )
                global_batch += 1
            policy.eval()
            validation_loss = 0.0
            validation_count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    loss = masked_action_likelihood(
                        policy.distribution(batch.inputs),
                        batch.actions,
                        batch.action_valid_mask,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "validation action likelihood became non-finite"
                        )
                    count = int(batch.action_valid_mask.sum())
                    validation_loss += float(loss) * count
                    validation_count += count
            validation_loss /= validation_count
            metrics.append(
                {
                    "epoch": float(epoch),
                    "training_nll": total_loss / decisions,
                    "validation_nll": validation_loss,
                }
            )
            logger.info(
                f"Policy epoch {epoch + 1}/{config.epochs}: "
                f"action_nll={total_loss / decisions:.6f}, "
                f"validation_action_nll={validation_loss:.6f}"
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = deepcopy(policy.state_dict())
            optimization.step_scheduler(validation_loss)
        if best_weights is None:
            raise ValueError(
                "policy training did not produce finite validation likelihood"
            )
        policy.load_state_dict(best_weights, strict=True)
        policy.eval()
    return PolicyTrainingResult(
        policy,
        tuple(metrics),
        best_epoch,
        optimization.optimizer_step,
        train_ids,
        validation_ids,
    )
