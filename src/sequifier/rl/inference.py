"""CLI policy prediction and environment execution using the shared model lifecycle."""

from pathlib import Path

import polars as pl
import torch
from loguru import logger

from sequifier.config.policy_infer_config import PolicyInferenceConfig
from sequifier.config.policy_train_config import PolicyTableColumns
from sequifier.helpers import configure_determinism, normalize_path, write_data
from sequifier.io.batch import collate_model_inputs
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.sequence import ordered_sequence_rows
from sequifier.io.values import domain_scalar
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.distributions import CategoricalActionDistribution, action_generator
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.rollout.factories import load_adapter
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.schema import SeedHistory
from sequifier.trajectory.store import save_trajectory
from sequifier.trajectory.tables import table_episode


def infer_policy(config: PolicyInferenceConfig) -> None:
    configure_determinism(config.seed, config.deterministic)
    policy = load_policy(
        config._artifacts.get(config.model_path)
        or normalize_path(config.model_path, config.project_root),
        device=config.device,
        interface_name=config.model_interface,
    )
    spec = policy.config
    path = normalize_path(config.data_path, config.project_root)
    data = pl.read_csv(path) if config.read_format == "csv" else pl.read_parquet(path)
    if not len(data):
        raise ValueError("policy inference input must be nonempty")
    required = set(spec.observation_columns + spec.static_columns)
    if config.previous_action_column is not None and config.rollout is None:
        required.add(config.previous_action_column)
    if spec.return_conditioned and config.desired_return is None:
        required.add(config.desired_return_column)
    if not spec.return_conditioned and config.desired_return is not None:
        raise ValueError("behavior cloning has no desired-return input")
    if required - set(data.columns):
        raise ValueError(
            f"policy inference is missing columns: {sorted(required - set(data.columns))}"
        )

    def budget(row):
        return (
            numeric_scalar(
                config.desired_return
                if config.desired_return is not None
                else row[config.desired_return_column],
                "desired_return",
            )
            if spec.return_conditioned
            else None
        )

    if config.rollout is not None:
        assert config.generation_steps is not None
        rollout = config.rollout
        if rollout.seed_index >= len(data):
            raise ValueError("policy rollout seed_index is outside the input rows")
        row = data.row(rollout.seed_index, named=True)
        seed = SeedHistory(
            observations={
                column: [0.0] * (spec.context_length - 1)
                + [domain_scalar(row[column], column)]
                for column in spec.observation_columns
            },
            attention_valid_mask=[False] * (spec.context_length - 1) + [True],
            static_inputs={
                column: domain_scalar(row[column], column)
                for column in spec.static_columns
            },
        )
        desired_return = budget(row)
        trajectory = collect_policy_rollout(
            policy,
            load_adapter(rollout.adapter),
            seed,
            rollout,
            config.generation_steps,
            desired_return=desired_return,
            deterministic=config.deterministic,
            random_seed=config.seed,
        )
        trajectory.metadata.update(
            {
                "model_path": config.model_path,
                "model_interface": config.model_interface,
                "seed_index": rollout.seed_index,
            }
        )
        destination = normalize_path(rollout.trajectory_path, config.project_root)
        save_trajectory(trajectory, destination)
        logger.info(
            f"Wrote {len(trajectory.transitions)} policy transitions to '{destination}'"
        )
        return

    # Rows without a sequence ID are independent single-observation decisions.
    # Within sequences, recorded actions determine the causal previous-action input.
    indexed = data.with_row_index("__policy_row")
    groups = (
        indexed.partition_by(config.sequence_id_column, maintain_order=True)
        if config.sequence_id_column in data.columns
        else [indexed.slice(i, 1) for i in range(len(indexed))]
    )
    outputs = []
    generator = action_generator(policy.device, config.seed)
    pending_inputs = []
    pending_rows = []

    def flush():
        if not pending_inputs:
            return
        with torch.no_grad():
            inputs = collate_model_inputs(pending_inputs).to(policy.device)
            decision = CategoricalActionDistribution(policy(inputs)[:, -1, :])
            actions = decision.select(config.deterministic, generator).cpu().tolist()
            probabilities = (
                decision.distribution.probs.cpu().tolist()
                if config.output_probabilities
                else None
            )
        for index, (row_index, action) in enumerate(zip(pending_rows, actions)):
            output = {
                "__policy_row": row_index,
                spec.action_column: spec.action_values[action],
            }
            if probabilities is not None:
                output.update(
                    {
                        f"{spec.action_column}__probability_{action_id}": value
                        for action_id, value in enumerate(probabilities[index])
                    }
                )
            outputs.append(output)
        pending_inputs.clear()
        pending_rows.clear()

    view = PolicyWindowView(spec)
    for group in groups:
        if config.item_position_column not in group.columns:
            group = group.with_row_index(config.item_position_column)
        group = ordered_sequence_rows(group, config.item_position_column)
        first_position = group[config.item_position_column][0]
        previous_column = config.previous_action_column
        if previous_column is not None:
            previous = group[previous_column].to_list()
            if any(
                value is None for value in previous[0 if first_position > 0 else 1 :]
            ):
                raise ValueError(
                    "null previous actions are allowed only at sequence start"
                )
            actions = previous[1:]
        else:
            if first_position > 0:
                raise ValueError(
                    "fragment inference requires an explicit preceding action"
                )
            if len(group) > 1 and (
                spec.action_column not in group.columns
                or spec.action_column in spec.observation_columns + spec.static_columns
            ):
                raise ValueError(
                    "sequential inference requires recorded actions or previous_action_column"
                )
            actions = group[spec.action_column].to_list()[:-1] if len(group) > 1 else []
        label = "__inference_action"
        while label in group.columns:
            label += "_"
        episode = table_episode(
            group.with_columns(pl.Series(label, actions + [spec.action_values[0]])),
            spec,
            PolicyTableColumns(
                action=label,
                previous_action=previous_column,
                item_position=config.item_position_column,
                reward=None,
                terminated=None,
                truncated=None,
            ),
            sequence_id=0,
        )
        returns = (
            [budget(row) for row in group.iter_rows(named=True)]
            if spec.return_conditioned
            else None
        )
        for step, row_index in enumerate(group["__policy_row"]):
            pending_inputs.append(view.episode_inputs(episode, step, returns))
            pending_rows.append(row_index)
            if len(pending_inputs) >= config.inference_batch_size:
                flush()
    flush()
    predictions = pl.DataFrame(outputs).sort("__policy_row").drop("__policy_row")
    identity = [
        column
        for column in (config.sequence_id_column, config.item_position_column)
        if column in data.columns and column not in predictions.columns
    ]
    predictions = (
        pl.concat([data.select(identity), predictions], how="horizontal")
        if identity
        else predictions
    )
    if config.output_path is None:
        from sequifier.infer import inference_output_path

        destination = inference_output_path(
            config.project_root,
            config.write_format,
            "predictions",
            Path(config.model_path).stem,
            0,
        )
    else:
        destination = normalize_path(config.output_path, config.project_root)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_data(predictions, destination, config.write_format)
    logger.info(f"Wrote {len(predictions)} policy predictions to '{destination}'")
