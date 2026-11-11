"""CLI policy prediction and environment execution using the shared model lifecycle."""

from itertools import islice
from pathlib import Path
from typing import cast

import polars as pl
import torch
from loguru import logger

from sequifier.config.policy_infer_config import ResolvedPolicyInferenceConfig
from sequifier.helpers import configure_determinism, normalize_path, write_data
from sequifier.io.actions import action_record, decode_action, encode_action
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.sequence import ordered_sequence_rows
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import load_online_adapter
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.schema import SeedHistory
from sequifier.trajectory.store import save_trajectory


def infer_policy(config: ResolvedPolicyInferenceConfig) -> None:
    configure_determinism(config.seed, config.enforce_determinism)
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
    if spec.return_conditioned and config.desired_return is None:
        required.add(config.desired_return_column)
    if not spec.return_conditioned and config.desired_return is not None:
        raise ValueError("behavior cloning has no desired-return input")
    if required - set(data.columns):
        raise ValueError(
            f"policy inference is missing columns: {sorted(required - set(data.columns))}"
        )

    if config.rollout is not None:
        infer_rollout(policy, data, config)
        return
    predictions = predict_policy(policy, data, config)
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


def desired_return(config, spec, row):
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


def infer_rollout(policy, data, config):
    spec = policy.config
    rollout = config.rollout
    if rollout.seed_row_index >= len(data):
        raise ValueError("policy rollout seed_row_index is outside the input rows")
    row = data.row(rollout.seed_row_index, named=True)
    seed = SeedHistory(
        observations={
            column: [0.0] * (spec.context_length - 1) + [row[column]]
            for column in spec.observation_columns
        },
        attention_valid_mask=[False] * (spec.context_length - 1) + [True],
        static_inputs={column: row[column] for column in spec.static_columns},
    )
    budget = desired_return(config, spec, row)
    trajectory = collect_policy_rollout(
        policy,
        load_online_adapter(rollout.environment),
        seed,
        rollout,
        rollout.max_steps,
        desired_return=budget,
        action_selection=config.action_selection,
        random_seed=config.seed,
    )
    trajectory.metadata.update(
        {
            "model_path": config.model_path,
            "model_interface": config.model_interface,
            "seed_row_index": rollout.seed_row_index,
            "enforce_determinism": config.enforce_determinism,
            "seed": config.seed,
        }
    )
    destination = normalize_path(rollout.trajectory_path, config.project_root)
    save_trajectory(trajectory, destination)
    logger.info(
        f"Wrote {len(trajectory.transitions)} policy transitions to '{destination}'"
    )


def policy_windows(data, spec, config):
    # Rows without a sequence ID are independent single-observation decisions.
    # Within sequences, recorded actions determine the causal previous-action input.
    yield from joint_policy_windows(data, spec, config)
    return


@torch.no_grad()
def predict_policy(policy, data, config):
    spec = policy.config
    windows = policy_windows(data, spec, config)
    generator = action_generator(policy.device, config.seed)
    outputs = []
    while batch := list(islice(windows, config.inference_batch_size)):
        rows, histories = zip(*batch)
        inputs = cast(SequifierInputs, collate_model_inputs(histories)).to(
            policy.device
        )
        distribution = policy.distribution(inputs)
        sampled = distribution.select(
            config.action_selection == "deterministic", generator
        )
        actions = [action_record(sampled, i) for i in range(len(rows))]
        probabilities = (
            distribution.categorical_probabilities(sampled)
            if config.output_probabilities
            else {}
        )
        for index, (row_index, action) in enumerate(zip(rows, actions)):
            output = {
                "__policy_row": row_index,
                **decode_action(action, spec),
            }
            if config.output_probabilities:
                output["action_log_probability"] = float(
                    distribution.log_prob(sampled)[index]
                )
            for name, values in probabilities.items():
                for action_id, value in enumerate(values[index]):
                    output[f"{name}__probability_{action_id}"] = float(value)
            outputs.append(output)
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
    return predictions


def joint_policy_windows(data, spec, config):
    indexed = data.with_row_index("__policy_row")
    groups = (
        indexed.partition_by(config.sequence_id_column, maintain_order=True)
        if config.sequence_id_column in data.columns
        else [indexed.slice(i, 1) for i in range(len(indexed))]
    )
    view = PolicyWindowView(spec)
    for group in groups:
        if config.item_position_column not in group.columns:
            group = group.with_row_index(config.item_position_column)
        group = ordered_sequence_rows(group, config.item_position_column)
        rows = group.to_dicts()
        previous, observations, budgets = [], [], []
        for index, row in enumerate(rows):
            if spec.previous_action_inputs:
                if index:
                    prior = {name: rows[index - 1][name] for name in spec.actions}
                    previous.append(encode_action(prior, spec))
                elif row.get(config.item_position_column, 0) != 0:
                    if not config.previous_action_columns:
                        raise ValueError(
                            "fragment inference requires previous_action_columns"
                        )
                    previous.append(
                        encode_action(
                            {
                                name: row[column]
                                for name, column in config.previous_action_columns.items()
                            },
                            spec,
                        )
                    )
                else:
                    previous.append(None)
            else:
                previous.append(None)
            observations.append({name: row[name] for name in spec.observation_columns})
            if spec.return_conditioned:
                budgets.append(desired_return(config, spec, row))
            yield (
                row["__policy_row"],
                view.inputs(
                    observations,
                    previous,
                    budgets if spec.return_conditioned else None,
                    {name: row[name] for name in spec.static_columns},
                ),
            )
