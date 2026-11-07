"""CLI policy prediction and environment execution using the shared model lifecycle."""

from collections import deque
from pathlib import Path

import polars as pl
import torch
from loguru import logger

from sequifier.config.policy_infer_config import PolicyInferenceConfig
from sequifier.helpers import configure_determinism, normalize_path, write_data
from sequifier.io.batch import collate_model_inputs
from sequifier.io.policy_window import PolicyWindowView, action_index
from sequifier.io.sequence import ordered_sequence_rows
from sequifier.io.values import domain_fields, domain_scalar
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.distributions import CategoricalActionDistribution
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.rollout.factories import load_adapter
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.schema import SeedHistory
from sequifier.trajectory.store import save_trajectory


def infer_policy(config: PolicyInferenceConfig) -> None:
    configure_determinism(config.seed, config.deterministic)
    policy = load_policy(
        normalize_path(config.model_path, config.project_root),
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
    generator = torch.Generator(
        device=policy.device if policy.device.type == "cuda" else "cpu"
    ).manual_seed(config.seed)
    pending_inputs = []
    pending_rows = []

    def flush():
        if not pending_inputs:
            return
        with torch.no_grad():
            inputs = collate_model_inputs(pending_inputs).to(policy.device)
            decision = CategoricalActionDistribution(policy(inputs)[:, -1, :])
            actions = (
                (
                    decision.mode()
                    if config.deterministic
                    else decision.sample(generator)
                )
                .cpu()
                .tolist()
            )
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

    with torch.no_grad():
        for group in groups:
            group = ordered_sequence_rows(group, config.item_position_column)
            first_position = (
                group[config.item_position_column][0]
                if config.item_position_column in group.columns
                else 0
            )
            if first_position > 0 and config.previous_action_column is None:
                raise ValueError(
                    "fragment inference requires an explicit preceding action"
                )
            observations, previous_actions, budgets = (
                deque(maxlen=spec.context_length) for _ in range(3)
            )
            previous_action = len(spec.action_values)
            initial_static = None
            if (
                len(group) > 1
                and config.previous_action_column is None
                and spec.action_column not in group.columns
            ):
                raise ValueError(
                    "sequential policy inference requires recorded actions or previous_action_column"
                )
            if (
                len(group) > 1
                and config.previous_action_column is None
                and spec.action_column in spec.observation_columns + spec.static_columns
            ):
                raise ValueError(
                    "previous_action_column is required when observations and actions share a name"
                )
            for position, row in enumerate(group.iter_rows(named=True)):
                static = domain_fields(
                    {column: row[column] for column in spec.static_columns},
                    set(spec.static_columns),
                    "policy static inputs",
                )
                if initial_static is not None and static != initial_static:
                    raise ValueError(
                        "policy static inputs must remain constant within a sequence"
                    )
                initial_static = static
                observations.append(
                    {column: row[column] for column in spec.observation_columns}
                )
                if config.previous_action_column is not None:
                    value = row[config.previous_action_column]
                    if (position > 0 or first_position > 0) and value is None:
                        raise ValueError(
                            "null previous actions are allowed only at sequence start"
                        )
                    previous_action = (
                        len(spec.action_values)
                        if value is None
                        else action_index(value, spec)
                    )
                previous_actions.append(previous_action)
                budgets.append(budget(row))
                pending_inputs.append(
                    PolicyWindowView(spec).inputs(
                        list(observations),
                        list(previous_actions),
                        list(budgets) if spec.return_conditioned else None,
                        static,
                    )
                )
                pending_rows.append(row["__policy_row"])
                if len(pending_inputs) >= config.inference_batch_size:
                    flush()
                if config.previous_action_column is None and position + 1 < len(group):
                    previous_action = action_index(row[spec.action_column], spec)
    flush()
    predictions = pl.DataFrame(outputs).sort("__policy_row").drop("__policy_row")
    identity = [
        column
        for column in (config.sequence_id_column, "itemPosition")
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
