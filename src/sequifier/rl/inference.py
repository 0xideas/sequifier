"""CLI policy prediction and environment execution using the shared model lifecycle."""

from collections import deque
from contextlib import contextmanager
from itertools import chain, islice
from pathlib import Path
from typing import cast

import polars as pl
import torch
from loguru import logger

from sequifier.config.inference_common import inference_image_store
from sequifier.config.policy_infer_config import ResolvedPolicyInferenceConfig
from sequifier.helpers import configure_determinism
from sequifier.io.actions import encode_action
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.inference_workspace import probability_table, rollout_journal
from sequifier.io.inference_workspace import write_inference_data as write_data
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.sequence import (
    CONTEXT_COLUMN,
    CONTEXT_RECORDS,
    DOMAIN_OBSERVATIONS,
    OBSERVATION_MASK,
    ordered_sequence_rows,
    validate_sequence_ids,
)
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.distributions import action_generator
from sequifier.rl.environment import load_online_adapter
from sequifier.rl.history import PolicyFrame, history_inputs
from sequifier.rl.runtime import collect_policy_rollout
from sequifier.rollout.history import numeric_scalar
from sequifier.trajectory.schema import SeedHistory
from sequifier.trajectory.tables import (
    ACTION_RECORD,
    PREVIOUS_ACTION_RECORD,
    project_prepared_rows,
)


def infer_policy(config: ResolvedPolicyInferenceConfig) -> None:
    configure_determinism(config.seed, config.enforce_determinism)
    policy = load_policy(config._model, device=config.device)
    spec = policy.config
    if not spec.return_conditioned and config.desired_return is not None:
        raise ValueError("behavior cloning has no desired-return input")
    required = set(spec.observation_columns + spec.static_columns)
    if spec.return_conditioned and config.desired_return is None:
        required.add(config.desired_return_column)
    with _policy_input_frames(policy, config) as frames:
        first = next((frame for frame in frames if len(frame[0])), None)
        if first is None:
            raise ValueError("policy inference input must be nonempty")
        first_rows, _ = first
        if spec.previous_action_inputs and not config.rollout:
            required.update(config.previous_action_columns.values())
        if missing := required - set(first_rows.columns):
            raise ValueError(f"policy inference is missing columns: {sorted(missing)}")
        frames = chain((first,), frames)
        if (
            config.sequence_id_column != "sequenceId"
            and config.sequence_id_column in first_rows.columns
        ):
            # A user-defined grouping key can span physical sequences.
            combined = pl.concat([rows for rows, _ in frames], how="diagonal_relaxed")
            frames = iter(((combined, None),))
        if config.rollout is not None:
            offset = config.rollout.seed_row_index
            for rows, context in frames:
                coordinate = config.rollout.seed
                if coordinate is not None:
                    if {config.sequence_id_column, config.item_position_column} - set(
                        rows.columns
                    ):
                        raise ValueError(
                            "Coordinate seeds require explicit sequence and position columns"
                        )
                    matches = (
                        (rows[config.sequence_id_column] == coordinate.sequenceId)
                        & (rows[config.item_position_column] == coordinate.itemPosition)
                    ).arg_true()
                    if not len(matches):
                        continue
                    if len(matches) != 1:
                        raise ValueError("Ambiguous rollout seed coordinate")
                    offset = int(matches[0])
                assert offset is not None
                if offset < len(rows):
                    key = policy_row_key(rows, spec, config)
                    histories = _decision_histories(
                        [(rows, context)],
                        spec,
                        config,
                        key,
                        seed_index=offset,
                    )
                    for _, history, identity in histories:
                        infer_rollout(policy, history, config, identity)
                        return
                offset -= len(rows)
            raise ValueError(
                "policy rollout seed coordinate or seed_row_index is outside the input rows"
            )
        predictions = predict_policy(
            policy,
            frames,
            config,
            policy_row_key(first_rows, spec, config),
        )
    from sequifier.infer import inference_output_path

    destination = inference_output_path(
        config.project_root,
        config.write_format,
        "predictions",
        Path(config.model_paths[0]).stem,
        0,
        output_path=config.output_path,
        config=config,
    )
    write_data(predictions, destination, config.write_format)
    logger.info(f"Wrote {len(predictions)} policy predictions to '{destination}'")


@contextmanager
def _policy_input_frames(policy, config):
    spec = policy.config
    if config.input_layout == "raw":
        from sequifier.io.prepare_inference import fresh_view_from_raw

        uses_feedback = bool(spec.previous_action_inputs) and (
            config.rollout is None or config.rollout.seed_history == "available"
        )
        auxiliary = (
            set(config.previous_action_columns.values()) if uses_feedback else set()
        )
        for column in (config.sequence_id_column, config.item_position_column):
            if column not in {"sequenceId", "itemPosition"}:
                auxiliary.add(column)
        if spec.return_conditioned and config.desired_return is None:
            auxiliary.add(config.desired_return_column)
        with fresh_view_from_raw(
            config,
            required_columns=(
                name
                for name in spec.observation_columns
                if name in policy.interface.column_data_types
            ),
            passthrough_columns=auxiliary - set(spec.observation_columns),
            optional_passthrough=(*spec.actions, ACTION_RECORD, PREVIOUS_ACTION_RECORD)
            if uses_feedback
            else (),
        ) as view:
            yield _indexed_policy_frames(view, spec, config)
    elif config._indexed_view is not None:
        yield _indexed_policy_frames(config._indexed_view, spec, config)
    else:
        assert config._prepared_source is not None
        yield _stored_policy_frames(config._prepared_source, spec, config)


def _validate_policy_coordinates(rows, config):
    for column in {"sequenceId", config.sequence_id_column}:
        if column in rows.columns:
            validate_sequence_ids(rows, column)
    for column, default in (
        (config.sequence_id_column, "sequenceId"),
        (config.item_position_column, "itemPosition"),
    ):
        if column != default and column not in rows.columns:
            raise ValueError(f"Missing configured coordinate column {column!r}")


def _indexed_policy_frames(view, spec, config):
    metadata = view.metadata()
    raw_columns = set(metadata.indexed_passthrough)
    if conflicts := raw_columns.intersection(
        spec.observation_columns + spec.static_columns
    ):
        raise ValueError(
            f"Policy model inputs are stored as raw auxiliary columns: {sorted(conflicts)}"
        )
    static = pl.read_parquet(view.static_path) if view.static_path else None
    media = inference_image_store(config, metadata, view)
    # Inline frames and masks are already present in the policy rows.
    if metadata.indexed_inline_media:
        media = None
    bounds = view.split_bounds()
    predict = view.view["view"] == "predict"
    coordinate = getattr(config.rollout, "seed", None)
    if (
        coordinate
        and config.sequence_id_column == "sequenceId"
        and config.item_position_column == "itemPosition"
    ):
        from sequifier.rollout.seeds import indexed_history

        rows = indexed_history(view, coordinate, spec.context_length + 1)
        yield _prepared_frame(
            rows,
            coordinate.sequenceId,
            coordinate.itemPosition,
            static,
            media,
            metadata,
            spec,
            config,
        )
        return
    for key in sorted(view.dataset["sequences"], key=int):
        sequence = int(key)
        if not predict and sequence not in bounds:
            continue
        start, end = bounds.get(sequence, (None, None))
        read_start = _context_start(start, spec, config)
        rows = view.read(sequence, read_start, end)
        rows, previous = _prepared_frame(
            rows, sequence, start, static, media, metadata, spec, config
        )
        yield rows, previous


def _context_start(start, spec, config):
    if (
        start is None
        or config.sequence_id_column != "sequenceId"
        or config.item_position_column != "itemPosition"
    ):
        return start
    return max(
        0, start - spec.context_length + 1 - int(bool(spec.previous_action_inputs))
    )


def _stored_policy_frames(source, spec, config):
    from sequifier.trajectory.prepared import prepared_table
    from sequifier.trajectory.tables import scan_tables

    metadata = source.metadata
    media = inference_image_store(config, metadata)
    static = (
        scan_tables((source.static_path,)).collect() if source.static_path else None
    )
    for rows in prepared_table(source).partition_by("sequenceId", maintain_order=True):
        decisions = rows.filter(pl.col("__decision"))
        if decisions.is_empty():
            continue
        start = decisions["itemPosition"][0]
        decisions = rows.filter(
            pl.col("itemPosition") >= _context_start(start, spec, config)
        )
        yield _prepared_frame(
            decisions.drop("__decision"),
            decisions["sequenceId"][0],
            start,
            static,
            media,
            metadata,
            spec,
            config,
        )


def _prepared_frame(rows, sequence, start, static, media, metadata, spec, config):
    if static is not None:
        rows = rows.join(
            static.filter(pl.col("sequenceId") == sequence), on="sequenceId"
        )
    _validate_policy_coordinates(rows, config)
    if media is not None:
        images, masks = media.gather(
            torch.tensor([sequence]),
            torch.tensor([rows["itemPosition"].to_list()]),
            torch.ones((1, len(rows)), dtype=torch.bool),
        )
        for name, value in {**images, **masks}.items():
            values = (
                [value[0].tolist()] * len(rows)
                if name == "__preconditioning_image"
                else value[0].tolist()
            )
            rows = rows.with_columns(pl.Series(name, values))
    rows = project_prepared_rows(
        rows,
        metadata,
        spec.observation_columns,
        spec.static_columns,
        spec.actions if spec.previous_action_inputs else (),
        physical_inputs=config.rollout is not None,
    )
    previous = None
    if start is not None and rows["itemPosition"][0] < start:
        previous = rows.filter(pl.col("itemPosition") < start)
        rows = rows.filter(pl.col("itemPosition") >= start)
    return rows, previous


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


def infer_rollout(policy, history, config, identity):
    from sequifier.rollout.seeds import seed_origin

    spec = policy.config
    rollout = config.rollout
    seed = SeedHistory.from_rows(
        [dict(frame.observation) for frame in history],
        spec.observation_columns,
        spec.context_length,
        dict(history[-1].static_inputs),
        [frame.masked for frame in history],
    )
    budget = history[-1].desired_return
    trajectory = collect_policy_rollout(
        policy,
        load_online_adapter(rollout.environment),
        seed,
        rollout,
        rollout.max_steps,
        desired_return=budget,
        seed_previous_actions=[
            {name: frame.previous_action[name] for name in spec.actions}
            if frame.previous_action is not None
            else None
            for frame in history
        ],
        seed_returns=[frame.desired_return for frame in history]
        if spec.return_conditioned
        else None,
        action_selection=config.action_selection,
        random_seed=config.seed,
        journal=rollout_journal(),
    )
    trajectory.metadata.update(
        {
            "model_path": config.model_paths[0],
            "model_interface": config.model_interface,
            "seed_row_index": rollout.seed_row_index,
            "seed_origin": seed_origin(
                config, identity["sequenceId"], identity["itemPosition"]
            ),
            "enforce_determinism": config.enforce_determinism,
            "seed": config.seed,
        }
    )
    from sequifier.trajectory.sequence_output import publish_inference_rollout

    publish_inference_rollout(trajectory, config)


@torch.no_grad()
def predict_policy(policy, frames, config, row_key):
    spec = policy.config.model_copy(update={"input_encoding": "model"})
    if config.output_path is not None and config.output_probabilities:
        from sequifier.infer import validate_probability_destinations

        validate_probability_destinations(
            config,
            Path(config.model_paths[0]).stem,
            {name: field.type for name, field in spec.actions.items()},
            1,
        )
    windows = policy_windows(frames, spec, config, row_key)
    generator = action_generator(policy.device, config.seed)
    outputs = []
    probability_rows = {
        name: [] for name, field in spec.actions.items() if field.type == "categorical"
    }
    while batch := list(islice(windows, config.inference_batch_size)):
        rows, histories, identities = zip(*batch)
        inputs = cast(SequifierInputs, collate_model_inputs(histories)).to(
            policy.device
        )
        distribution = policy.distribution(inputs, input_encoding="model")
        deterministic = config.action_selection == "deterministic"
        if config.output_probabilities:
            selected = distribution.select_and_score(
                deterministic, generator, include_probabilities=True
            )
            sampled = selected.decision
            scores, probabilities = selected.log_prob, selected.probabilities
        else:
            sampled = distribution.select_decision(deterministic, generator)
            scores, probabilities = None, {}
        actions = distribution.domain_actions(sampled)
        for index, (row_index, action) in enumerate(zip(rows, actions)):
            output = {
                row_key: row_index,
                **identities[index],
                **action,
            }
            if scores is not None:
                output["action_log_probability"] = float(scores[index])
            for name, values in probabilities.items():
                probability_rows[name].append((row_index, values[index].tolist()))
            outputs.append(output)
    predictions = pl.DataFrame(outputs).sort(row_key).drop(row_key)
    if config.output_probabilities:
        from sequifier.infer import inference_output_path

        for name, records in probability_rows.items():
            if not records:
                continue
            destination = inference_output_path(
                config.project_root,
                config.write_format,
                "probabilities",
                Path(config.model_paths[0]).stem,
                0,
                name,
                config=config,
            )
            write_data(
                probability_table(
                    spec.actions[name].values, [row for _, row in sorted(records)]
                ),
                destination,
                config.write_format,
            )
    return predictions


def policy_row_key(data, spec, config):
    reserved = (
        set(data.columns)
        | set(spec.actions)
        | {config.sequence_id_column, config.item_position_column}
        | {str(value) for field in spec.actions.values() for value in field.values}
    )
    key = "__policy_row"
    while key in reserved:
        key += "_"
    return key


def policy_windows(frames, spec, config, row_key):
    view = PolicyWindowView(spec)
    for row_index, history, identity in _decision_histories(
        frames, spec, config, row_key
    ):
        yield row_index, history_inputs(spec, history, window=view), identity


def _stored_context_frames(row, spec, config, sequence, position):
    context = CONTEXT_RECORDS.validate_json(row[CONTEXT_COLUMN])
    budget = desired_return(config, spec, row)
    static = tuple((name, row[name]) for name in spec.static_columns)
    frames = []
    for i, step in enumerate(reversed(context[-spec.context_length :]), 1):
        if budget is not None:
            budget = step.return_before(budget, spec.returns.gamma)
        frames.append(
            PolicyFrame(
                tuple(step.observation.items()),
                encode_action(step.previous_action, spec)
                if step.previous_action is not None
                else None,
                static,
                str(sequence),
                position - i,
                budget,
                step.masked,
            )
        )
    return reversed(frames)


def _lagged_action(row, prior, spec, config, position):
    if position and config.previous_action_columns:
        physical = row.get(DOMAIN_OBSERVATIONS) or {}
        lagged = {
            name: physical[column] if column in physical else row[column]
            for name, column in config.previous_action_columns.items()
        }
        if any(value is not None for value in lagged.values()):
            return encode_action(lagged, spec)
    if row.get(PREVIOUS_ACTION_RECORD):
        return encode_action(row[PREVIOUS_ACTION_RECORD], spec)
    if prior is not None:
        recorded = prior.get(ACTION_RECORD)
        if recorded and all(recorded.get(name) is not None for name in spec.actions):
            return encode_action({name: recorded[name] for name in spec.actions}, spec)
    return None


def _decision_histories(frames, spec, config, row_key, *, seed_index=None):
    offset = 0
    for data, context in frames:
        indexed = data.with_row_index(row_key, offset=offset)
        offset += len(data)
        if context is not None:
            indexed = pl.concat([context, indexed], how="diagonal_relaxed")
        for group in indexed.partition_by(
            config.sequence_id_column, maintain_order=True
        ):
            group = ordered_sequence_rows(group, config.item_position_column)
            if seed_index is not None:
                selected = group.filter(pl.col(row_key) == seed_index)
                if selected.is_empty():
                    continue
                group = (
                    selected
                    if config.rollout.seed_history == "current"
                    else group.filter(
                        pl.col(config.item_position_column)
                        <= selected[config.item_position_column][0]
                    ).tail(spec.context_length + 1)
                )
            history = deque(maxlen=spec.context_length)
            prior = None
            index = 0
            for row in group.iter_rows(named=True):
                position = row[config.item_position_column]
                previous_action = None
                if spec.previous_action_inputs and (
                    seed_index is None or config.rollout.seed_history != "current"
                ):
                    previous_action = _lagged_action(row, prior, spec, config, position)
                    if position and previous_action is None:
                        if row[row_key] is None:
                            prior = row
                            continue
                        if seed_index is None:
                            raise ValueError(
                                f"Missing preceding actions at position {position}; "
                                "fragment inference requires previous_action_columns"
                            )
                        history.clear()
                observation = {name: row[name] for name in spec.observation_columns}
                sequence = row[config.sequence_id_column]
                if row.get(CONTEXT_COLUMN) and (
                    seed_index is None or config.rollout.seed_history == "available"
                ):
                    if history:
                        raise ValueError(
                            "A policy seed context must precede the first decision"
                        )
                    history.extend(
                        _stored_context_frames(row, spec, config, sequence, position)
                    )
                history.append(
                    PolicyFrame(
                        tuple(observation.items()),
                        previous_action,
                        tuple((name, row[name]) for name in spec.static_columns),
                        str(sequence),
                        position,
                        desired_return(config, spec, row),
                        bool(row.get(OBSERVATION_MASK, False)),
                    )
                )
                prior = row
                if row[row_key] is None:
                    continue
                if seed_index is None or row[row_key] == seed_index:
                    yield (
                        row[row_key],
                        history,
                        {
                            "sequenceId": sequence,
                            "subsequenceId": index,
                            "windowStartOffset": 0,
                            "itemPosition": position,
                        },
                    )
                index += 1
