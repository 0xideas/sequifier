"""CLI policy prediction and environment execution using the shared model lifecycle."""

from collections import deque
from contextlib import contextmanager
from itertools import chain, islice
from pathlib import Path
from typing import cast

import polars as pl
import torch
from loguru import logger

from sequifier.config.policy_infer_config import ResolvedPolicyInferenceConfig
from sequifier.helpers import configure_determinism, normalize_path
from sequifier.io.actions import encode_action
from sequifier.io.batch import SequifierInputs, collate_model_inputs
from sequifier.io.inference_workspace import rollout_journal
from sequifier.io.inference_workspace import write_inference_data as write_data
from sequifier.io.inference_workspace import write_inference_trajectory
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.sequence import ordered_sequence_rows, validate_sequence_ids
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
    model_decoder,
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
    with _policy_input_frames(policy, config) as (frames, prepared):
        first = next((frame for frame in frames if len(frame[0])), None)
        if first is None:
            raise ValueError("policy inference input must be nonempty")
        first_rows, _ = first
        if spec.previous_action_inputs and not config.rollout:
            required.update(config.previous_action_columns.values())
        if missing := required - set(first_rows.columns):
            raise ValueError(f"policy inference is missing columns: {sorted(missing)}")
        frames = chain((first,), frames)
        if config.rollout is not None:
            offset = config.rollout.seed_row_index
            for frame, _ in frames:
                if offset < len(frame):
                    infer_rollout(policy, frame.row(offset, named=True), config)
                    return
                offset -= len(frame)
            raise ValueError("policy rollout seed_row_index is outside the input rows")
        if (
            prepared
            and config.sequence_id_column != "sequenceId"
            and config.sequence_id_column in first_rows.columns
        ):
            # A user-defined grouping key can span physical sequences.
            combined = pl.concat([rows for rows, _ in frames], how="diagonal_relaxed")
            frames = iter(((combined, None),))
        predictions = predict_policy(
            policy,
            frames,
            config,
            policy_row_key(first_rows, spec, config),
            prepared=prepared,
        )
    if config.output_path is None:
        from sequifier.infer import inference_output_path

        destination = inference_output_path(
            config.project_root,
            config.write_format,
            "predictions",
            Path(config.model_paths[0]).stem,
            0,
            config=config,
        )
    else:
        destination = normalize_path(config.output_path, config.project_root)
        config.validate_output_destination(destination, label="output_path")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_data(predictions, destination, config.write_format)
    logger.info(f"Wrote {len(predictions)} policy predictions to '{destination}'")


@contextmanager
def _policy_input_frames(policy, config):
    spec = policy.config
    if config.input_layout == "raw":
        from sequifier.io.prepare_inference import fresh_view_from_raw

        uses_feedback = bool(spec.previous_action_inputs) and config.rollout is None
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
            passthrough_columns=auxiliary,
            optional_passthrough=spec.actions if uses_feedback else (),
        ) as view:
            yield _indexed_policy_frames(view, spec, config), True
    elif config._indexed_view is not None:
        yield _indexed_policy_frames(config._indexed_view, spec, config), True
    elif config._prepared_source is not None:
        yield _stored_policy_frames(config._prepared_source, spec, config), True
    else:
        path = normalize_path(config.data_path, config.project_root)
        data = (
            pl.read_csv(path) if config.read_format == "csv" else pl.read_parquet(path)
        )
        _validate_policy_coordinates(data, config)
        yield iter(((data, None),)), False


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
    decode = model_decoder(metadata, spec.actions)
    raw_columns = set(metadata.indexed_passthrough)
    if conflicts := raw_columns.intersection(
        spec.observation_columns + spec.static_columns
    ):
        raise ValueError(
            f"Policy model inputs are stored as raw auxiliary columns: {sorted(conflicts)}"
        )
    if view.view["view"] == "predict" and not config.rollout:
        required_raw = {
            name
            for name in spec.previous_action_inputs.values()
            if name in metadata.column_data_types
            and (spec.actions[name].type == "real" or spec.raw_mask_column is not None)
        }
        if config.previous_action_columns:
            required_raw = set(config.previous_action_columns.values())
        if missing := required_raw - raw_columns:
            raise ValueError(
                f"Prepare lagged action columns with passthrough_columns: {sorted(missing)}"
            )
    static = pl.read_parquet(view.static_path) if view.static_path else None
    media = None if metadata.indexed_inline_media else view.image_store()
    bounds = view.split_bounds()
    predict = view.view["view"] == "predict"
    for key in sorted(view.dataset["sequences"], key=int):
        sequence = int(key)
        if not predict and sequence not in bounds:
            continue
        start, end = bounds.get(sequence, (None, None))
        read_start = start - 1 if _use_predecessor(start, spec, config) else start
        rows = view.read(sequence, read_start, end)
        rows, previous = _prepared_frame(
            rows, sequence, start, static, media, metadata, decode, spec, config
        )
        yield rows, previous


def _use_predecessor(start, spec, config):
    return (
        start is not None
        and start > 0
        and spec.previous_action_inputs
        and not config.rollout
        and not config.previous_action_columns
        and config.sequence_id_column == "sequenceId"
        and config.item_position_column == "itemPosition"
    )


def _stored_policy_frames(source, spec, config):
    from sequifier.io.images import ImageStore
    from sequifier.trajectory.prepared import prepared_table
    from sequifier.trajectory.tables import scan_tables

    metadata = source.metadata
    decode = model_decoder(metadata, spec.actions)
    media = (
        ImageStore(
            metadata.image_sources,
            metadata.preconditioning_image,
            str(source.project_root),
        )
        if metadata.image_sources or metadata.preconditioning_image
        else None
    )
    static = (
        scan_tables((source.static_path,)).collect() if source.static_path else None
    )
    for rows in prepared_table(source).partition_by("sequenceId", maintain_order=True):
        decisions = rows.filter(pl.col("__decision"))
        if decisions.is_empty():
            continue
        start = decisions["itemPosition"][0]
        if _use_predecessor(start, spec, config):
            decisions = rows.filter(
                pl.col("__decision") | (pl.col("itemPosition") == start - 1)
            )
        yield _prepared_frame(
            decisions.drop("__decision"),
            decisions["sequenceId"][0],
            start,
            static,
            media,
            metadata,
            decode,
            spec,
            config,
        )


def _prepared_frame(
    rows, sequence, start, static, media, metadata, decode, spec, config
):
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
    domain_columns = (
        set(spec.actions)
        if not config.rollout and ACTION_RECORD not in rows.columns
        else set()
    )
    if config.rollout is not None and spec.input_encoding == "raw":
        domain_columns.update(spec.observation_columns + spec.static_columns)
    encoded_columns = set(metadata.column_data_types) - set(
        metadata.indexed_passthrough
    )
    for column in domain_columns.intersection(rows.columns, encoded_columns):
        rows = rows.with_columns(
            pl.Series(
                column, [decode(value, column) for value in rows[column].to_list()]
            )
        )
    previous = None
    if start is not None and rows["itemPosition"][0] < start:
        previous = rows.row(0, named=True)
        rows = rows.slice(1)
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


def infer_rollout(policy, row, config):
    spec = policy.config
    rollout = config.rollout
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
        journal=rollout_journal(),
    )
    trajectory.metadata.update(
        {
            "model_path": config.model_paths[0],
            "model_interface": config.model_interface,
            "seed_row_index": rollout.seed_row_index,
            "enforce_determinism": config.enforce_determinism,
            "seed": config.seed,
        }
    )
    destination = normalize_path(rollout.trajectory_path, config.project_root)
    config.validate_output_destination(destination, label="trajectory_path")
    write_inference_trajectory(trajectory, destination)
    logger.info(
        f"Wrote {len(trajectory.transitions)} policy transitions to '{destination}'"
    )


@torch.no_grad()
def predict_policy(policy, frames, config, row_key, *, prepared=False):
    spec = policy.config
    if prepared:
        spec = spec.model_copy(update={"input_encoding": "model"})
    if config.output_path is not None and config.output_probabilities:
        from sequifier.infer import validate_probability_destinations

        validate_probability_destinations(
            config,
            Path(config.model_paths[0]).stem,
            {name: field.type for name, field in spec.actions.items()},
            1,
        )
    windows = policy_windows(frames, spec, config, row_key, prepared=prepared)
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
        distribution = policy.distribution(
            inputs, input_encoding="model" if prepared else None
        )
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
                probability_rows[name].append(
                    {
                        row_key: row_index,
                        **{
                            str(label): float(values[index][action_id])
                            for action_id, label in enumerate(spec.actions[name].values)
                        },
                    }
                )
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
                pl.DataFrame(records).sort(row_key).drop(row_key),
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


def policy_windows(frames, spec, config, row_key, *, prepared):
    view = PolicyWindowView(spec)
    offset = 0
    for data, predecessor in frames:
        indexed = data.with_row_index(row_key, offset=offset)
        offset += len(data)
        groups = (
            indexed.partition_by(config.sequence_id_column, maintain_order=True)
            if config.sequence_id_column in indexed.columns
            else indexed.iter_slices(1)
        )
        for group in groups:
            if config.item_position_column not in group.columns:
                group = group.with_row_index(config.item_position_column)
            group = ordered_sequence_rows(group, config.item_position_column)
            history = deque(maxlen=spec.context_length)
            prior = predecessor
            for index, row in enumerate(group.iter_rows(named=True)):
                position = row[config.item_position_column]
                previous_action = None
                if spec.previous_action_inputs:
                    lagged = (
                        {
                            name: row[column]
                            for name, column in config.previous_action_columns.items()
                        }
                        if position != 0
                        else {}
                    )
                    if any(value is not None for value in lagged.values()):
                        previous_action = encode_action(lagged, spec)
                    elif prior is not None:
                        recorded = (
                            prior.get(ACTION_RECORD, prior) if prepared else prior
                        )
                        if missing := spec.actions.keys() - recorded.keys():
                            raise ValueError(
                                f"Missing preceding actions {sorted(missing)} at position {position}; "
                                "supply previous_action_columns for this decision"
                            )
                        previous_action = encode_action(
                            {name: recorded[name] for name in spec.actions}, spec
                        )
                    elif prepared and position != 0 and row.get(PREVIOUS_ACTION_RECORD):
                        previous_action = encode_action(
                            row[PREVIOUS_ACTION_RECORD], spec
                        )
                    elif position != 0:
                        raise ValueError(
                            "fragment inference requires previous_action_columns"
                        )
                observation = {name: row[name] for name in spec.observation_columns}
                if not prepared and spec.raw_mask_column in row:
                    observation[spec.raw_mask_column] = row[spec.raw_mask_column]
                sequence = row.get(config.sequence_id_column, row[row_key])
                history.append(
                    PolicyFrame(
                        tuple(observation.items()),
                        previous_action,
                        tuple((name, row[name]) for name in spec.static_columns),
                        str(sequence),
                        position,
                        desired_return(config, spec, row),
                    )
                )
                yield (
                    row[row_key],
                    history_inputs(spec, history, window=view),
                    {
                        "sequenceId": sequence,
                        "subsequenceId": index,
                        # Each policy subsequence materializes exactly one model window.
                        "windowStartOffset": 0,
                        "itemPosition": position,
                    },
                )
                prior = row
