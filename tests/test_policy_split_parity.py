import json

import polars as pl
import pytest

from sequifier.config.policy_train_config import SequenceSourceConfig
from sequifier.config.preprocess_config import PreprocessorModel
from sequifier.config.sequence_import_config import PreparePolicyConfig
from sequifier.io.indexed_sequences import IndexedView
from sequifier.io.policy_dataset import EpisodeDecisions, episode_partition_indices
from sequifier.io.prepare_policy import prepare_policy
from sequifier.preprocess import Preprocessor
from sequifier.trajectory.importers import source_episodes
from sequifier.trajectory.schema import Trajectory
from sequifier.trajectory.sequence_output import (
    manifest_values,
    trajectory_rows,
    write_parts,
)
from sequifier.trajectory.store import save_trajectory


@pytest.mark.parametrize("grouped", [False, True])
def test_between_sequence_membership_across_modes(tmp_path, grouped):
    ids = list(reversed(range(10)))
    table = pl.DataFrame(
        [
            {
                "sequenceId": seq,
                "itemPosition": pos,
                "x": float(seq + pos),
                "action": "left",
                **({"__group": "pair" if seq < 2 else None} if grouped else {}),
            }
            for seq in ids
            for pos in range(4)
        ]
    )
    path = tmp_path / "events.parquet"
    table.write_parquet(path)
    common = dict(
        split_method="between_sequence",
        splits={
            "train": dict(fraction=0.8, window_length=2),
            "validation": dict(fraction=0.2, window_length=2),
        },
        run={"project_root": str(tmp_path), "reproducibility": {"seed": 1010}},
        real_columns=["x"],
    )
    Preprocessor(
        PreprocessorModel(
            **common,
            preprocessing_data_path=str(path),
            indexed_output_path="supervised",
            selected_columns=["x"],
            read_format="parquet",
        )
    ).run()
    config = PreparePolicyConfig(
        **common,
        operation="prepare_policy",
        output_path="policy",
        source={"paths": [str(path)], "columns": {"observations": ["x"]}},
        policy_binding={
            "actions": {"action": {"type": "categorical", "values": ["left", "right"]}}
        },
    )
    prepare_policy(config)
    source = config.resolved_source()
    decisions = EpisodeDecisions([source], config.resolved_policy(source))
    direct = episode_partition_indices(decisions, seed=1010, group_identical=True)
    for split in range(2):
        view = f"{split}.split.json"
        expected = set(IndexedView.load(tmp_path / "supervised" / view).split_bounds())
        assert (
            set(IndexedView.load(tmp_path / "policy" / view).split_bounds()) == expected
        )
        assert {decisions.sequence_ids[i] for i in direct[split]} == expected
        if split == 0:
            policy_stats = (
                IndexedView.load(tmp_path / "policy" / view)
                .metadata()
                .selected_columns_statistics
            )
            supervised_stats = (
                IndexedView.load(tmp_path / "supervised" / view)
                .metadata()
                .selected_columns_statistics
            )
            assert policy_stats["x"] == pytest.approx(supervised_stats["x"])
        if grouped:
            assert (0 in expected) == (1 in expected)
        elif split == 1:
            assert expected == {6, 7, 9}


def test_rollout_json_and_table_share_seed_group(tmp_path):
    from sequifier.config.policy_config import ResolvedPolicyBinding

    policy = ResolvedPolicyBinding(
        actions={"action": {"type": "real", "scale": {"fixed": 1.0}}},
        action_order=["action"],
        observation_columns=["x"],
        interface_name="actor",
        context_length=1,
    )
    groups = []
    for i, pos in enumerate([8, 8, 9, None]):
        identity = str(i)
        trajectory = Trajectory.model_validate(
            dict(
                format="sequifier.numeric-forecast-trajectory",
                episode_id=identity,
                seed=dict(
                    observations={"x": [0.0]},
                    attention_valid_mask=[True],
                    static_inputs={},
                ),
                bindings=[dict(input="x", source="prediction", key="action")],
                prediction_columns=["action"],
                transitions=[
                    dict(
                        episode_id=identity,
                        step_id=0,
                        observation={"x": 0.0},
                        predictions={"action": float(i)},
                        external_inputs={},
                        reward=1.0,
                        successor_observation={"x": float(i)},
                        terminated=True,
                        truncated=False,
                        bootstrap_allowed=False,
                    )
                ],
                collection_cutoff=False,
                total_return=1.0,
                metadata={
                    "seed_origin": {
                        "dataset": "events",
                        "sequenceId": 2,
                        "itemPosition": pos,
                    }
                }
                if pos is not None
                else {},
            )
        )
        path = tmp_path / f"{i}.json"
        save_trajectory(trajectory, path)
        manifest = tmp_path / f"{i}.manifest.json"
        parts = list(write_parts(manifest, trajectory_rows(trajectory)))
        manifest.write_text(
            json.dumps(
                manifest_values(
                    parts, ["x"], {"task": "forecast", "prediction_columns": ["action"]}
                )
            )
        )

        def imported_group(**location):
            source = SequenceSourceConfig(
                **location, action_semantics="executed_predictions"
            )
            (reference,) = source_episodes(source, policy)
            return reference.load().group_id

        group = imported_group(paths=[str(path)])
        assert group == imported_group(manifest_path=str(manifest))
        groups.append(group)
    assert groups[0] == groups[1]
    assert len({groups[0], groups[2], None}) == 3
    assert groups[3] is None


def test_duplicate_content_keeps_linked_episodes_together():
    from types import SimpleNamespace

    from sequifier.io.policy_dataset import episode_split_assignments

    ids = list(range(10))
    groups = ["linked", "linked", *([None] * 8)]
    data = SimpleNamespace(
        sequence_ids=ids, group_ids=groups, demonstration_keys=ids.copy()
    )
    initial = episode_split_assignments(data, [0.8, 0.2], 1010, group_identical=True)
    other = next(i for i in ids if initial[i] != initial[0])
    data.demonstration_keys[other] = data.demonstration_keys[1]
    result = episode_split_assignments(data, [0.8, 0.2], 1010, group_identical=True)
    assert result[0] == result[1] == result[other]
    assert set(result) == {0, 1}
