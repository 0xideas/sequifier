import json

import polars as pl
import pytest

from sequifier.config.preprocess_config import PreprocessorModel
from sequifier.config.sequence_import_config import PreparePolicyConfig
from sequifier.io.indexed_sequences import IndexedView
from sequifier.io.prepare_policy import prepare_policy
from sequifier.preprocess import Preprocessor
from sequifier.trajectory.sequence_output import (
    decision_row,
    publish_sequences,
    write_parts,
)


@pytest.mark.parametrize("categorical", [True, False])
def test_shared_rollout_preparation(tmp_path, categorical):
    actions = ["right", "left"] if categorical else [10.0, 20.0]
    rows = [
        decision_row(
            "episode",
            i,
            {"x": float(i), "tick": i},
            {},
            action={"x": actions[i % 2]},
            previous={"x": actions[(i - 1) % 2]} if i else None,
            reward=1.0,
            successor={"x": float(i + 1), "tick": i + 1},
            terminated=i == 9,
            truncated=False,
            masked=i == 1,
        )
        for i in range(10)
    ]
    manifest = tmp_path / "manifest.json"
    observations = ["x", "tick"]
    parts = list(write_parts(manifest, rows, observations=observations))
    publish_sequences(manifest, parts, observations, {"task": "policy"})
    settings = dict(
        run={"project_root": str(tmp_path)},
        split_method="value_cutoff",
        split_column="tick",
        splits={
            "train": {"cutoff": 6, "window_length": 3},
            "validation": {"window_length": 3},
        },
        normalize_real_columns=False,
    )
    config = PreparePolicyConfig(
        **settings,
        operation="prepare_policy",
        source={"manifest_path": str(manifest)},
        policy_binding={
            "actions": {
                "x": {"type": "categorical", "values": actions}
                if categorical
                else {"type": "real", "scale": {"fixed": 1.0}}
            }
        },
        indexed_output_path="policy",
    )
    prepare_policy(config)
    policy = IndexedView.load(tmp_path / "policy/0.split.json")
    metadata = json.loads(policy.metadata_path.read_text())
    target = "__executed_x"
    sequence = rows[0]["sequenceId"]
    expected = [
        metadata["id_maps"][target][v] if categorical else v for v in actions
    ] * 5
    assert policy.read(sequence)[target].to_list() == expected

    metadata["raw_mask_column"] = "hide"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(metadata))
    Preprocessor(
        PreprocessorModel(
            **settings,
            preprocessing_data_path=str(manifest),
            read_format="sequence",
            indexed_output_path="supervised",
            metadata_config_path=str(frozen),
            mask_column="hide",
            selected_columns=["x", target, "tick"],
        )
    ).run()
    supervised = IndexedView.load(tmp_path / "supervised/0.split.json")
    assert supervised.split_bounds() == policy.split_bounds()
    published = json.loads(supervised.metadata_path.read_text())
    assert (
        {k: metadata[k] for k in config.split_metadata()}
        == {k: published[k] for k in config.split_metadata()}
        == {
            "split_names": ["train", "validation"],
            "split_method": "value_cutoff",
            "split_ratios": None,
            "split_column": "tick",
            "split_values": [6],
        }
    )
    prepared = supervised.read(sequence)
    assert prepared["__observation_mask"].to_list() == [False, True] + [False] * 8
    assert prepared.filter(~pl.col("__observation_mask"))[target].to_list() == [
        expected[0],
        *expected[2:],
    ]

    # Frozen datasets written before the action alias existed still prepare successfully.
    for field in (
        "column_data_types",
        "id_maps",
        "selected_columns_statistics",
        "n_classes",
    ):
        metadata[field].pop(target, None)
    frozen.write_text(json.dumps(metadata))
    prepare_policy(
        config.model_copy(
            update={
                "indexed_output_path": "legacy",
                "metadata_config_path": str(frozen),
            }
        )
    )
    legacy = IndexedView.load(tmp_path / "legacy/0.split.json")
    assert legacy.read(sequence)[target].to_list() == expected
