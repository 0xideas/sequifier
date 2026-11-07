"""Behavioral checks for shared policy execution, loading, and preprocessing."""

from copy import deepcopy

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from sequifier.artifacts.model_artifact import build_model_artifact, save_model_artifact
from sequifier.artifacts.model_config import resolved_config_from_model_config
from sequifier.artifacts.model_export import model_execution_config
from sequifier.config.infer_config import ResolvedInferenceConfig, load_inferer_config
from sequifier.config.policy_infer_config import PolicyInferenceConfig
from sequifier.config.train_config import PolicyTrainConfig
from sequifier.infer import infer_worker
from sequifier.io.policy_dataset import PolicySequenceDataset
from sequifier.io.policy_window import PolicyWindowView
from sequifier.io.preprocessing import denormalize_values, normalize_values
from sequifier.io.pt_payload import StoredTensorBatch, save_pt_payload
from sequifier.io.sequence import SequenceEpisode, TransitionColumns
from sequifier.make import policy_train_config_string
from sequifier.model.factory import build_transformer_network
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.inference import infer_policy
from sequifier.rl.policy import CategoricalPolicy, policy_execution_config
from sequifier.rl.training import train_policy
from sequifier.rollout.preprocessing import FixedNumericPreprocessing


def config_at(root, epochs=2):
    values = yaml.safe_load(policy_train_config_string)
    values["project_root"] = str(root)
    values["global_training"].update(batch_size=2, log_interval=2)
    values["training_plan"]["phases"][0]["epochs"] = epochs
    return PolicyTrainConfig.model_validate(values)


def episodes(config, prefix):
    return PolicySequenceDataset(
        [
            SequenceEpisode(
                episode_id=f"{prefix}-{index}",
                item_positions=(0, 1, 2),
                observations={"OBSERVATION_COLUMN": (1.0 + index, 2.0, 3.0)},
                actions=(0.0, 1.0, 0.0),
                transitions=TransitionColumns(
                    rewards=(1.0, 1.0, 1.0), terminated=(False, False, True)
                ),
            )
            for index in range(2)
        ],
        config.policy,
    )


@pytest.mark.parametrize("encoding", ["global", "legacy"])
@pytest.mark.parametrize("conditioned", [True, False])
@pytest.mark.parametrize("collision", [True, False])
def test_policy_artifact_round_trip_keeps_predictions(
    tmp_path, encoding, conditioned, collision
):
    authored = config_at(tmp_path)
    schema = authored.policy.model_copy(
        update={
            "categorical_encoding": encoding,
            "return_conditioned": conditioned,
            "action_column": "OBSERVATION_COLUMN" if collision else "action",
        }
    )
    model = authored.model.model_copy(deep=True)
    if not conditioned:
        ingestion = model.interfaces["actor"].ingestion
        assert ingestion.type == "embedding"
        dims = ingestion.feature_embedding_dims
        assert dims is not None
        dims.pop("__policy_return")
        dims["__policy_previous_action"] += 1
    config = policy_execution_config(schema, model)
    network = build_transformer_network(config, device=torch.device("cpu")).network
    policy = CategoricalPolicy.from_network(network, config, "actor").eval()
    inputs = PolicyWindowView(schema).inputs(
        [{"OBSERVATION_COLUMN": 2.0}], [2], [3.0] if conditioned else None, {}
    )
    exported = model_execution_config(config)
    restored, _ = resolved_config_from_model_config(exported, device="cpu")
    assert model_execution_config(restored) == exported
    path = tmp_path / "policy.pt"
    save_policy(policy, path)
    with torch.no_grad():
        torch.testing.assert_close(
            load_policy(path)(inputs), policy(inputs), rtol=0, atol=0
        )


@pytest.mark.parametrize("boundary", ["epoch", "batch"])
def test_policy_resume_matches_uninterrupted_training(tmp_path, boundary):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        complete = config_at(tmp_path / "complete")
        complete.global_training.save_interval_batches = 1
        complete.global_training.save_interval_val_loss = False
        expected = train_policy(
            episodes(complete, "train"), episodes(complete, "val"), complete
        )
        if boundary == "epoch":
            partial = config_at(tmp_path / "resumed", epochs=1)
            train_policy(episodes(partial, "train"), episodes(partial, "val"), partial)
            values = deepcopy(partial.model_dump())
            values["global_training"]["resume"] = {"policy": "if_exists"}
        else:
            values = deepcopy(complete.model_dump())
            values["project_root"] = str(tmp_path / "resumed")
            checkpoint = next((tmp_path / "complete").rglob("*-epoch-1-batch-1.pt"))
            values["global_training"]["resume"] = {
                "policy": "required",
                "checkpoint_path": str(checkpoint),
            }
        values["training_plan"]["phases"][0]["epochs"] = 2
        resumed = PolicyTrainConfig.model_validate(values)
        actual = train_policy(
            episodes(resumed, "train"), episodes(resumed, "val"), resumed
        )
        assert actual.metrics == expected.metrics
        assert actual.optimizer_steps == expected.optimizer_steps
        assert actual.best_epoch == expected.best_epoch
        for name, tensor in expected.policy.network.state_dict().items():
            torch.testing.assert_close(
                actual.policy.network.state_dict()[name], tensor, rtol=0, atol=0
            )
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("checkpoint", [True, False])
def test_policy_inference_reads_artifact_once(tmp_path, monkeypatch, checkpoint):
    authored = config_at(tmp_path)
    config = policy_execution_config(authored.policy, authored.model)
    network = build_transformer_network(config, device=torch.device("cpu")).network
    path = tmp_path / "policy.pt"
    save_policy(CategoricalPolicy.from_network(network, config, "actor"), path)
    if checkpoint:
        artifact = build_model_artifact(network, config)
        torch.save(
            {
                "artifact_type": "sequifier_run_checkpoint",
                "model": artifact.state_dict(),
            },
            path,
        )
    pl.DataFrame({"OBSERVATION_COLUMN": [1.0, 2.0]}).write_csv(tmp_path / "data.csv")
    inference = tmp_path / "infer.yaml"
    inference.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "model_path": "policy.pt",
                "data_path": "data.csv",
                "desired_return": 3.0,
                "output_path": "predictions.csv",
                "output_probabilities": True,
            }
        )
    )
    reads = []
    original_load = torch.load

    def counted_load(*args, **kwargs):
        reads.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    resolved = load_inferer_config(str(inference), {}, False)
    assert isinstance(resolved, PolicyInferenceConfig)
    infer_policy(resolved)
    assert len(reads) == 1
    assert "_artifacts" not in resolved.model_dump()
    predictions = pl.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == 2
    assert set(predictions["action"]) <= {0.0, 1.0}
    np.testing.assert_allclose(
        predictions["action__probability_0"] + predictions["action__probability_1"],
        1.0,
        rtol=1e-6,
    )


def test_shared_normalization_preserves_padding_and_zero_scale():
    statistics = {"x": {"mean": 7.0, "std": 0.0}}
    values = np.array([7.0, 7.000000001], dtype=np.float64)
    expected = normalize_values(values, statistics["x"])
    np.testing.assert_allclose(denormalize_values(expected, statistics["x"]), values)
    torch.testing.assert_close(
        normalize_values(torch.from_numpy(values), statistics["x"]),
        torch.from_numpy(expected),
    )
    frame = pl.DataFrame({"x": values}).select(
        normalize_values(pl.col("x"), statistics["x"])
    )
    np.testing.assert_allclose(frame["x"], expected)
    codec = FixedNumericPreprocessing(["x"], statistics, True, {"x": "float64"})
    seed = codec.seed_from_model(
        {"x": np.array([[0.0, expected[1]]])}, np.array([[False, True]]), ["x"], []
    )
    features, metadata = codec.model_inputs(seed)
    assert features["x"][0, 0] == 0.0
    assert not metadata["attention_valid_mask"][0, 0]
    np.testing.assert_allclose(features["x"][0, 1], expected[1])


def test_generative_inference_reads_artifact_once(tmp_path, monkeypatch):
    authored = config_at(tmp_path)
    config = policy_execution_config(authored.policy, authored.model)
    config.interface().policy = None
    config.model.interfaces["actor"].policy = None
    network = build_transformer_network(config, device=torch.device("cpu")).network
    save_model_artifact(build_model_artifact(network, config), tmp_path / "model.pt")
    payload = StoredTensorBatch(
        sequences={
            "OBSERVATION_COLUMN": torch.ones(1, 5),
            "__policy_previous_action": torch.full((1, 5), 3, dtype=torch.int64),
            "__policy_return": torch.ones(1, 5),
            "action": torch.full((1, 5), 3, dtype=torch.int64),
        },
        sequence_ids=torch.zeros(1, dtype=torch.int64),
        subsequence_ids=torch.zeros(1, dtype=torch.int64),
        start_item_positions=torch.zeros(1, dtype=torch.int64),
        left_pad_lengths=torch.zeros(1, dtype=torch.int64),
        split_start_item_positions=torch.zeros(1, dtype=torch.int64),
        split_end_item_positions=torch.full((1,), 5, dtype=torch.int64),
    )
    save_pt_payload(payload, tmp_path / "data.pt")
    inference = tmp_path / "infer.yaml"
    inference.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "model_path": "model.pt",
                "data_path": "data.pt",
                "model_type": "generative",
                "read_format": "pt",
                "device": "cpu",
            }
        )
    )
    reads = []
    original_load = torch.load

    def counted_load(path, *args, **kwargs):
        if str(path).endswith("model.pt"):
            reads.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    monkeypatch.setattr(torch, "compile", lambda model: model)
    resolved = load_inferer_config(str(inference), {}, False)
    assert isinstance(resolved, ResolvedInferenceConfig)
    metadata = resolved.dataset_metadata
    assert metadata is not None
    infer_worker(
        resolved, {}, metadata.id_maps, metadata.selected_columns_statistics, None, True
    )
    assert len(reads) == 1
    predictions = pl.read_csv(
        tmp_path / "outputs" / "predictions" / "model" / "part-000.csv"
    )
    assert len(predictions) > 0
