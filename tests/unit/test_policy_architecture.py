"""Behavioral coverage for resolved policy contracts, data ownership and recovery."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
import torch
import yaml

from sequifier.config.metadata import DatasetMetadata
from sequifier.config.policy_plan import PolicyImitationPhaseConfig
from sequifier.config.train_config import (
    GlobalTrainingSpecModel,
    ModelInterfaceSpecModel,
    PolicyRunConfig,
    _resolve_interface,
)
from sequifier.io.partitioning import split_components
from sequifier.io.policy_window import PolicyWindowView, action_global_ids
from sequifier.rl import run_checkpoint
from sequifier.rl.checkpoint import load_policy, save_policy
from sequifier.rl.collection import decide
from sequifier.rl.distributions import action_generator
from sequifier.rl.dpo import dpo_loss
from sequifier.rl.learner import run_policy_training
from sequifier.rl.models import HistoryCritic
from sequifier.rl.preparation import (
    build_actor,
    input_identity,
    prepare_data,
    validate_data,
)
from sequifier.rl.run_checkpoint import RunCheckpoint
from sequifier.rl.run_state import weight_fingerprint
from sequifier.trajectory import importers

EXAMPLES = Path(__file__).resolve().parents[2] / "documentation/example_inputs"


@pytest.fixture(scope="module")
def run_values(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("policy")
    runpy.run_path(str(EXAMPLES / "policy_control_data.py"))["generate"](root / "data")
    path = root / "data/control-decisions.csv"
    data = pl.read_csv(path)
    data.filter(
        pl.col("contextId").is_in([f"context-{i}" for i in range(4)])
    ).write_csv(path)
    path = root / "data/control-preferences.json"
    preferences = json.loads(path.read_text())
    preferences["pairs"] = preferences["pairs"][:4]
    path.write_text(json.dumps(preferences))
    values = yaml.safe_load(
        (EXAMPLES / "configs/train-multiphase-control.yaml").read_text()
    )
    values.update(
        project_root=str(root),
        context_length=2,
        episode_cache_size=1,
        resume={"policy": "never", "mode": "exact"},
    )
    values["model"]["backbone"]["architecture"].update(
        dim_model=8,
        max_context_length=2,
        attention={"n_heads": 2},
        feed_forward={"dim": 16},
    )
    for phase in values["phases"]:
        phase.update(evaluation_interval=1, checkpoint_interval=1)
        if phase["algorithm"] == "policy_imitation":
            phase.update(epochs=1, batch_size=8)
        elif phase["algorithm"] == "iql":
            phase.update(updates=2, batch_size=4)
        elif phase["algorithm"] == "dpo":
            phase.update(updates=2, batch_size=2, scoring_chunk_size=3)
        elif phase["algorithm"] == "grpo":
            phase.update(
                total_groups=2,
                groups_per_update=1,
                group_size=2,
                max_steps=2,
                optimization_epochs=1,
                batch_size=2,
            )
        elif phase["algorithm"] == "ppo":
            phase.update(
                total_env_steps=4,
                rollout_length=2,
                optimization_epochs=1,
                minibatch_size=2,
            )
        if "environment" in phase:
            phase["environment"]["config"] = {"horizon": 2}
            evaluation = phase["evaluation"]
            evaluation.update(seeds=[17], max_episode_steps=2)
            evaluation["environment"]["config"] = {"horizon": 2}
    return values


def config_for(values, name, *, resume="never"):
    values = deepcopy(values)
    values["model_name"] = name
    values["resume"]["policy"] = resume
    return PolicyRunConfig.model_validate(values)


@pytest.fixture(scope="module")
def completed(run_values):
    config = config_for(run_values, "uninterrupted")
    return config, run_policy_training(config)


def test_action_vocabulary_and_portable_roundtrip(run_values, tmp_path):
    config = config_for(run_values, "vocabulary")
    actor = build_actor(config, config.policy, {"cue": {"mean": 0.0, "std": 1.0}})
    target = actor.interface.target_columns[0]
    assert (
        actor.interface.target_decoder_ids[target]
        == action_global_ids(actor.config)
        == [3, 4]
    )
    assert actor.interface.target_global_to_decoder[target] == [-1, -1, -1, 0, 1]
    assert actor.interface.target_n_classes[target] == 2
    assert actor.interface.categorical_decoder_special_tokens == {target: []}
    history = PolicyWindowView(actor.config).inputs([{"cue": 1.0}], [2], None, {})
    logits = actor(history)
    assert logits.shape[-1] == 2
    path = tmp_path / "actor.pt"
    save_policy(actor, path)
    restored = load_policy(path)
    assert torch.equal(restored(history), logits)
    # Validate a decision once, sharing the checked transforms with the critic.
    critic = HistoryCritic(deepcopy(actor.execution_config), actor.interface_name, None)
    actor.validate_inputs = Mock(wraps=actor.validate_inputs)
    decide(actor, critic, history, action_generator(actor.device, 1))
    actor.validate_inputs.assert_called_once()
    broken = deepcopy(history)
    broken.inputs.pop("cue")
    with pytest.raises(ValueError, match="persisted named feature schema"):
        decide(actor, None, broken, action_generator(actor.device, 1))


def test_supervised_decoder_defaults_stay_unchanged():
    spec = ModelInterfaceSpecModel.model_validate(
        {
            "input_columns": ["x"],
            "target_columns": ["y"],
            "ingestion": {"type": "embedding", "output_dim": 4},
            "decoder": {"type": "linear", "prediction_length": 1},
        }
    )
    metadata = DatasetMetadata(
        window_length=3,
        column_data_types={"x": "Float32", "y": "Int64"},
        n_classes={"y": 5},
        selected_columns_statistics={"x": {"mean": 2.0, "std": 3.0}},
    )
    training = GlobalTrainingSpecModel(
        training_objective="causal",
        inference_batch_size=1,
        context_length=2,
        batch_size=1,
        learning_rate=1e-3,
    )
    resolved = _resolve_interface(
        "forecast", spec, metadata, training, metadata.storage_layout
    )
    assert resolved.policy is None
    assert resolved.target_decoder_ids == {"y": [1, 3, 4]}
    assert resolved.categorical_decoder_special_tokens == {"y": ["other"]}
    assert resolved.selected_columns_statistics == metadata.selected_columns_statistics


def test_shared_corpus_transitions_and_bounded_access(run_values, monkeypatch):
    scan = Mock(wraps=importers._scan)
    monkeypatch.setattr(importers, "_scan", scan)
    from sequifier.io.iql_dataset import IQLTransitionDataset

    initialize = IQLTransitionDataset.__init__
    calls = []

    def count(self, dataset):
        calls.append(dataset)
        initialize(self, dataset)

    monkeypatch.setattr(IQLTransitionDataset, "__init__", count)
    config = config_for(run_values, "views")
    data = prepare_data(config, config.policy)
    assert data[0] is data[1]
    assert scan.call_count == 1
    assert len(calls) == 2
    actor = build_actor(config, config.policy, {"cue": {"mean": 0.0, "std": 1.0}})
    validate_data(config, data, actor)
    assert len(calls) == 2
    transitions = data[1].prepare_transitions()
    assert transitions is data[1].prepare_transitions()
    training, validation = data[0].training, data[0].validation
    assert set(training.group_ids).isdisjoint(validation.group_ids)
    assert set(training.demonstration_keys).isdisjoint(validation.demonstration_keys)
    for part in (training, validation):
        for index in range(len(part)):
            part[index]
        for corpus, _ in part.episodes.entries:
            assert len(corpus.cache) <= config.episode_cache_size
    assert scan.call_count == 1
    order = list(training.sampler(torch.Generator().manual_seed(1)))
    assert sorted(order) == list(range(len(training)))
    episode_indices = [
        next(i for i, end in enumerate(training.cumulative_lengths) if decision < end)
        for decision in order
    ]
    assert len(
        [
            i
            for i, e in enumerate(episode_indices)
            if i == 0 or e != episode_indices[i - 1]
        ]
    ) == len(training.episodes)


def test_transitive_components_are_never_split():
    keys = {0: ["a"], 1: ["a", "b"], 2: ["b"], 3: ["c"], 4: ["d"]}
    training, validation = split_components(keys, keys.__getitem__, 0.4, 17)
    assert {0, 1, 2} <= set(training) or {0, 1, 2} <= set(validation)
    assert (training, validation) == split_components(keys, keys.__getitem__, 0.4, 17)
    with pytest.raises(ValueError, match="independent components"):
        split_components([0, 1, 2], keys.__getitem__, 0.5, 1)


@pytest.mark.parametrize("role", ["actor", "value", "critic"])
def test_optimizer_roles_share_validation(run_values, role):
    values = deepcopy(run_values["phases"][1])
    values[f"{role}_optimizer"] = {"name": "Adam", "arguments": {"lr": 1e-3}}
    with pytest.raises(ValueError, match="phase learning rate"):
        config_for({**run_values, "phases": [values]}, f"bad-{role}")
    values.pop(f"{role}_optimizer")
    values[f"{role}_scheduler"] = {
        "name": "OneCycleLR",
        "arguments": {"total_steps": 3, "max_lr": 0.001},
    }
    with pytest.raises(ValueError, match="budget"):
        config_for({**run_values, "phases": [values]}, f"bad-budget-{role}")


def test_imitation_scheduler_budget_uses_actual_partition():
    phase = PolicyImitationPhaseConfig.model_validate(
        {
            "name": "imitation",
            "algorithm": "policy_imitation",
            "sources": ["data"],
            "epochs": 2,
            "batch_size": 3,
            "accumulation_steps": 2,
            "actor_scheduler": {
                "name": "OneCycleLR",
                "arguments": {"total_steps": 4, "max_lr": 0.001},
            },
        }
    )
    assert phase.optimization_plans(10)["actor"].step_budget == 4
    with pytest.raises(ValueError, match="budget"):
        phase.optimization_plans(20)


def test_completed_run_resume_and_snapshot_deduplication(run_values, completed):
    config, result = completed
    resumed = run_policy_training(
        config_for(run_values, config.model_name, resume="required")
    )
    assert weight_fingerprint(
        resumed.policy.network.state_dict()
    ) == weight_fingerprint(result.policy.network.state_dict())
    assert resumed.phases == result.phases
    payload = torch.load(config.checkpoint_path, weights_only=False)
    assert payload["incoming_actor"] == result.phases[-1].selected_weight_hash
    assert len(payload["snapshots"]) <= len(result.phases)
    assert payload["phase_state"] is None
    assert (
        not {"phase_index", "status", "last_committed_transaction", "selected_actors"}
        & payload.keys()
    )


@pytest.mark.parametrize("phase_index", range(5))
def test_interrupted_resume_matches_uninterrupted(
    run_values, completed, monkeypatch, phase_index
):
    config = config_for(run_values, f"interrupt-{phase_index}")
    save = run_checkpoint.atomic_save

    def interrupt(path, payload):
        save(path, payload)
        if (
            len(payload["completed_results"]) == phase_index
            and payload["phase_state"] is not None
        ):
            raise InterruptedError("simulated interruption after committed update")

    with monkeypatch.context() as patch:
        patch.setattr(run_checkpoint, "atomic_save", interrupt)
        with pytest.raises(InterruptedError):
            run_policy_training(config)
    recovered = run_policy_training(
        config_for(run_values, config.model_name, resume="required")
    )
    assert [p.selected_weight_hash for p in recovered.phases] == [
        p.selected_weight_hash for p in completed[1].phases
    ]
    assert weight_fingerprint(
        recovered.policy.network.state_dict()
    ) == weight_fingerprint(completed[1].policy.network.state_dict())


def test_handoff_resume_republishes_missing_artifacts(
    run_values, completed, monkeypatch
):
    config = config_for(run_values, "handoff")
    save = run_checkpoint.atomic_save

    def interrupt(path, payload):
        save(path, payload)
        if len(payload["completed_results"]) == 1 and payload["phase_state"] is None:
            raise InterruptedError("simulated interruption before publication")

    with monkeypatch.context() as patch:
        patch.setattr(run_checkpoint, "atomic_save", interrupt)
        with pytest.raises(InterruptedError):
            run_policy_training(config)
    assert not (Path(config.phase_directory) / "000-imitation/actor.pt").exists()
    result = run_policy_training(
        config_for(run_values, config.model_name, resume="required")
    )
    assert (Path(config.phase_directory) / "000-imitation/actor.pt").is_file()
    assert [p.selected_weight_hash for p in result.phases] == [
        p.selected_weight_hash for p in completed[1].phases
    ]


@pytest.mark.parametrize("corruption", ["weights", "result", "boundary"])
def test_corrupt_checkpoint_is_rejected(run_values, completed, monkeypatch, corruption):
    original_config, _ = completed
    config = config_for(run_values, original_config.model_name, resume="required")
    payload = torch.load(config.checkpoint_path, weights_only=False)
    if corruption == "weights":
        weights = next(iter(payload["snapshots"].values()))
        next(iter(weights.values())).fill_(float("nan"))
    elif corruption == "result":
        payload["completed_results"][0]["index"] = 1
    else:
        payload["phase_state"] = {"version": 1}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    plan = config.model_dump(mode="json", exclude={"resume"})
    with pytest.raises(ValueError):
        RunCheckpoint.load(config.checkpoint_path, config, plan, input_identity(config))


def test_dpo_objective_retains_gradient_semantics():
    chosen = torch.tensor([1.5], requires_grad=True)
    loss, margin = dpo_loss(
        chosen, torch.tensor([0.5]), torch.tensor([0.3]), torch.tensor([0.1]), 0.2
    )
    expected = -torch.nn.functional.logsigmoid(0.2 * ((chosen - 0.5) - (0.3 - 0.1)))
    assert torch.equal(loss, expected)
    loss.sum().backward()
    assert chosen.grad is not None and chosen.grad.item() < 0
    assert margin.item() == pytest.approx(0.8)


def test_indexed_table_reads_exact_episode_across_row_groups():
    # One episode crosses two row-group boundaries; its neighbors must be excluded.
    table = importers.IndexedTable(
        pl.DataFrame(
            {
                "id": [1] * 7000 + [2] * 12000 + [3] * 1000,
                "position": list(range(20000)),
            }
        ).lazy(),
        "id",
    )
    assert table.load(1)["position"].to_list() == list(range(7000))
    assert table.load(2)["position"].to_list() == list(range(7000, 19000))
    assert table.load(3)["position"].to_list() == list(range(19000, 20000))


def test_canonical_import_preserves_episode_semantics(run_values, tmp_path):
    from sequifier.config.policy_train_config import SequenceSourceConfig
    from sequifier.config.sequence_import_config import SequenceImportConfig

    config = config_for(run_values, "import")
    source = config.sources["decisions"]
    expected = {
        reference.load().episode_id: reference.load().model_dump()
        for reference in importers.source_episodes(source, config.policy)
    }
    manifest = importers.import_sequences(
        SequenceImportConfig.model_validate(
            {
                "operation": "import_sequences",
                "project_root": str(tmp_path),
                "output_path": "canonical",
                "source": source.model_dump(),
                "policy": config.policy.model_dump(
                    exclude={"interface_name", "context_length"}
                ),
                "episodes_per_shard": 3,
            }
        )
    )
    actual = {
        reference.load().episode_id: reference.load().model_dump()
        for reference in importers.source_episodes(
            SequenceSourceConfig(manifest=str(manifest)), config.policy
        )
    }
    assert actual == expected


def test_return_conditioning_and_fitted_categorical_inputs(run_values):
    from sequifier.config.policy_config import CategoricalPolicyConfig
    from sequifier.io.policy_dataset import PolicySequenceDataset, split_by_episode
    from sequifier.io.sequence import SequenceEpisode, TransitionColumns

    config = CategoricalPolicyConfig.model_validate(
        {
            "observation_columns": ["cue"],
            "categorical_columns": ["cue"],
            "static_columns": ["bias"],
            "context_length": 2,
            "interface_name": "actor",
            "action_values": ["left", "right"],
            "returns": {"gamma": 0.5, "scale": 2.0},
        }
    )
    episodes = [
        SequenceEpisode(
            episode_id=str(i),
            item_positions=(0, 1),
            observations={"cue": ("red", "blue")},
            static_inputs={"bias": float(i)},
            actions=("left", "right"),
            transitions=TransitionColumns(rewards=(1.0, 2.0), terminated=(False, True)),
        )
        for i in range(4)
    ]
    dataset = PolicySequenceDataset(episodes, config, episode_cache_size=1)
    training, validation = split_by_episode(dataset, 0.25, 1)
    fitted = training.fit_categorical_config()
    assert fitted.id_maps["cue"]["red"] != fitted.id_maps["cue"]["blue"]
    for part in (training, validation):
        part.use_config(fitted)
        part.validate_normalization(training.normalization_statistics())
        assert part[0].inputs["__policy_return"].tolist() == [[0.0, 2.0]]
        assert part[1].inputs["__policy_return"].tolist() == [[2.0, 2.0]]
    values = deepcopy(run_values)
    values["model"]["interfaces"]["actor"]["policy"].update(
        return_conditioned=True,
        observation_columns=["cue"],
        categorical_columns=["cue"],
        static_columns=["bias"],
    )
    values["model"]["interfaces"]["actor"]["input_columns"] = ["cue", "bias"]
    values["model"]["interfaces"]["actor"]["ingestion"]["feature_embedding_dims"] = {
        "cue": 2,
        "bias": 2,
        "__policy_previous_action": 2,
        "__policy_return": 2,
    }
    values["phases"] = values["phases"][:1]
    run = config_for(values, "conditioned")
    actor = build_actor(run, fitted, training.normalization_statistics())
    assert "__policy_return" not in actor.interface.selected_columns_statistics
    assert actor(training[0].model_inputs).shape[-1] == 2


def test_embedded_reference_survives_artifact_removal(
    run_values, tmp_path, monkeypatch
):
    values = deepcopy(run_values)
    values["phases"] = [values["phases"][2]]
    config = config_for(values, "reference-seed")
    actor = build_actor(config, config.policy, {"cue": {"mean": 0.0, "std": 1.0}})
    path = tmp_path / "reference.pt"
    save_policy(actor, path)
    values["phases"][0]["reference"] = {"source": "artifact", "path": str(path)}
    config = config_for(values, "reference-removal")
    save = run_checkpoint.atomic_save

    def interrupt(path, payload):
        save(path, payload)
        if payload["phase_state"] is not None:
            raise InterruptedError("committed DPO reference")

    with monkeypatch.context() as patch:
        patch.setattr(run_checkpoint, "atomic_save", interrupt)
        with pytest.raises(InterruptedError):
            run_policy_training(config)
    path.unlink()
    result = run_policy_training(
        config_for(values, config.model_name, resume="required")
    )
    assert len(result.phases) == 1 and result.phases[0].algorithm == "dpo"


def test_indexed_tables_preserve_model_encoding_and_static_fields(tmp_path):
    from sequifier.config.policy_config import PolicySchemaConfig
    from sequifier.config.policy_train_config import SequenceSourceConfig

    pl.DataFrame(
        {
            "sequenceId": [2, 1, 2, 1],
            "itemPosition": [1, 1, 0, 0],
            "cue": [1.0, -1.0, -1.0, 1.0],
            "action": [4, 3, 3, 4],
        }
    ).write_csv(tmp_path / "episodes.csv")
    pl.DataFrame({"sequenceId": [1, 2], "bias": [2.0, -2.0]}).write_csv(
        tmp_path / "static.csv"
    )
    metadata = DatasetMetadata(
        window_length=3,
        column_data_types={"cue": "Float32", "bias": "Float32", "action": "Int64"},
        n_classes={"action": 5},
        id_maps={"action": {"left": 3, "right": 4}},
        selected_columns_statistics={
            "cue": {"mean": 10.0, "std": 2.0},
            "bias": {"mean": 3.0, "std": 4.0},
        },
    )
    (tmp_path / "metadata.json").write_text(metadata.model_dump_json())
    source = SequenceSourceConfig(
        paths=[str(tmp_path / "episodes.csv")],
        static_data_path=str(tmp_path / "static.csv"),
        metadata_config_path=str(tmp_path / "metadata.json"),
        encoding="model",
    )
    policy = PolicySchemaConfig(
        observation_columns=["cue"],
        static_columns=["bias"],
        action_values=["left", "right"],
        return_conditioned=False,
    )
    episodes = {
        episode.episode_id: episode
        for reference in importers.source_episodes(source, policy)
        for episode in [reference.load()]
    }
    assert episodes["1"].actions == ("right", "left")
    assert episodes["2"].actions == ("left", "right")
    assert episodes["1"].observations["cue"] == pytest.approx((12.0, 8.0))
    assert episodes["1"].static_inputs["bias"] == pytest.approx(11.0)
    pl.DataFrame({"sequenceId": [1, 1, 2], "bias": [2.0, 2.0, -2.0]}).write_csv(
        tmp_path / "static.csv"
    )
    with pytest.raises(ValueError, match="without duplicates"):
        importers.source_episodes(source, policy)
