import pytest
import yaml
from pydantic import ValidationError

from sequifier.config.hyperparameter_search_config import (
    TrainingSpecHyperparameterSampling,
)
from sequifier.io.yaml import TrainModelDumper


class RecordingTrial:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.params = {}

    def suggest_categorical(self, name, choices):
        value = self.overrides.get(name, choices[0])
        assert value in choices
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None, log=False):
        value = self.overrides.get(name, low)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        value = self.overrides.get(name, low)
        self.params[name] = value
        return value


def make_training_sampling(**overrides):
    config = {
        "device": "cpu",
        "epochs": [1],
        "save_interval_epochs": 1,
        "batch_size": [2],
        "learning_rate": [0.001],
        "criterion": {"itemId": "CrossEntropyLoss"},
        "accumulation_steps": [1],
        "dropout": [0.0],
        "optimizer": [{"name": "Adam"}],
        "scheduler": [{"name": "StepLR", "step_size": 1, "gamma": 0.99}],
        "continue_training": False,
    }
    config.update(overrides)
    return TrainingSpecHyperparameterSampling(**config)


def bert_spec_sampling_config():
    return {
        "masking_probability": [0.15, 0.30],
        "replacement_distribution": [
            {"masked": 0.8, "random": 0.1, "identical": 0.1},
            {"masked": 0.6, "random": 0.2, "identical": 0.2},
        ],
        "span_masking": [
            {"type": "GeometricDistribution", "p": 1.0},
            {"type": "PoissonDistributionFloor", "rate": 2.0},
        ],
    }


def sample_training_spec(sampling, trial):
    return sampling.sample_trial(trial, max_lookahead=1, sample_length=9)


def test_training_objective_defaults_to_causal_without_bert_spec():
    sampling = make_training_sampling()
    trial = RecordingTrial()

    training_spec = sample_training_spec(sampling, trial)

    assert training_spec.training_objective == "causal"
    assert training_spec.bert_spec is None
    assert training_spec.sample_length == 9
    assert trial.params["training_objective"] == "causal"


def test_bert_spec_fields_are_sampled_separately_for_bert_objective():
    sampling = make_training_sampling(
        training_objective=["causal", "bert"],
        bert_spec=bert_spec_sampling_config(),
    )
    trial = RecordingTrial(
        {
            "training_objective": "bert",
            "bert_masking_probability": 0.30,
            "bert_replacement_distribution_index": 1,
            "bert_span_masking_index": 1,
        }
    )

    training_spec = sample_training_spec(sampling, trial)

    assert training_spec.training_objective == "bert"
    assert training_spec.bert_spec is not None
    assert training_spec.bert_spec.masking_probability == 0.30
    assert training_spec.bert_spec.replacement_distribution.masked == 0.6
    assert training_spec.bert_spec.replacement_distribution.random == 0.2
    assert training_spec.bert_spec.replacement_distribution.identical == 0.2
    assert training_spec.bert_spec.span_masking.type == "PoissonDistributionFloor"
    assert set(trial.params).issuperset(
        {
            "training_objective",
            "bert_masking_probability",
            "bert_replacement_distribution_index",
            "bert_span_masking_index",
        }
    )


def test_causal_objective_does_not_sample_bert_fields():
    sampling = make_training_sampling(
        training_objective=["causal", "bert"],
        bert_spec=bert_spec_sampling_config(),
    )
    trial = RecordingTrial({"training_objective": "causal"})

    training_spec = sample_training_spec(sampling, trial)

    assert training_spec.training_objective == "causal"
    assert training_spec.bert_spec is None
    assert "bert_masking_probability" not in trial.params
    assert "bert_replacement_distribution_index" not in trial.params
    assert "bert_span_masking_index" not in trial.params


def test_bert_objective_requires_bert_spec_sampling_config():
    with pytest.raises(ValidationError, match="bert_spec"):
        make_training_sampling(training_objective=["bert"])


def test_bert_training_spec_dumps_to_plain_yaml():
    sampling = make_training_sampling(
        training_objective=["bert"],
        bert_spec=bert_spec_sampling_config(),
    )
    training_spec = sample_training_spec(sampling, RecordingTrial())

    dumped = yaml.dump(training_spec, Dumper=TrainModelDumper, sort_keys=False)
    loaded = yaml.safe_load(dumped)

    assert loaded["training_objective"] == "bert"
    assert loaded["bert_spec"]["masking_probability"] == 0.15
    assert loaded["bert_spec"]["replacement_distribution"] == {
        "masked": 0.8,
        "random": 0.1,
        "identical": 0.1,
    }
    assert loaded["bert_spec"]["span_masking"] == {
        "type": "GeometricDistribution",
        "p": 1.0,
    }
