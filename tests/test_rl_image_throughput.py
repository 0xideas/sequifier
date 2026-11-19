"""Verify compact policy histories and padded-image skipping without training."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from sequifier.config.policy_config import ResolvedPolicyBinding
from sequifier.io.values import portable_fields
from sequifier.model.execution_schema import ExecutionSchema, InputDescriptor
from sequifier.model.images import ImageEncoder
from sequifier.rl.collection import PolicyCursor, PolicyDecision
from sequifier.rl.environment import (
    OnlineResetResult,
    OnlineStepResult,
    validate_reset,
    validate_step,
)
from sequifier.rl.history import (
    PolicyFrame,
    PolicyHistory,
    _json_payload_bytes,
    frame_storage_bytes,
    history_inputs,
)


@pytest.fixture(scope="module")
def spec():
    inputs = [
        InputDescriptor(f"feature_{index}", column, "feature", "float32", ("batch", 32))
        for index, column in enumerate(
            ("observation_state_x", "observation_state_y", "action_x", "action_y")
        )
    ]
    inputs += [
        InputDescriptor(
            "image_0", "__image:camera", "image", "uint8", ("batch", 32, 1, 3, 224, 224)
        ),
        InputDescriptor(
            "image_mask_0", "__image_mask:camera", "metadata", "bool", ("batch", 32, 1)
        ),
        InputDescriptor(
            "attention_valid_mask",
            "attention_valid_mask",
            "metadata",
            "bool",
            ("batch", 32),
        ),
    ]
    return ResolvedPolicyBinding(
        actions={
            name: {"type": "real", "scale": {"fixed": 10.0}}
            for name in ("action_x", "action_y")
        },
        previous_action_inputs={"action_x": "action_x", "action_y": "action_y"},
        initial_inputs={"action_x": 256.0, "action_y": 256.0},
        context_length=32,
        interface_name="actor",
        observation_columns=[
            "observation_state_x",
            "observation_state_y",
            "__image:camera",
            "__image_mask:camera",
        ],
        input_schema=ExecutionSchema(tuple(inputs), (), {}, {}).to_dict(),
    )


def observation():
    rng = np.random.default_rng(17)
    return {
        "observation_state_x": 100.0,
        "observation_state_y": 200.0,
        "__image:camera": rng.integers(0, 256, (1, 3, 224, 224), dtype=np.uint8),
        "__image_mask:camera": np.array([True]),
    }


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (1,), (2, 3), (1, 3, 7, 9)])
def test_compact_json_size_matches_portable_json(dtype, shape):
    rng = np.random.default_rng(12)
    array = rng.integers(0, 256, shape, dtype=np.uint8).astype(dtype)
    value = {"é": array, "nested": [None, 1.25, "漢字", (False, 1)], 17: {None: True}}
    expected = len(json.dumps(portable_fields(value), allow_nan=False).encode("utf-8"))
    assert _json_payload_bytes(value) == expected


def test_validators_can_own_compact_images_without_changing_portable_default():
    raw = observation()
    reset = OnlineResetResult(raw)
    step = OnlineStepResult(raw, 1.0)
    for validator, result in ((validate_reset, reset), (validate_step, step)):
        standard = validator(result, {})
        compact = validator(result, {}, compact_images=True)
        assert standard.observation is not None
        assert compact.observation is not None
        assert isinstance(standard.observation["__image:camera"], list)
        assert isinstance(compact.observation["__image:camera"], np.ndarray)
        assert portable_fields(compact.observation) == standard.observation
        assert not np.shares_memory(
            raw["__image:camera"], compact.observation["__image:camera"]
        )


def test_history_preserves_tensors_checkpoints_and_accounting(spec):
    raw = observation()
    history = PolicyHistory(spec)
    history.reset(raw, {}, "episode")
    owned = dict(history.current.observation)["__image:camera"]
    assert isinstance(owned, np.ndarray)
    assert not np.shares_memory(owned, raw["__image:camera"])
    frame = history.current
    portable = frame.state_dict()
    legacy = PolicyFrame.from_state_dict(portable)
    assert isinstance(portable["observation"]["__image:camera"], list)
    baseline = history_inputs(spec, [legacy])
    compact = history.inputs()
    for field in ("inputs", "metadata"):
        left, right = getattr(baseline, field), getattr(compact, field)
        assert all(torch.equal(left[key], right[key]) for key in left)
    expected = 1024 + 4 * len(json.dumps(portable, allow_nan=False).encode("utf-8"))
    assert frame_storage_bytes(frame) == expected
    # The second accounting pass must use the cached cost, not touch pixels.
    assert frame.storage_bytes == expected
    raw["__image:camera"].fill(0)
    assert np.any(owned)
    saved = json.loads(json.dumps(history.state_dict()))
    restored = PolicyHistory(spec)
    restored.load_state_dict(saved)
    assert restored.state_dict()["frames"] == saved["frames"]
    assert isinstance(dict(restored.current.observation)["__image:camera"], np.ndarray)


def test_compact_history_retains_lagged_actions(spec):
    history = PolicyHistory(spec)
    history.reset(observation(), {}, "episode")
    history.advance({"action_x": 123.0, "action_y": 234.0}, observation())
    inputs = history.inputs()
    assert inputs.inputs["action_x"][0, -1] == 123.0
    assert inputs.inputs["action_y"][0, -1] == 234.0
    assert inputs.attention_valid_mask.sum() == 2


def test_collection_cursor_uses_compact_images(spec):
    class Actor:
        config = spec
        device = torch.device("cpu")

        def model_inputs(self, inputs):
            return inputs.inputs, inputs.metadata

    class Adapter:
        def step(self, actions):
            return OnlineStepResult(observation(), 0.5, truncated=True)

    cursor = PolicyCursor(Actor(), Adapter(), {}, 17)
    cursor.reset(OnlineResetResult(observation()), "episode")
    actions = {"action_x": 123.0, "action_y": 234.0}
    result = cursor.step(PolicyDecision(actions, actions, 0.0, 0.0, 0.0))
    assert cursor.ended and result.truncated
    assert result.observation is not None
    assert isinstance(result.observation["__image:camera"], np.ndarray)
    assert isinstance(
        dict(cursor.history.current.observation)["__image:camera"], np.ndarray
    )
    assert cursor.inputs.inputs["action_x"][0, -1] == 123.0


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3), requires_grad=False)
        self.seen = 0

    def forward(self, values):
        self.seen += values.shape[0]
        return values.mean(dim=(-1, -2)) * self.weight


def test_frozen_encoder_skips_padding_and_preserves_projection_gradients():
    # Exercise the real encoder methods using a tiny stand-in backbone; no
    # pretrained weights, network access, model optimizer, or training run.
    encoder = ImageEncoder.__new__(ImageEncoder)
    nn.Module.__init__(encoder)
    encoder.frozen = True
    encoder.backbone = CountingBackbone()
    encoder.projection = nn.Linear(3, 2)
    encoder.output_dim = 2
    encoder.register_buffer("mean", torch.zeros(1, 3, 1, 1))
    encoder.register_buffer("std", torch.ones(1, 3, 1, 1))
    encoder.eval()
    images = (
        torch.arange(6 * 3 * 8 * 8)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 3, 1, 3, 8, 8)
    )
    valid = torch.tensor([[[False], [False], [True]], [[False], [True], [True]]])
    dense = encoder._encode(images.reshape(-1, 3, 8, 8)).reshape(2, 3, 1, 2)
    (dense * valid[..., None]).sum().backward()
    gradients = {
        name: p.grad.clone() for name, p in encoder.projection.named_parameters()
    }
    encoder.zero_grad()
    encoder.backbone.seen = 0
    sparse = encoder(images, valid)
    (sparse * valid[..., None]).sum().backward()
    assert encoder.backbone.seen == int(valid.sum())
    torch.testing.assert_close(sparse[valid], dense[valid])
    for name, parameter in encoder.projection.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name])
