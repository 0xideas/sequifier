# Local numeric forecast rollouts

For trajectory-backed categorical policy training and return-conditioned
execution through the same coordinator, see [policy learning](policy-learning.md).

The first rollout slice collects one seed-initialized episode or episode fragment
on CPU. It supports real scalar features and real static conditioning, with
one-step `causal` or `final_value` forecast models (`target_offset: 1`). Model
predictions and a persistent local adapter jointly produce the next input row.
There is no learning algorithm in this path.

## Inference configuration

Add this to an otherwise valid inference configuration for a model with inputs
`x`, `d` and target `x`:

```yaml
device: cpu
model_type: generative
autoregressive: true
generation_steps: 3
rollout:
  mode: forecast
  seed_index: 0
  trajectory_path: outputs/trajectory.json
  adapter:
    factory: sequifier.rollout.dependent:LinearDependentAdapter
    protocol_version: 1
    config:
      prediction: x
      external_input: d
      multiplier: 2.0
      terminate_after: 3
  bindings:
    - {input: x, source: prediction, key: x}
    - {input: d, source: environment, key: d}
```

Run the existing `sequifier infer --config-path configs/infer.yaml` command.
PT and existing schema-bearing ONNX models use the same coordinator and their
existing inference backends. This feature adds no export format.

`seed_index` is required and selects one prepared model window, using zero-based
dataset/chunk order. Unlike legacy autoregression, selection happens before the
first-subsequence-per-sequence filter. It uses the usual inference window
selection rules, including `window_stride`. An out-of-range index fails. The
seed must have at least one valid position, end in a valid position, and contain
only left padding before its valid observations. Every feature and the attention
mask must have the same context length. Static values come from the existing
processed static feature table and stay constant.

Each changing input must have exactly one binding. `prediction` keys name model
targets; `environment` keys name adapter outputs. Static inputs cannot have
bindings. Missing producers, duplicate input bindings, unknown predictions, and
extra or missing adapter fields fail clearly. Outputs that are only consumed by
the adapter do not need a feedback binding.

The seed dataset uses the existing **preprocessed training representation**.
The adapter and stored trajectory use **original units**. The inference wrapper
decodes the seed and forecasts and encodes every successor using the fixed
training `mean`, `std`, and `normalize_real_columns` metadata. It uses the same
`std + 1e-9` transform as preprocessing, without fitting statistics on rollout
values. Float overflow, non-finite values, arrays in place of scalars, and
incompatible model inputs are rejected. Model padding stays zero with a false
attention mask. Features and attention masks slide together, appending a valid
successor position even for the last executed prediction.

With `rollout` present, inference writes the trajectory to `trajectory_path`
(relative to `project_root` unless absolute). It replaces that file atomically
after successful collection and validation. This path collects one selected
seed with one model; it does not produce legacy flat prediction/probability
files. Omitting `rollout` preserves existing inference output and configuration
rules.

## Adapter protocol

Adapters declare `protocol_version = 1` and implement:

```python
from sequifier.rollout_api import ResetRequest, StepRequest, StepResult

def reset(self, request: ResetRequest) -> None: ...
def step(self, request: StepRequest) -> StepResult: ...
def close(self) -> None: ...
```

A `module:factory` receives `adapter.config` as keyword arguments and is called
once per collection. The adapter stays alive for all steps and closes on success
or failure, including reset, prediction, and result-validation failures. A
factory that fails during construction owns cleanup of any partial resources.
Python callers can pass a supplied adapter instance to `collect_rollout`.

`reset` receives the episode ID, complete seed history, validity mask, static
inputs, and random seed. It initializes the environment from this history and
must not execute a transition or replace the seed. `step` receives the current
observation, all named forecasts, static inputs, and zero-based step ID. It
returns external fields, one finite scalar reward, and separate Python boolean
`terminated` and `truncated` flags. Reset and step requests contain independent
copies of coordinator data. Adapters must return the final pre-reset successor
and must not autoreset.

Every prediction executes exactly once before collection can end. A boundary
stops further prediction and stepping. `generation_steps` is a collection
budget: exhausting it leaves `collection_cutoff: true` when the episode is
unfinished, without fabricating termination or truncation. Optional
`rollout.max_episode_steps` is an episode time limit; reaching it records
`truncated: true`. An adapter can also report its own truncation. Both boundary
flags may be true together; termination takes precedence for bootstrap.

## Trajectory format and Python API

The version-1 JSON artifact contains the domain seed history and mask, static
values, input bindings, prediction names, episode ID, and ordered transitions.
Each transition records the observation, predictions (the executed forecast
actions), external fields, reward, successor, step ID, and separate boundary
flags. `bootstrap_allowed` is false on termination and true otherwise, including
time-limit truncation with its recorded final successor. The artifact also
contains `total_return`, `collection_cutoff`, and inference provenance, including
the exact preprocessing statistics and seed origin.

Seed history plus successors reconstruct every later history and attention mask;
they are not duplicated in each record. Artifacts record one episode fragment
and do not promise simulator snapshots or exact resumability.

```python
from sequifier.rollout_api import load_trajectory, validate_trajectory

trajectory = load_trajectory("outputs/trajectory.json")
validated = validate_trajectory(trajectory)
```

Saving and loading validate format/version, finite scalar data, seed shape and
mask, producer coverage, successor bindings, observation continuity, contiguous
episode/step IDs, total return, bootstrap semantics, and absence of transitions
after an episode boundary. The same validation is available for in-memory data.

For direct Python collection, a predictor is a callable receiving a copied
`SeedHistory` in original units and returning a dictionary of scalar forecasts.
The collector owns adapter closure. `history_validator`, if provided, validates
the initial history and every successor (including a terminal successor) for
the predictor's representation; the inference integration uses the fixed codec
and existing `ExecutionSchema` here.

```python
from sequifier.rollout_api import (
    RolloutConfig, SeedHistory, collect_rollout, save_trajectory,
)
from sequifier.rollout.dependent import LinearDependentAdapter

config = RolloutConfig(
    adapter={"factory": "sequifier.rollout.dependent:LinearDependentAdapter"},
    bindings=[
        {"input": "x", "source": "prediction", "key": "x"},
        {"input": "d", "source": "environment", "key": "d"},
    ],
    seed_index=0,
    trajectory_path="outputs/trajectory.json",
)
seed = SeedHistory(
    observations={"x": [0.0, 0.0], "d": [0.0, 0.0]},
    attention_valid_mask=[False, True],
    static_inputs={},
)

def predict(history):
    return {"x": history.observations["x"][-1] + history.observations["d"][-1] + 1.0}

trajectory = collect_rollout(
    predict, LinearDependentAdapter(), seed, config, ["x"], max_steps=3,
)
save_trajectory(trajectory, config.trajectory_path)
```

The hand-calculated trajectory for that predictor is:

| Step ID | Observation | Prediction | Successor | Reward | Terminated |
| --- | --- | --- | --- | --- | --- |
| 0 | `(0, 0)` | `x=1` | `(1, 2)` | 1 | false |
| 1 | `(1, 2)` | `x=4` | `(4, 8)` | 4 | false |
| 2 | `(4, 8)` | `x=13` | `(13, 26)` | 13 | true |

The return is 18, with no collection cutoff. The first mask is `[False, True]`;
the next and subsequent masks are `[True, True]`. No fourth call occurs.

Multimodal and categorical observations, reset-only initialization, vectorized
environments, remote transport, learning algorithms, and new export support are
outside this slice.
