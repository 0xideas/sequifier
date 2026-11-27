# Local rollouts

Forecast and policy rollout share trajectory recording but retain distinct adapter
protocols. Both execute one local episode/fragment, own adapter cleanup, and atomically
publish the observation table and sequence metadata under `output.manifest_path`.

## Forecast configuration and protocol

Forecast rollout supports CPU, real scalar temporal/static inputs, and one-step
`causal`/`final_value` models with `target_offset: 1`, using PT or schema-bearing ONNX:

```yaml
task: forecast
rollout:
  max_steps: 100
  seed_window_index: 0
  output: {manifest_path: outputs/trajectory/manifest.json}
  environment:
    factory: application.adapters:make_environment
    config: {}
  bindings:
  - input: x
    source: prediction
    key: x
  - input: disturbance
    source: environment
    key: disturbance
artifact:
  paths: [models/forecast.pt]
input:
  metadata_config_path: configs/metadata_configs/events.json
  split: test
```

Run `sequifier infer --config-path configs/infer.yaml`. `seed_window_index` selects
one prepared window in dataset/chunk order, following normal stride/window rules
before first-subsequence filtering. Each temporal input needs exactly one producer;
static inputs cannot have bindings. Extra/missing environment fields fail validation.

Forecast adapters subclass `ForecastRolloutAdapterBase` from
`sequifier.rollout_api`. They implement `reset(ResetRequest)`,
`step(StepRequest) -> StepResult`, and `close()`. The same module exports the
request and result types.
Factories receive `environment.config` as keyword arguments. Reset initializes from
the supplied history without executing a transition; step receives copied current
observations, forecasts and static values. It returns external inputs, a finite reward
and separate Python boolean `terminated`/`truncated` flags. Adapters must not autoreset.

Adapters and trajectories use original units. Both seed selectors read the physical
observations and static values retained by preprocessing. The predictor uses the
frozen training codecs. An observation hidden by the preprocessing mask still
initializes the environment with its recorded value. In the model window it
receives the fitted feature's mask value. Padding, tracked by
`attention_valid_mask`, has no corresponding observation. Unsupported shapes, overflow and non-finite transformed inputs fail.

## Policy rollout

Policy rollout uses the [online policy adapter](policy-learning.md#online-adapters),
matching `units` and seeded `reset_from_observation`. It selects
a row with `seed_row_index` from the same input as policy prediction, including
a named preprocessing split. The seed coordinate identifies the last observation available to the policy. It executes
named policy actions. By default `seed_history: available` uses preceding observations from the selected sequence
up to the artifact’s context length, with recorded action feedback and return budgets.
For fresh raw input and prepared prediction views, the adapter receives the original observation and static values. The policy applies its frozen codecs and input mask to the seed history. A feedback policy
reads the incoming action on each observation. Set `seed_history: current`
to start with the selected observation and its incoming action. See
[inference options](configs/infer.md#policy-model-interfaces) and the
[Python example](example_inputs/ppo_toy_control.py) using the shipped PPO scaffold.

Both modes accept a physical seed coordinate in the selected input split. Use exactly
one selector. Existing configs can continue using a zero-based `seed_window_index`
for forecasts or `seed_row_index` for policies. Coordinate selection stays stable
when the same items are repacked into different shards. In a prepared split, a
seed must be covered by a preprocessing window and belong to that split. If a
sparse stride leaves a gap, history starts again at the first covered item after
the gap. Aligned windows can retain context from before the split. These rules
apply to both indexed and materialized inputs.

```yaml
rollout:
  seed: {sequenceId: 42, itemPosition: 17}
  output: {manifest_path: data/simulation/manifest.json}
  # Keep the existing environment, max_steps and any forecast bindings here.
```

## Reusing rollout data

A sequence with `N` actions has `N + 1` observation rows. Each sequence starts at
`itemPosition = 0`. After each action the collector appends its resulting
observation, executed action, raw reward and termination flags at the next
position. Reset has no incoming action. A seeded rollout can have an incoming
action at position zero, which supplies context only.

| Position | Observation | Incoming action | Incoming reward |
|---|---|---|---|
| 0 | O0 | empty at reset | empty |
| 1 | O1 | A0 | R0 |
| 2 | O2 | A1 | R1 |

Set `output.manifest_path` for inference, or `collection.output.manifest_path`
for PPO and GRPO. The manifest references checksummed Parquet files for temporal
rows and per-sequence metadata. The latter holds `sequenceName`, `groupId`, and
static inputs once per sequence. Action aliases and the observation mask column
are declared in the manifest. Collection diagnostics on row 1 describe the
choice of A0 at O0. At a PPO chunk boundary the next chunk appends only the
arrivals produced by its own actions.

Use the manifest as `preprocessing_data_path` for supervised learning. History
through position `i` predicts the action on row `i+1`. The final row supplies
the last target. A split owns a label by its target coordinate. The derived
numeric feature `incomingActionPresent` distinguishes reset from a recorded zero
action. Forecast rollouts populate this feature automatically when it is a model input.
An observation mask hides sensor features while retaining recorded action feedback.

For policy learning, set the sequence source's `manifest_path` or pass it to
`prepare_policy`. Forecast predictions used as actions require
`action_semantics: executed_predictions` and matching policy action names.

A later phase in the same training plan can refer to an earlier PPO or GRPO
phase by name. The producer must have `collection.output` configured.

```yaml
datasets:
  simulations:
    model_interface: actor
    sources:
      recorded:
        type: rollout
        phase: collect
        preprocessing_config_path: configs/preprocess-simulations.yaml
```

The preprocessing config supplies the feature roles and named splits. The runner
supplies the committed manifest and indexed output location. An omitted preprocessing
seed uses the standalone preprocessing default. An explicit integer fixes the split
assignment in both routes. In a training plan, an explicit `random` uses the resolved
run seed so recovery can reproduce the split. Explicit views can select existing
parts before collection and the produced part after collection. A generated
partition uses all its source parts to assign membership. For direct policy
consumption, omit `preprocessing_config_path` and select the whole source or use
the dataset's `partition` settings. Supervised phases consume prepared splits.
Use the normal phase source references, such as `simulations.train`, to select
the resulting view.

## Boundaries and trajectory API

Every issued prediction/action executes once. True termination disables bootstrap;
truncation retains the final pre-reset successor. `max_steps` is a collection budget:
exhaustion sets `collection_cutoff` without inventing a boundary. Optional
`max_episode_steps` imposes a truncation. For return-conditioned policies with a
finite return horizon, that horizon ends the episode without bootstrap. It is
used automatically, and an explicit `max_episode_steps` must match it.
Terminal policy steps may omit successors.
The collector closes adapters on success/failure; construction failures remain the
factory's cleanup responsibility.

Trajectories record seed/mask/static values, bindings, ordered transitions, original
unit predictions/actions/rewards, successors, boundary flags, total return and provenance.
Validation checks continuity, producer coverage, IDs, totals, finite values and boundary
semantics. Policy records also retain joint execution metadata and transform latents.

`sequifier.rollout_api` exposes `collect_rollout`, `save_trajectory`, `load_trajectory`
and `validate_trajectory`. Direct forecast predictors consume a copied `SeedHistory`
and return named forecasts; optional `history_validator` checks the initial and every
successor history. `sequifier.rl_api.collect_policy_rollout` provides policy collection.
Standalone trajectories do not imply simulator snapshots or exact resumability.
