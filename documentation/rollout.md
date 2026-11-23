# Local rollouts

Forecast and policy rollout share trajectory recording but retain distinct adapter
protocols. Both execute one local episode/fragment, own adapter cleanup, and atomically
write `trajectory_path` after successful validation. They do not add an export format.

## Forecast configuration and protocol

Forecast rollout supports CPU, real scalar temporal/static inputs, and one-step
`causal`/`final_value` models with `target_offset: 1`, using PT or schema-bearing ONNX:

```yaml
task: forecast
rollout:
  max_steps: 100
  seed_window_index: 0
  trajectory_path: outputs/trajectory.json
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

Adapters and trajectories use original units. Fixed training codecs decode seeds/
forecasts and encode successors with the same `std + 1e-9` normalization; rollout
never fits statistics. Padding stays zero with a false mask; history and mask slide
together. Unsupported shapes, overflow and non-finite transformed inputs fail.

## Policy rollout

Policy rollout uses the [online policy adapter](policy-learning.md#online-adapters),
matching `units` and seeded `reset_from_observation`. It selects
a row with `seed_row_index` from the same input as policy prediction, including
a named preprocessing split. Row zero of a split is its first owned decision. It executes
named policy actions. By default `seed_history: available` uses preceding observations from the selected sequence
up to the artifact’s context length, with recorded action feedback and return budgets.
The adapter resets at the selected observation and executes from there. A feedback policy
starts a fresh history when preceding actions are unavailable. Set `seed_history: current`
to start with one observation and the binding’s initial feedback. See
[inference options](configs/infer.md#policy-model-interfaces) and the
[Python example](example_inputs/ppo_toy_control.py) using the shipped PPO scaffold.

Both seed indices are required, zero-based, and checked before environment creation.
Only the route-specific seed index name is accepted and recorded in trajectory
metadata. Neither `seed_index` nor `rollout.type` is accepted.

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
