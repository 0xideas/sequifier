# Local rollouts

Forecast and policy rollout share trajectory recording but retain distinct adapter
protocols. Both execute one local episode/fragment, own adapter cleanup, and atomically
write `trajectory_path` after successful validation. They do not add an export format.

## Forecast configuration and protocol

Forecast rollout supports CPU, real scalar temporal/static inputs, and one-step
`causal`/`final_value` models with `target_offset: 1`, using PT or schema-bearing ONNX:

```yaml
model_type: generative
rollout:
  max_steps: 100
  seed_window_index: 0
  trajectory_path: outputs/trajectory.json
  environment: {factory: application.adapters:make_environment, config: {}}
  bindings:
    - {input: x, source: prediction, key: x}
    - {input: disturbance, source: environment, key: disturbance}
```

Run `sequifier infer --config-path configs/infer.yaml`. `seed_window_index` selects
one prepared window in dataset/chunk order, following normal stride/window rules
before first-subsequence filtering. Each temporal input needs exactly one producer;
static inputs cannot have bindings. Extra/missing environment fields fail validation.

Forecast adapters declare `protocol_version = 1` and implement `reset(ResetRequest)`,
`step(StepRequest) -> StepResult`, and `close()`, exported by `sequifier.rollout_api`.
Factories receive `environment.config` as keyword arguments. Reset initializes from
the supplied history without executing a transition; step receives copied current
observations, forecasts and static values. It returns external inputs, a finite reward
and separate Python boolean `terminated`/`truncated` flags. Adapters must not autoreset.

Adapters and trajectories use original units. Fixed training codecs decode seeds/
forecasts and encode successors with the same `std + 1e-9` normalization; rollout
never fits statistics. Padding stays zero with a false mask; history and mask slide
together. Unsupported shapes, overflow and non-finite transformed inputs fail.

## Policy rollout

Policy rollout uses the [version-3 online adapter](policy-learning.md#online-adapters),
matching `implementation_id`/`units` and seeded `reset_from_observation`. It selects
raw CSV/Parquet rows with `seed_row_index`, rejects forecast bindings, and executes
named policy actions. A seed supplies exactly one current observation; model history,
codecs and normalization come from its artifact. See
[inference options](configs/infer.md#policy-model-interfaces) and the
[Python example](example_inputs/ppo_toy_control.py).

Both seed indices are required, zero-based, and checked before environment creation.
Legacy `seed_index` aliases the route-specific field; supplying both names fails.
Trajectory metadata retains its public `seed_index` key. No `rollout.type` is accepted.

## Boundaries and trajectory API

Every issued prediction/action executes once. True termination disables bootstrap;
truncation retains the final pre-reset successor. `max_steps` is a collection budget:
exhaustion sets `collection_cutoff` without inventing a boundary. Optional
`max_episode_steps` imposes a truncation. Terminal policy steps may omit successors.
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
