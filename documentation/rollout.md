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
a named preprocessing split. Row zero of a split is its first owned decision. It executes
named policy actions. By default `seed_history: available` uses preceding observations from the selected sequence
up to the artifact’s context length, with recorded action feedback and return budgets.
For fresh raw input and prepared prediction views, the adapter receives the original observation and static values. The policy applies its frozen codecs and input mask to the seed history. A feedback policy
starts a fresh history when preceding actions are unavailable. Set `seed_history: current`
to start with one observation and the binding’s initial feedback. See
[inference options](configs/infer.md#policy-model-interfaces) and the
[Python example](example_inputs/ppo_toy_control.py) using the shipped PPO scaffold.

Both modes accept a physical seed coordinate in the selected input split. Use exactly
one selector. Existing configs can continue using a zero-based `seed_window_index`
for forecasts or `seed_row_index` for policies. Coordinate selection stays stable
when the same items are repacked into different windows or shards.

```yaml
rollout:
  seed: {sequenceId: 42, itemPosition: 17}
  output: {manifest_path: data/simulation/manifest.json}
  # Keep the existing environment, max_steps and any forecast bindings here.
```

## Reusing rollout data

Forecast and policy inference publish one checksummed Parquet decision table beside the configured
manifest. Set `trajectory_path` to also keep the single-trajectory JSON. For PPO
and GRPO, use the same `output` mapping under the phase's `collection`. The phase
report records the manifest path and checksum. A completed PPO or GRPO phase publishes one Parquet file of decisions, referenced
by the manifest. Each decision carries its collection diagnostics in a
`__collection` struct at the same `sequenceId` and `itemPosition`. Preprocessing
selects observations by default. PPO stores the policy version, scaled reward and
collection boundary. Set `collection.output.include_scores: true` for PPO log
probabilities, entropy, value estimates, advantages and value targets. GRPO stores
the policy version, reference identity, group and sample indices, JSON start token,
scaled reward, log probabilities, trajectory score, advantage and cutoff reason.
The top-level reward is the raw environment reward. Immutable intermediate shards
remain available to resume older checkpoints.

Trajectory JSON stores the row-level mask in `seed.observation_mask`, aligned
with the padded seed history. The Parquet decision table carries
`__observation_mask` and its pre-decision context carries each preceding
observation's mask. Both supervised preprocessing and offline policy training
apply the recorded masks when consuming these outputs. A configured raw mask name
can read the recorded mask when the original column is absent. If both columns
are present, either can mark an observation as masked.

Each decision has an integer `sequenceId` and an `itemPosition`. `sequenceName`
carries the external episode identity. On import, `episode_namespace` prefixes this
label. The trajectory JSON and sequence manifest retain the same integer coordinates.
The `__group` column associates related samples,
such as GRPO siblings. For inference rollouts, both trajectory JSON and sequence
manifests group runs seeded from the same dataset, sequence ID, and item position. A PPO update boundary can split a sequence across shards.
Rows carry the current observation, static fields, raw reward, recorded successor
and separate boundary flags. A missing successor keeps the Arrow type of its observation,
and a missing previous action keeps the executed-action struct type. Predictions use `__prediction_<name>`. Policy
decisions also retain the structured executed action and previous action.

For supervised training, point `preprocessing_data_path` at the manifest and use
the usual split, feature and window settings. Use `read_format: sequence` for
a manifest without a JSON suffix. The default `sequence_view: decisions` exposes
one row per action, including named action fields when selected. When an observation
and an action share a name such as `x`, select `x` for the observation and
`__executed_x` for the action. With
`sequence_view: observations`, each sequence also includes its final successor
when recorded. For value-cutoff splits, a successor uses its own observed split
value when the split column is an observation. An auxiliary split column carries
the final decision's value forward to its successor. Auxiliary columns named by
`curriculum_column` also carry the final decision's values onto that row. The
usual requirement that each curriculum value be constant within a generated
subsequence applies. A declared static field can supply the cutoff or curriculum
value for every item in its sequence, including the final successor. Its feature
values are stored in the static table.
Existing inference trajectory JSON can be read through the same
interface. Supervised preprocessing extracts the declared static fields into a
separate table with one row per sequence.

Policy preparation exposes the same flat action names, including `__executed_x`,
alongside the structured action record. Both preparation routes record the split
names, method, column, fractions and cutoffs in dataset metadata.

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
