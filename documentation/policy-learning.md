# Policy learning through `sequifier train` and `sequifier infer`

Policy learning uses Sequifier's configurable ingestion, transformer network,
decoders, portable model artifacts, optimizer transactions, and run checkpoints.
A policy adds a decision-alignment contract to a model interface: at time `t`,
the inputs contain the current observation, the **previous** action, and,
optionally, a desired return. The current action is the supervised target.

## Train from ordinary sequence tables

The phase algorithm is `policy_imitation` for both behavior cloning and
return-conditioned imitation. The policy's `return_conditioned` setting selects
between them. Offline IQL uses a separate `iql` phase with an unconditioned
categorical actor.

Use the existing `sequenceId` and `itemPosition` conventions. Each row describes
**the current observation, the action executed at that observation, and the
reward/boundary resulting from that action**. One sequence is one episode or an
explicit episode fragment. CSV and Parquet files can contain many episodes;
files are not assumed to be episode boundaries. Positions must be nonnegative,
unique and contiguous within each sequence, including across input shards.

```yaml
project_root: .
model_name: table-policy
device: cpu
seed: 1010

model:
  backbone:
    architecture:
      dim_model: 32
      num_layers: 2
      max_context_length: 16
      attention: {n_heads: 4}
      feed_forward: {dim: 128}
      positional_encoding_scope: global
  interfaces:
    actor:
      input_columns: [position, velocity]
      target_columns: [action]
      policy:
        observation_columns: [position, velocity]
        action_column: action
        action_values: [left, right]
        return_conditioned: false
      ingestion:
        type: embedding
        output_dim: 8
        feature_embedding_dims:
          position: 1
          velocity: 1
          __policy_previous_action: 6
      decoder: {type: linear, prediction_length: 1, support: 1}

dataset_training:
  demonstrations:
    model_interface: actor
    parts:
      recorded:
        paths: [data/demonstrations.parquet]

global_training:
  training_objective: policy
  context_length: 16
  batch_size: 64
  learning_rate: 0.005
  validation_fraction: 0.2

training_plan:
  phases:
    - name: behavior_cloning
      algorithm: policy_imitation
      policy: actor
      epochs: 80
      sources: [{source: demonstrations.recorded}]
```

Run `sequifier train --config-path configs/train-policy.yaml`. BC
(`return_conditioned: false`) needs observations and action labels, without
rewards, successor observations, bindings, simulator metadata or fabricated seed
windows. A recorded current action is never also accepted as an observation from
the same source column. Return-conditioned modeling additionally needs recorded
rewards and a known terminal outcome or an explicitly configured finite horizon.
Unknown/truncated reward tails are excluded and reported; missing rewards never
become zero. Setting `return_conditioned: true` selects this mode.

Defaults use the policy feature names and the columns `sequenceId`,
`itemPosition`, `reward`, `terminated` and `truncated`. Map different source names
explicitly:

```yaml
dataset_training:
  demonstrations:
    model_interface: actor
    parts:
      recorded:
        paths: [data/recorded.csv]
        columns:
          sequence_id: episode
          item_position: step
          observations: {position: measured_position, velocity: measured_velocity}
          action: command
          reward: score
          terminated: task_finished
          truncated: time_limit
          previous_action: preceding_command
          group: collection_session
```

Boundary columns, when present, contain booleans. No row may follow a boundary
within the same episode. Both termination and truncation may be true; termination
disables bootstrap. `episode_end: terminated` or `truncated` is an explicit claim
about each sequence's final row and must agree with recorded flags. The default
is `unknown`, never an inferred terminal outcome at file end.

Position zero with no preceding action is an episode start. Fragments require
`columns.previous_action` at their first position; subsequent entries must lag
current actions by exactly one decision. At window boundaries, the preceding
action is fetched from the sequence rather than replaced with a start token.
Static features can be constant inline columns or supplied in a separate
`static_data_path` CSV/Parquet table keyed by the sequence ID. The table must cover
every selected ID without duplicates; fitting reads only training episodes.

Optional `columns.successors` maps every observation name to its next-observation
column. An all-null successor is explicitly missing; partial successors are
invalid. Adjacent observations must agree with supplied successors. Optional
`columns.bootstrap_allowed` cannot permit bootstrap after termination or at a
final decision without an observed successor. BC and return-conditioned modeling
do not require successors. Task outcomes and collection groups remain data
metadata rather than model features.

Without `validation_sources`, training splits whole episodes deterministically,
keeping repeated demonstrations and declared collection groups together. At
least two independent groups are required. Alternatively declare a held-out source
and set `validation_sources: [held_out.recorded]` on the phase. Duplicate paths and episode
IDs, overlapping collection groups and repeated demonstrations across partitions
are rejected. Normalization and categorical vocabulary fitting use training data.

Declare raw categorical observation/static fields with `categorical_columns` on
the policy. They use the original preprocessing vocabulary fitter and reserved
IDs. Alternatively supply fixed `observation_n_classes` / `static_n_classes` for
already encoded integer IDs and optional `id_maps` for fixed domain labels.
String action labels are supported through `action_values`. Executable action
IDs use the usual global-to-decoder mapping; the start marker is input-only and
cannot be sampled as an action. Previously saved policy artifacts retain their
legacy vocabulary layout.

Sources accept `format: auto` (default), `table`, `trajectory` or `prepared`.
Automatic selection recognizes CSV/Parquet tables, JSON rollout logs and PT
prepared histories. Prepared CSV/Parquet windows require `format: prepared`.
Prepared sources require `metadata_config_path`; overlapping windows are checked
and reconstructed into unique, split-owned decisions. A preceding action can be
recovered from stored context for a fragment. Categorical IDs and normalized
values are decoded using the original metadata, then fitted for the selected
training partition. Missing reward/boundary information remains unknown.
For a metadata-referenced prepared split, use `metadata_config_path` and
`split_index` instead of `paths`. Scalar prepared histories are currently supported;
policy image/depth ingestion remains planned; IQL and PPO support the scalar
transition-capable sources described below.

A source can still specify `paths: [episode-a.json, episode-b.json]` or a legacy
`manifest` listing version-1 JSON episodes. These are validated rollout recordings
adapted into the same sequence representation, with bindings retained in their
recording format. Forecast recordings require
`action_semantics: executed_predictions`; this does not convert forecast checkpoint
alignment into policy alignment.

Run paths resolve relative to `project_root`; manifest entries resolve relative
to the manifest directory. `global_training.num_workers` controls loader workers;
`episode_cache_size` bounds cached episodes per worker. Table-backed episodes use
filtered lazy scans. Prepared window reconstruction is eager per source and JSON
records still require one episode to fit in memory.

The best model is selected by held-out action NLL. `model_path` defaults to
`models/<model_name>/<model_name>.pt` and contains an ordinary `sequifier_model`.
The report records partitions, exclusions, the fitted
policy codec, decision counts and per-epoch NLL. Epochs belong to the phase, seed
and device to the run, and optimizer/scheduler settings to `global_training`.

## Optional committed import

Direct training from a table needs no conversion. For reusable imported data,
`sequifier preprocess` supports a separate operation:

```yaml
operation: import_sequences
project_root: .
source:
  paths: [data/demonstrations.csv]
policy:
  observation_columns: [position, velocity]
  action_column: action
  action_values: [left, right]
  return_conditioned: false
output_path: data/imported-demonstrations
episodes_per_shard: 64
```

The output directory must be new. Import writes raw Parquet shards and a
checksummed `manifest.json`, then atomically publishes the directory. It does not
fit statistics, pad contexts or invent missing transitions. Train with
`dataset_training.demonstrations.parts.recorded.manifest: data/imported-demonstrations/manifest.json`;
the recorded column mappings are loaded automatically. The manifest retains
source provenance, episode identities, optional successors and collection groups.

## Shared model and dataset authoring

Policy training uses the ordinary `model.backbone` and `model.interfaces`
schemas, with a `policy` data schema on each interface. Ingestion and decoder
components keep their existing syntax; the policy adapter adds previous-action
and return features. `global_training.context_length` selects a context no larger
than the backbone maximum. `global_training` supplies optimizer settings and
`training_objective: policy`. The former compact policy registries and aliases
have been consolidated into this shared authoring surface.

Likewise, `dataset_training.<name>` accepts `model_interface` and `parts`, with each
part containing a sequence source specification. Reference it with the ordinary
`{source: dataset.part}` phase syntax. The algorithm supplies action NLL, so these
parts do not need fabricated criteria. Model/interface references are validated.
The shared phase schema preserves supervised defaults; this learner still supports
one `policy_imitation` or `iql` phase, with BC selected on its policy. Mixed algorithms,
backbone repositories, image/depth policies and online learning remain planned.
Use `initialize_from` for backbone artifact transfer.

See [train-policy-from-table.yaml](example_inputs/configs/train-policy-from-table.yaml)
for a complete example using the shared model, dataset and phase authoring surface.

## Configure the shared model

Configure the backbone at `model.backbone.architecture` and ingestion/decoder
components at `model.interfaces.<name>`. For a larger model, for example:

```yaml
model:
  backbone:
    architecture:
      dim_model: 64
      num_layers: 3
      max_context_length: 16
      attention: {n_heads: 4}
      feed_forward: {dim: 128, activation: swiglu}
      positional_encoding_scope: global
```

Use an embedding ingestion to project real features and embed categorical
features before adapting to the backbone width. Supply ingestion and decoder
components explicitly as in the complete examples. A policy decoder must have
`support: 1` and `prediction_length: 1`.

The internal input names `__policy_previous_action` and `__policy_return` are
reserved. The latter exists only for return-conditioned policies. Static columns
are broadcast over the decision window. Real observation and static sample statistics
are persisted in the model interface's preprocessing metadata. Encoded categorical
observations can be declared with `observation_n_classes: {state_id: 8}`; their
recorded values must be integral IDs in `[0, 8)` and are not normalized.
When authoring a mixed real/categorical embedding ingestion, provide
`feature_embedding_dims` under the normal ingestion rules, including the internal
previous-action column and, when enabled, return column.

Actions use decoder-local IDs in the order of `action_values`, mapped to the
standard global category IDs. A separate input-only ID represents episode start
and is excluded from decoder outputs.
An observation may have the same external name as the action; in that case the
model uses `__policy_action` as its separate target name. Default decoders infer
the target automatically.

Context length must fit `max_context_length`.
The standard ingestion also supports
feature-scoped learned positions when configured.
The standard attention, normalization, position encodings, dropout, and layer
sharing options remain available.

To transfer a forecast or policy backbone, set a top-level
`initialize_from: models/forecast.pt`. Portable models and run checkpoints are
accepted. The source must have the same backbone architecture. Ingestion and
decoder parameters are initialized for the policy's own interface. Resumption
restores the entire existing policy instead of applying initialization again.

## Resume training

Every completed epoch saves a standard `sequifier_run_checkpoint` at
`checkpoints/runs/<model_name>/<model_name>-latest.pt`. It contains current model
weights, preprocessing and policy semantics, optimizer/scheduler/scaler state,
random state, data-order generator state, best weights, and training progress.
The portable model at `model_path` contains the best validation weights; the run
checkpoint contains the latest training weights.

```yaml
global_training:
  resume:
    policy: if_exists
```

`never` is the default. `required` also requires an explicit `checkpoint_path`,
using the shared resume configuration. Policy runs use the same `TrainingRun`,
`TrainingEngine`, evaluation and checkpoint services as supervised runs. Resume
restores the saved optimizer boundary and loader positions. Trajectory content, partition order, policy,
and learning settings must match. Device, output paths, worker/cache settings,
and total epoch budget may change. Restored scheduler state retains its existing
schedule when extending the epoch budget. Numerical reproducibility across
devices is governed by PyTorch.

Private policy run checkpoints written before the shared-engine consolidation
use a different run-state schema. Start a new run with `initialize_from` pointing
to the old model or checkpoint; its execution weights remain loadable. New runs
write ordinary Sequifier run state and loader state.

## Predict actions with `sequifier infer`

Inference reads policy alignment and preprocessing directly from a portable
model or run checkpoint; a training YAML or preprocessing metadata file is not
needed. It accepts `model_type: generative` (or the explicit alias `policy`):

```yaml
project_root: .
model_path: models/policy/policy.pt
data_path: data/observations.csv
read_format: csv
write_format: csv
device: cpu
deterministic: true
desired_return: 10.0
```

See [infer-policy.yaml](example_inputs/configs/infer-policy.yaml). CSV and Parquet
inputs contain observations and any static columns in original units. Without
a `sequenceId` column, each row is an independent first decision. Within each
sequence, `itemPosition` defines decision order when present and must be unique
and contiguous; otherwise row order is used. Static inputs must be constant.
Provide the recorded `action_column` to derive previous actions by shifting one
row, or set `previous_action_column` to a column containing previous **domain
action values**, with null for episode start. Use a new sequence ID for each
episode; null previous actions inside a sequence are rejected. When observation
and action names coincide, use an explicit `previous_action_column` for sequential
CSV/Parquet input. Current recorded actions never
enter the inputs for the same decision. Historical context is bounded by the
model's context length.

Return-conditioned models accept a fixed `desired_return` or a per-row column
named by `desired_return_column` (default `desired_return`). Per-row returns are
supplied by the caller; offline inference does not update budgets from rewards.
BC requires no return input and rejects a scalar `desired_return`.

`inference_batch_size` bounds batched model calls (default 64). Predictions go to `outputs/predictions/<model-stem>/part-000.csv` by default.
`output_path` overrides that location. Sequence IDs and the configured `item_position_column` (default `itemPosition`), when
present, are retained in input order. `output_probabilities: true` adds one
`<action_column>__probability_<action-id>` column per action. Set
`deterministic: false` for seeded categorical sampling. Select `model_interface`
when an artifact contains multiple policy interfaces.

## Execute an environment rollout

Add the ordinary `rollout` configuration to policy inference:

```yaml
autoregressive: true
generation_steps: 1000
rollout:
  mode: policy
  adapter: {factory: my_environment:create_adapter, config: {}}
  bindings:
    - {input: position, source: environment, key: position}
    - {input: velocity, source: environment, key: velocity}
  trajectory_path: outputs/trajectories/episode.json
  seed_index: 0
```

`seed_index` selects one raw input row, forming a single-observation reset seed.
The shared rollout coordinator owns reset, step ordering, boundaries, and adapter
closure. Each predicted action enters policy history only after it is executed.
The [adapter protocol](rollout.md#adapter-protocol) is the same as for forecasts.
The resulting trajectory can be used as a later training source.

The raw return budget starts at `desired_return` (or the seed row's return
column) and updates after each executed reward:

```text
G_next = (G_current - reward) / gamma
```

Each reset restores the requested budget and clears policy history. The trajectory
records action log probabilities, previous-action IDs, and remaining raw budgets.
Termination and truncation remain separate; a collection cutoff preserves the
unfinished-episode flag. The coordinator always closes the adapter.

## Labels, windows, and boundaries

Raw return labels obey `G_t = r_t + gamma * G_(t+1)`, with terminal `G_T = 0`.
`returns.gamma` defaults to 1 and is in `(0, 1]`; `returns.scale` defaults to 1
and divides the return only at ingestion. For rewards `[1, 2, 3]`, gamma 1 gives
`[6, 5, 3]`, and gamma 0.5 gives `[2.75, 3.5, 3]`.

Return-conditioned training excludes truncations and collection cutoffs with
unknown tails. `returns.finite_horizon: H` can label a fully observed prefix of
H decisions. Shorter fragments need true termination. Online finite-horizon
execution requires `rollout.max_episode_steps: H`. No unknown tail is assigned
zero. BC uses **all observed decisions**, including truncated and cutoff episodes,
because its loss does not consume return labels.

Offline and online paths share `PolicyWindowView.inputs`, producing left-padded,
causal histories. The dataset slices to the context window before building
inputs, retaining the actual preceding action at a cropped boundary. Each
training example labels only its last position, so every decision contributes
once. Current actions, current rewards, and future observations are excluded
from inputs; return-to-go is the intentional future-derived condition.

Policy trajectories use `sequifier.scalar-policy-trajectory`, version 1.
Forecast trajectories can be demonstrations with explicit
`action_semantics: executed_predictions` and a finite `action_values` vocabulary.
They retain the latest reset observation and start previous-action history with
the start token, because earlier forecast seed actions are unknown.

## Python interfaces and scope

```python
from sequifier.config.train_config import load_train_config
from sequifier.train import run_training
from sequifier.rl_api import load_policy, PolicyWindowView

config = load_train_config("configs/train-policy.yaml", {}, False)
run_training(config)
policy = load_policy(config.model_path, device="cpu")
# inputs = PolicyWindowView(policy.config).inputs(
#     observations, previous_action_ids, desired_returns, static_inputs
# ).to(policy.device)
# distribution = policy.distribution(inputs)
```

Python callers use `SequifierInputs` for named model tensors and
`PolicyWindowView.inputs` for raw histories. `PolicySequenceDataset` accepts
`SequenceSourceConfig` entries or in-memory `SequenceEpisode` records; JSON logs
are adapted with `source_episodes`. The low-level `train_policy` takes training
and validation datasets plus a `PolicyTrainConfig` from
`sequifier.config.train_config`, sharing the same
options and checkpoint lifecycle as `run_policy_training`.

`CategoricalPolicy` is an alignment adapter over the ordinary network, available
as `policy.network`. `policy.execution_config` is the canonical model config,
compatible with the shared artifact and export services. Distribution methods
include `mode`, `sample`, `log_prob`, and `entropy`. `collect_policy_rollout`
remains available for application-owned adapters. Generic analysis/loading tools
also accept policy model and run artifacts.

The previous standalone numeric-policy artifacts (including versions 1 and 2)
use a different architecture and must be retrained. This workflow currently
supports one categorical actor phase, including the online PPO route below. Mixed supervised/policy plans,
distributed policy learning, custom run integrations, multimodal
trajectory storage, and ONNX policy execution through the CLI remain outside
this workflow. PyTorch policy execution supports CPU, CUDA, and MPS.

## Offline IQL

Use a single `iql` phase to improve an unconditioned categorical actor from a
fixed offline dataset. See the complete
[IQL configuration](example_inputs/configs/train-iql-from-table.yaml). The
actor can start from a compatible BC model with `initialize_from`. IQL creates
separate value, Q1, Q2, and target-Q networks; the portable model artifact
contains only the selected actor. The run checkpoint contains the full learner.
In this categorical slice, V and Q use independent copies of the configured
actor history encoder plus their own scalar heads; their architecture is not
authored in a separate model registry.

Each selected decision needs a finite scalar reward, `terminated`, `truncated`,
and `bootstrap_allowed`. An ordinary nonterminal decision must bootstrap from
its genuine successor observation. An interior row may obtain that successor
from the following row; a final nonterminal or truncated row needs explicitly
mapped successor columns. A true terminal may omit its successor and cannot
bootstrap. Termination takes precedence when both boundary flags are true.
Unknown boundaries and missing rewards are errors for IQL even though BC can
use those demonstrations. Current observations and previous actions form the
actor/value history; Q alone receives the current recorded action. Splits,
categorical fitting, and numeric statistics come only from training episodes.

`updates` is the number of optimizer cycles, each containing one value, critic,
and actor step. The phase controls uniform transition batch size, discount,
expectile, inverse temperature, maximum actor weight, target-Q update rate,
separate optimizer learning rates/specs and scheduler specs, gradient clipping,
and evaluation/checkpoint intervals. Schedulers advance after their own
optimizer update. The held-out action likelihood selects the portable actor;
the report also records value, Q, Bellman, advantage, and weight diagnostics.
The supported per-role optimizers are Adam, AdamW, SGD, and RMSprop.
`global_training.batch_size` and `global_training.learning_rate` remain required
by the shared policy config; the IQL phase's batch size and three learning rates
control its actual updates.
Use a separate application evaluation with `collect_policy_rollout` to compare
returns against BC in a local environment.
The [toy control example](example_inputs/iql_toy_control.py) generates a fixed
mixed-quality dataset for its [BC](example_inputs/configs/train-bc-toy.yaml)
and [IQL](example_inputs/configs/train-iql-toy.yaml) configs. Train BC first;
the IQL config warm-starts from its actor artifact. The example compares saved
actors over 100 fixed initial states and reports mean and standard deviation
rather than asserting an improvement for every seed.

```console
python documentation/example_inputs/iql_toy_control.py generate data/offline-transitions.csv
sequifier train --config-path documentation/example_inputs/configs/train-bc-toy.yaml
sequifier train --config-path documentation/example_inputs/configs/train-iql-toy.yaml
python documentation/example_inputs/iql_toy_control.py evaluate models/toy-bc/toy-bc.pt models/toy-iql/toy-iql.pt
```

`global_training.resume: {policy: if_exists}` resumes the IQL learner from its
complete update checkpoint. `policy: required` needs an explicit
`checkpoint_path`. Exact resume validates the configuration, source content,
split, fitted vocabulary, and normalization. A BC model passed to
`initialize_from` is a warm start, not a learner checkpoint; the actor's policy
schema, normalization, model architecture, and context must match. IQL requires
float32 layers without autocast and supports scalar histories, one categorical
action, and local execution; a
return-conditioned actor, distributed training, mixed precision, continuous or
joint actions, and mixed phase plans are rejected.

Python callers can use `run_iql_training(config)` from `sequifier.rl_api` with a
validated `PolicyTrainConfig`. The return value contains the selected
`CategoricalPolicy`, recorded update metrics, and training/validation episode IDs.

## Online categorical PPO

PPO runs one persistent local environment with one categorical action. Start
with the [complete control config](example_inputs/configs/train-ppo-toy.yaml):

```sh
sequifier train --config-path documentation/example_inputs/configs/train-ppo-toy.yaml
```

`sequifier make` also writes `configs/train-ppo.yaml`. The PPO root schema is
`PPOTrainConfig`: it has `model`, `context_length`, `seed`, `device`, and exactly
one `training_plan.phases` entry with `algorithm: ppo`. It has no
`dataset_training`, offline `sources`, `epochs`, or `global_training`. Phase
fields own both optimizers and their update clocks. See the
[PPO configuration reference](configs/train.md#online-ppo-configuration) and
[Python workflow](example_inputs/ppo_toy_control.py).

The bundled `sequifier.rl.control:CategoricalControlAdapter` alternates a scalar
cue between -1 and +1. Choosing left for -1 and right for +1 earns +1; the other
action earns -1. Each episode executes eight actions. A random policy's expected
return is zero; the optimum is eight. The example config uses CPU, seed 1010,
context length 4, a 32-dimensional one-layer transformer, 256 decisions per
collection, and 8,192 total actions. The training report records hardware,
parameter counts, actual throughput and peak process memory. Learning quality
and multiple-seed improvement have not been measured for this implementation;
no training or test jobs were run for this increment. No CI job was added,
because the implementation request explicitly excluded writing or running tests.

### Online adapter contract

A factory receives `environment.config` as keyword arguments. It must return a
fresh adapter with `protocol_version = 2`, implementing:

```python
from sequifier.rl.environment import OnlineResetResult, OnlineStepResult

class Environment:
    protocol_version = 2
    supports_exact_snapshot = False

    def reset(self, seed: int) -> OnlineResetResult:
        # Return initial domain values and episode-static inputs.
        return OnlineResetResult({"observation": 0.0})

    def step(self, action: dict) -> OnlineStepResult:
        # Execute exactly once. Return the genuine successor, immediate finite
        # scalar reward, and independent termination/truncation flags.
        return OnlineStepResult({"observation": 1.0}, reward=1.0)

    def close(self):
        pass
```

The actor reads observations through the current decision and previous actions
only. After executing an action, its genuine successor and that action enter
the next causal window. An environment must never substitute an autoreset
observation for a final successor. A true terminal may return `observation=None`
and never bootstraps; truncation and collection cuts require a genuine successor
and bootstrap V from it. If both flags are true, termination wins. A horizon
cut preserves the live environment and its bounded causal history for the next
policy version. Only recorded episode boundaries trigger resets before the next
decision. Rewards must be final and immediate; delayed/provisional rewards and
reward providers are outside this contract.

Observations and static fields must exactly match the frozen scalar schema.
Real values must fit finite FP32, categorical values must belong to declared
vocabularies, and units returned at every reset/step must equal the authored
`environment.units` mapping (empty for unitless values). Declare units for each
field whose units matter. Stochastic augmentation is unavailable. Dropout stays
disabled during collection and optimization, while scoring retains gradients.
The existing version-1 seeded inference adapter API is unchanged.
`SeededAdapterBridge` can wrap a v1 environment with same-named environment
observation bindings when an application supplies an initial-state factory.

### Frozen representation and optimization

The policy's observation/static schema, action order, class counts, label maps,
model architecture and context are fixed before the first reset. Author
`normalization: {source: identity}` deliberately, or use
`source: statistics` with a `statistics` map containing exactly the real fields'
`mean` and nonnegative `std`. Zero standard deviations from constant features
use the shared normalizer's epsilon, preserving BC/IQL warm-start statistics.
PPO never estimates statistics from changing online data. `initialize_from`
accepts an ordinary unconditioned BC/IQL actor only if
its policy schema, action ordering, normalization, model and context match
exactly. It initializes actor weights, not the learner state. A
return-conditioned actor is rejected.

A separate history-only V owns independent parameters; PPO creates no Q or
target-Q networks. Each buffer stores a single policy generation, old action
log-probabilities, old values and true successor values. Compact frames retain
pre-buffer context; minibatches reconstruct ordered, complete windows.
FP32 GAE breaks its recurrence at episode boundaries and horizon cuts, while
bootstrapping at nonterminal boundaries. Advantage normalization occurs once per
iteration. Old scores, advantages and targets stay fixed for all epochs.

The objective uses clipped probability ratios, independent squared value loss,
and current-policy categorical entropy. Optional value clipping is centered on
old V; it defaults off. Probability math is FP32. Gradient accumulation and the
last partial minibatch are weighted by their number of actual decisions. Actor
and V schedulers advance independently after their optimizer steps. Optional
KL limits stop further optimizer groups. Non-finite inputs, scores, losses,
gradients and parameters fail the run instead of admitting invalid decisions.

### Evaluation, artifacts and resuming

Evaluation uses a fresh adapter, frozen actor snapshot, fixed evaluation seeds,
and separate action/reset seed streams. It restores Python, NumPy and torch RNG
state afterward. Default evaluation selects mode actions; use
`selection: stochastic` explicitly for sampled actions. `max_episode_steps`
and `timeout_seconds` are mandatory. Evaluation runs on the main thread of a
local POSIX process, with an interval-timer deadline across creation, resets
and steps. Adapter methods must be interruptible and `close()` must return
promptly. A factory must return a different object from the live training slot.

Reports include completed returns and their spread, episode lengths, completed
and censored counts, and success rates only when the adapter supplies success.
An environment truncation or evaluation step cap is censored; it does not count
as a completed episode. Best-actor selection defaults to maximizing
`mean_completed_return`; selection can instead use `mean_completed_length` or
`success_rate` with an explicit direction. No completed episodes means the
return metric is unavailable. If no evaluation supplies the selection metric,
the latest actor is exported with `latest_actor_metric_unavailable` recorded;
it is not described as a best actor. Initial evaluation also participates in
best-actor selection.

The default outputs are `models/<name>/<name>.pt`,
`outputs/training/<name>/report.json`, and `checkpoints/<name>/ppo.pt`.
The portable model contains actor weights only and loads with `load_policy`,
`sequifier infer`, and `collect_policy_rollout`. The example includes ordinary
[inference](example_inputs/configs/infer-ppo-toy.yaml) and seeded
[rollout](example_inputs/configs/rollout-ppo-toy.yaml) configs:

```sh
sequifier infer --config-path documentation/example_inputs/configs/infer-ppo-toy.yaml
sequifier infer --config-path documentation/example_inputs/configs/rollout-ppo-toy.yaml
```
 The learner checkpoint contains
actor/V weights and specs, both optimizers/schedulers and step clocks, the
frozen codec, config fingerprint, best actor, seed/RNG state, live history and
episode counters. Checkpoints commit atomically only at complete iteration
boundaries. A crash during collection or optimization restarts from the last
committed iteration; partial buffers are not resumed.

Set the phase's `resume.policy` to `never`, `if_exists`, or `required` and choose
one of these declared `resume.mode` values:

- `exact` requires `supports_exact_snapshot = True` and working
  `snapshot_state()` / `restore_state(state)` methods. The snapshot must include
  simulator and simulator RNG state using dictionaries, lists, tuples, finite
  scalars, numeric NumPy arrays and torch tensors. Restore is checked against a fresh snapshot.
  Merely exposing `state_dict` does not declare this capability. Exact mode
  rejects unsupported adapters before collection. It preserves the live
  history and episode counters; equivalence still depends on deterministic
  adapter and device operations.
- `reset_environments` restores learner, action sampler and optimizer state,
  then resets the environment and causal history. It discards incomplete
  episode metrics and records a non-trajectory-equivalent continuation. Reset
  counts continue forward, so the next reset uses a new derived seed.

Resume validates the full run contract, including budget, model, codec, reward
scale, adapter identity/config and normalization. Resume policy and mode may
change; other changes require a new run. Evaluation seeds never affect training
reset counts. Optional `archive_directory` writes atomic, version-1
`sequifier.ppo-collection` records, with pre-fragment context, identities,
policy generation, cutoff reason, raw/scaled rewards and true successors. `archive_scores: true`
also retains old log-probability, entropy and values. The independent
`load_collection_record` reader does not change existing v1 JSON episode
readers. Archives are never fed back into online updates.

The report records return/evaluation curves, policy/value losses, entropy, clip
fraction, approximate KL, explained variance, gradient norms, ratio tails,
optimizer clocks, counts and separate collection/update/evaluation timing.
Invalid data fails immediately; successful reports therefore have zero invalid
episodes. A nonterminated final training episode is explicitly censored.

This scope is one in-process adapter slot and one fixed categorical action on
CPU or a supported local CUDA/MPS device. Vector/Gym/remote adapters,
continuous/joint actions, media/depth, distributed learning, reward providers,
mixed phases, same-run phase transfer, generalized monitors/search and ONNX
policy execution remain deferred. This does not complete the broader rollout
and reinforcement-learning plan.
