# Policy learning through `sequifier train` and `sequifier infer`

Policy learning uses Sequifier's configurable ingestion, transformer network,
decoders, portable model artifacts, optimizer transactions, and run checkpoints.
A policy adds a decision-alignment contract to a model interface: at time `t`,
the inputs contain the current observation, the **previous** action, and,
optionally, a desired return. The current action is the supervised target.

## Train from ordinary sequence tables

Use the existing `sequenceId` and `itemPosition` conventions. Each row describes
**the current observation, the action executed at that observation, and the
reward/boundary resulting from that action**. One sequence is one episode or an
explicit episode fragment. CSV and Parquet files can contain many episodes;
files are not assumed to be episode boundaries. Positions must be nonnegative,
unique and contiguous within each sequence, including across input shards.

```yaml
project_root: .
model_name: policy
data_sources:
  demonstrations:
    paths: [data/demonstrations.parquet]
policies:
  actor:
    observation_columns: [position, velocity]
    action_column: action
    action_values: [left, right]
    context_length: 16
    return_conditioned: false
training_plan:
  algorithm: return_conditioned
  epochs: 80
  sources: [{source: demonstrations}]
  policy: actor
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
data_sources:
  demonstrations:
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
and set `validation_sources: [held_out]` on the phase. Duplicate paths and episode
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
policy image/depth ingestion and IQL/PPO remain planned.

A source can still specify `paths: [episode-a.json, episode-b.json]` or a legacy
`manifest` listing version-1 JSON episodes. These are validated rollout recordings
adapted into the same sequence representation, with bindings retained in their
recording format. Forecast recordings require
`action_semantics: executed_predictions`; this does not convert forecast checkpoint
alignment into policy alignment. The old `trajectory_sources` registry and phase
field remain aliases for `data_sources` and `sources`.

Run paths resolve relative to `project_root`; manifest entries resolve relative
to the manifest directory. `optimization.num_workers` controls loader workers;
`episode_cache_size` bounds cached episodes per worker. Table-backed episodes use
filtered lazy scans. Prepared window reconstruction is eager per source and JSON
records still require one episode to fit in memory.

The best model is selected by held-out action NLL. `model_path` defaults to
`models/<model_name>/<model_name>.pt` and contains an ordinary `sequifier_model`.
`actor_path` remains an alias. The report records partitions, exclusions, the fitted
policy codec, decision counts and per-epoch NLL. Epochs belong to the phase, seed
and device to the run, and optimizer/scheduler settings to `optimization`.

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
`data_sources.demonstrations.manifest: data/imported-demonstrations/manifest.json`;
the recorded column mappings are loaded automatically. The manifest retains
source provenance, episode identities, optional successors and collection groups.

## Shared model and dataset authoring

The compact `policies` surface is optional. Policy training also accepts the
ordinary `model.backbone` and `model.interfaces` schemas, with a `policy` data
schema on each interface. Ingestion and decoder components keep their existing
syntax; the policy adapter adds the previous-action/return features. Use top-level
`context_length`, or `global_training.context_length`, to select a context smaller
than the backbone maximum. `global_training` accepts the policy optimization
settings as an alternative to `optimization` and uses `training_objective: policy`.

Likewise, `dataset_training.<name>` accepts `model_interface` and `parts`, with each
part containing a sequence source specification. Reference it with the ordinary
`{source: dataset.part}` phase syntax. The algorithm supplies action NLL, so these
parts do not need fabricated criteria. Model/interface references are validated.
The shared phase schema preserves supervised defaults; this learner still supports
one `return_conditioned` phase, with BC selected on its policy. Mixed algorithms,
backbone repositories, image/depth policies and online learning remain planned.
Use `initialize_from` for backbone artifact transfer.

See [train-policy-from-table.yaml](example_inputs/configs/train-policy-from-table.yaml)
for a complete example using the shared model, dataset and phase authoring surface.

## Configure the shared model

The policy's `backbone` accepts the architecture fields from ordinary training's
`model.backbone.architecture`. Optional `ingestion` and `decoder` use the same
component schemas and builders as forecasting. For example:

```yaml
policies:
  actor:
    observation_columns: [position, velocity]
    action_column: action
    action_values: [-1.0, 1.0]
    context_length: 16
    return_conditioned: true
    backbone:
      dim_model: 64
      num_layers: 3
      max_context_length: 16
      attention: {n_heads: 4}
      feed_forward: {dim: 128, activation: swiglu}
      positional_encoding_scope: global
    decoder: {type: mlp, hidden_dims: [64], prediction_length: 1, support: 1}
```

Without an authored ingestion, the standard embedding ingestion projects real
features and embeds categorical features before an adapter maps to the backbone
input width. Real features receive one embedding channel and categorical features
share the remaining width. The default decoder is linear. A policy decoder must
have `support: 1` and `prediction_length: 1`.

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

The default backbone has width 32, two layers, one attention head, and
feed-forward width 128. Context length must fit `max_context_length`.
Positional scope defaults to `global`; the standard ingestion also supports
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
optimization:
  resume:
    policy: if_exists
```

`never` is the default. `required` also requires an explicit `checkpoint_path`,
using the shared resume configuration. Resume occurs at epoch boundaries;
an interrupted epoch is repeated. Trajectory content, partition order, policy,
and learning settings must match. Device, output paths, worker/cache settings,
and total epoch budget may change. Restored scheduler state retains its existing
schedule when extending the epoch budget. Numerical reproducibility across
devices is governed by PyTorch.

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

Offline and online paths share `build_policy_inputs`, producing left-padded,
causal histories. The dataset slices to the context window before building
inputs, retaining the actual preceding action at a cropped boundary. Each
training example labels only its last position, so every decision contributes
once. Current actions, current rewards, and future observations are excluded
from inputs; return-to-go is the intentional future-derived condition.

Policy trajectories use `sequifier.numeric-policy-trajectory`, version 1.
Forecast trajectories can be demonstrations with explicit
`action_semantics: executed_predictions` and a finite `action_values` vocabulary.
They retain the latest reset observation and start previous-action history with
the start token, because earlier forecast seed actions are unknown.

## Python interfaces and scope

```python
from sequifier.config.train_config import load_train_config
from sequifier.train import run_training
from sequifier.rl_api import load_policy, build_policy_inputs

config = load_train_config("configs/train-policy.yaml", {}, False)
run_training(config)
policy = load_policy(config.model_path, device="cpu")
# inputs = build_policy_inputs(policy.config, observations, previous_action_ids,
#                              desired_returns, static_inputs).to(policy.device)
# distribution = policy.distribution(inputs)
```

`CategoricalPolicy` is an alignment adapter over the ordinary network, available
as `policy.network`. `policy.execution_config` is the canonical model config,
compatible with the shared artifact and export services. Distribution methods
include `mode`, `sample`, `log_prob`, and `entropy`. `collect_policy_rollout`
remains available for application-owned adapters. Generic analysis/loading tools
also accept policy model and run artifacts.

The previous standalone numeric-policy artifacts (including versions 1 and 2)
use a different architecture and must be retrained. This workflow currently
supports one offline categorical actor phase. Mixed supervised/policy plans,
critics, IQL/PPO, distributed policy learning, custom run integrations, multimodal
trajectory storage, and ONNX policy execution through the CLI remain outside
this workflow. PyTorch policy execution supports CPU, CUDA, and MPS.
