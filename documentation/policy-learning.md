# Policy learning through `sequifier train` and `sequifier infer`

Policy learning uses Sequifier's configurable ingestion, transformer network,
decoders, portable model artifacts, optimizer transactions, and run checkpoints.
A policy adds a decision-alignment contract to a model interface: at time `t`,
the inputs contain the current observation, the **previous** action, and,
optionally, a desired return. The current action is the supervised target.

## Train from trajectories

```console
sequifier train --config-path configs/train-policy.yaml
sequifier infer --config-path configs/infer-policy.yaml
```

A complete training configuration is in
[train-return-conditioned.yaml](example_inputs/configs/train-return-conditioned.yaml).
`sequifier make` also creates `configs/train-policy.yaml`. Define named
`trajectory_sources`, named `policies`, and one `training_plan` phase with
`algorithm: return_conditioned`. Set `return_conditioned: false` on the policy
for behavior cloning (BC). Offline training does not start an environment.

Each source specifies either `paths: [episode-a.json, episode-b.json]` or
`manifest: data/trajectories/manifest.json`. Version-1 manifests contain an
`episodes` list of objects with `path` and optional `episode_id`. Only listed
files are selected. Run paths resolve relative to `project_root`; manifest
entries resolve relative to the manifest directory.

Set `epochs` on the phase, `seed` and `device` on the run, and update settings
in `optimization`. Devices follow PyTorch's CPU, CUDA, and MPS conventions.
`optimization.num_workers` controls data-loader workers and
`optimization.episode_cache_size` bounds the number of cached file-backed
episodes per worker (default 8). The learner retains episode metadata and loads
episodes on demand. It shuffles episodes and the decisions within each episode
to keep disk access local. Normalization combines episode statistics
incrementally. JSON storage still requires an individual episode to fit in memory.

Without `validation_sources`, the runner splits whole episodes deterministically,
keeping identical demonstrations together. At least two distinct trajectory
groups are required. To supply held-out data, declare a separate source and add
`validation_sources: [held_out]` to the phase. Duplicate paths and episode IDs,
and identical content across training and validation, are rejected.
Normalization fits only the training partition.

The best model is selected by held-out action negative log likelihood (NLL).
`model_path` defaults to `models/<model_name>/<model_name>.pt` and contains an
ordinary `sequifier_model` artifact. `actor_path` remains an alias for `model_path` for existing configurations. `report_path`
defaults to `outputs/training/<model_name>/report.json`; it records sources,
partition IDs, exclusions, decision counts, per-epoch NLL, and optimizer steps.

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
are broadcast over the decision window. Real observation and static statistics
are persisted in the model interface's preprocessing metadata. Encoded categorical
observations can be declared with `observation_n_classes: {state_id: 8}`; their
recorded values must be integral IDs in `[0, 8)` and are not normalized.
When authoring a mixed real/categorical embedding ingestion, provide
`feature_embedding_dims` under the normal ingestion rules, including the internal
previous-action column and, when enabled, return column.

Actions use the ordered IDs of `action_values`; an extra input-only ID equal to
the action count represents episode start. It is excluded from decoder outputs.
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
sequence, row order defines decision order and static inputs must be constant.
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
`output_path` overrides that location. Sequence IDs and `itemPosition`, when
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
