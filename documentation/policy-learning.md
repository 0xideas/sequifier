# Policy training through `sequifier train`

Train an actor on saved trajectories with:

```console
sequifier train --config-path documentation/example_inputs/configs/train-return-conditioned.yaml
```

The configuration declares named `trajectory_sources`, named `policies`, and one
`training_plan.phases` entry with `algorithm: return_conditioned`. Ordinary
training phases default to `algorithm: supervised`. Policy phases have their own
source and model contract and require no supervised dataset, criterion, or
preprocessing metadata. This increment supports CPU execution and one categorical
actor in one phase. Mixed algorithm plans, phase transfer, IQL, PPO, distributed
policy learning, custom training integrations, and policy export remain planned.
Unsupported run fields are rejected before learning.

A trajectory source specifies either `paths: [episode-a.json, episode-b.json]` or
`manifest: dataset/manifest.json`. A version-1 manifest has an `episodes` list of
objects with a `path` for each file. Only listed files are loaded. Run paths resolve
relative to `project_root`; manifest entries resolve relative to the manifest's
directory. These JSON episodes are the interim numeric trajectory format, preceding
the plan's sharded multimodal storage.

The phase references `trajectory_sources: [demonstrations]` and `policy: actor`.
Set `epochs` on the phase and `seed` on the run. Optimizer, scheduler, batch size,
gradient clipping and automatic `validation_fraction` belong in `optimization`.
The actor's `returns` declares gamma, scale, and optional finite horizon. A concise
`training_plan: {algorithm: return_conditioned, epochs: 80, trajectory_sources:
[demonstrations], policy: actor}` is also accepted.

Without `validation_sources`, the runner splits whole episodes deterministically,
keeping identical recorded trajectories together. Repeated copies therefore do
not cross the partition boundary. At least two distinct trajectory groups are
required. For authored partitions, add `validation_sources: [held_out]` to the
phase and declare that separate source in the registry. Duplicate file paths,
episode IDs, or identical trajectory content across partitions are rejected.
Normalization is fit only on training episodes. Truncations and collection cutoffs
with unknown returns are excluded under the existing return-label contract;
excluded IDs are recorded in the report.

The runtime uses the existing optimization transactions and selects the best
actor by held-out action NLL. `actor_path` defaults to
`models/<model_name>/actor.pt`; `report_path` defaults to
`outputs/training/<model_name>/report.json`. The report records resolved config,
selected files, partition IDs, decision counts, excluded episodes, per-epoch NLL,
optimizer steps and checkpoint selection. Actor files contain model weights,
normalization buffers, vocabulary, alignment and return settings. They are
actor-only inference artifacts; optimizer and RNG state for exact resume are not
included. Use `load_policy` and `collect_policy_rollout` to execute them.

For behavior cloning, set the actor's `return_conditioned: false` in the same
workflow. Offline training requires only saved trajectories; it does not start an
environment. Python callers use `load_train_config` and `run_training` with the
same configuration as the CLI.

`sequifier docs policy` prints this guide.

## Policy alignment and causal windows

At decision position `t`, the policy receives `o_t`, `a_(t-1)`, and, for return
conditioning, `G_t`. The current action `a_t` is a separate target. The first
previous action is a dedicated start-token ID equal to the number of executable
actions. It cannot be selected by the action head.

Offline and online execution use `build_policy_inputs` to construct the same
left-padded causal windows. Observations, previous actions, and returns are
trimmed together. Padding has a false attention mask. Only the last position
of each training example has a true action-likelihood mask, so every recorded
decision contributes once even when context windows overlap.

The actor uses numeric projection, previous-action embedding, the same
`TransformerBackbone` and encoder layers as `sequifier train`, and one categorical
head. Attention is causal and excludes padded keys; padded logits are zero.
The model recomputes each finite context window rather than retaining recurrent
state across windows.
Forward calls take `PolicyInputs`, which contain no action labels or observed
current rewards. A change to a current-action label cannot affect that position's
inputs; future observations are excluded from its window. Return-to-go is the
intentional reward-derived training condition, with requested budgets replacing
it online. Changing rewards can therefore change the return condition.

Each entry in the `policies` registry and `CategoricalPolicyConfig` accepts a
`backbone` section using the same architecture fields as training's
`model.backbone.architecture`:

```yaml
context_length: 4
backbone:
  dim_model: 32
  num_layers: 2
  max_context_length: 4
  attention: {type: mha, n_heads: 4}
  feed_forward: {dim: 128, activation: swiglu}
  normalization: {type: rmsnorm, norm_first: true}
  position_encoding: {type: learned}
  positional_encoding_scope: global
  dropout: 0.0
  shared_layer_groups: []
```

This includes MHA/MQA/GQA, feed-forward activation, normalization, positional
encoding (learned, RoPE, sinusoidal, range or range-concat), dropout and layer
sharing. `context_length` must fit `max_context_length`. Learned positions require
`global` scope because policy inputs use a combined numeric projection rather
than per-feature position embeddings. Scope defaults to `global` for policies.
Without `backbone`, defaults are two layers, width 32, one attention head and
feed-forward width 128. Configure width with `backbone.dim_model`; the experimental
`hidden_size` option has been removed.

Transformer actor artifacts use version 2. Version 1 GRU artifacts cannot be
loaded into this architecture and must be retrained. Sharing backbone primitives
does not yet provide forecast-checkpoint transfer. The numeric policy input
projection, action head and CPU-only execution remain policy-specific.

Observation/static normalization is fit on the training episode partition only.
Its fixed means/scales are model buffers, saved with the actor. BC disables the return channel by feeding zero and retains the same parameter
count as the corresponding return-conditioned actor.

## Return definitions and episode boundaries

Raw labels use `G_t = r_t + gamma * G_(t+1)` and terminal `G_T = 0`.
For rewards `[1, 2, 3]`, gamma 1 produces `[6, 5, 3]`; gamma 0.5 produces
`[2.75, 3.5, 3]`. `returns.gamma` is explicit in `(0, 1]`, defaulting to 1.
`returns.scale` divides returns only at the numeric encoder boundary and defaults
to 1. Artifacts and execution traces preserve raw reward units.

By default, only trajectories with true termination have exact return labels.
Collection cutoffs and truncation without termination are excluded and their
episode IDs are exposed by the dataset. An explicit `returns.finite_horizon: H`
allows labeling an observed prefix of exactly H decisions, including a cutoff
at H. A fragment shorter than H is still excluded unless it truly terminates.
No unknown tail is silently assigned zero. For online use of this objective,
`rollout.max_episode_steps` must equal H.

Execution starts from the caller's requested return and updates the raw budget
after each executed reward:

```text
G_next = (G_current - reward) / gamma
```

The request is not clamped to an inferred feasible range. Each episode reset
restores the requested budget, clears previous-action/history state, and uses
the start token again. A collection cutoff preserves the unfinished episode
flag; it does not fabricate a reset. Each call collects one episode/fragment,
so collecting another episode uses a fresh adapter and an explicit reset.

Policy trajectories use `format: sequifier.numeric-policy-trajectory`, version 1,
with the same seed/transition schema and storage as numeric forecast trajectories.
`mode: policy` makes this alignment explicit. Forecast artifacts retain their
existing format and require an explicit declaration before their executed
predictions can be used as demonstration actions.
Policy episodes start with one valid observation because historical actions
before reset are unknown. The actor builds subsequent history from executed
transitions.

Execution metadata stores the full policy/return configuration, initial request,
remaining budgets (including the final successor budget), previous-action IDs,
and selected-action log probabilities. The trajectory validator checks action
vocabulary, previous-action alignment, budget counts, and reward/gamma arithmetic
in addition to ordinary continuity, reward, and boundary validation. Both flags
may be true; termination takes precedence for bootstrap. Adapter closure remains
owned by the shared coordinator on success and failure.

## Reusable Python interfaces

The public policy entry points live in `sequifier.rl_api`. Python callers can
use the same configuration loader and training entry point as the CLI:

```python
from sequifier.config.train_config import load_train_config
from sequifier.train import run_training
from sequifier.rl_api import load_policy

config = load_train_config("configs/train-policy.yaml", {}, False)
run_training(config)
actor = load_policy(config.actor_path)
```

First-increment forecast trajectories can also supply demonstrations when their
executed numeric predictions have a declared finite categorical vocabulary:
pass `action_semantics="executed_predictions"` to `TrajectoryDataset` and set
the appropriate `action_column` and `action_values` in the policy specification.
This is an explicit data interpretation, not a conversion of forecast model
alignment or weights. It uses the latest seed observation as decision zero,
discards earlier seed observations whose actions were never recorded, and
starts previous-action history with the start token. Episode splitting preserves
this declaration. Observation and action names may coincide because their
values are read from separate observation/target records; current labels are
never copied into current inputs.

Action values are original numeric domain values; the distribution uses their
ordered categorical IDs internally. `actor.distribution(inputs)` returns a
`CategoricalActionDistribution` supporting `mode()`, `sample(generator)`,
`log_prob(recorded_action_ids)`, and `entropy()`. `masked_action_likelihood`
uses the dataset's independent action-valid mask. Training calls the existing
`OptimizationRuntime` for backward, clipping, optimizer, scaler, and scheduler
steps. The validation partition selects the saved best actor by action NLL.

For execution, supply an application-owned environment adapter, a validated
`SeedHistory`, and a `RolloutConfig` with `mode: policy`. Bind each changing input
to the appropriate environment field or executed action. The adapter follows the
[rollout protocol](rollout.md#adapter-protocol).

```python
from sequifier.rl_api import collect_policy_rollout
from sequifier.rollout_api import save_trajectory

# actor, adapter, seed, rollout_config and requested_return come from the application.
trajectory = collect_policy_rollout(
    actor, adapter, seed, rollout_config,
    max_steps=1000, desired_return=requested_return, deterministic=True,
)
save_trajectory(trajectory, rollout_config.trajectory_path)
```

Set `return_conditioned=False` in the policy specification to train BC, and omit
`desired_return` when executing it. The actor artifacts contain validated CPU
weights, normalization buffers, action vocabulary, policy alignment, and return
semantics. Loading uses strict state-dict matching. These local inference
artifacts do not contain optimizer/RNG state for exact training resumption and
do not use forecast export or inference formats.

This increment has no critics, Q networks, PPO buffers, distributed execution,
or policy export additions. Return conditioning and unconditioned behavior
cloning are independent actor routes. Future offline IQL can reuse the
trajectory windows and distribution interface without requiring RC pretraining.
