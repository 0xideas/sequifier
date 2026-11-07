# Policy Learning Command Guide

`sequifier policy` runs a local, CPU-only categorical control experiment:

```console
sequifier policy --config-path documentation/example_inputs/configs/policy-control.yaml
```

It collects trajectories, computes return-to-go labels, splits by episode,
trains return-conditioned and behavior-cloning actors, saves/reloads both, and
executes them through the existing rollout coordinator. `report.json` records
the achieved returns. This command is separate from forecast model training and
inference; it does not change `target_offset` or convert forecast checkpoints
into policies. `sequifier docs policy` prints this guide.

## Task and collected data

The deterministic binary control task starts at `position=0` with three decisions
remaining. Action `0` holds position; action `1` increments it. Each action earns
its numeric value as reward. Observations contain current `position` and
`remaining_steps`. The environment terminates after its last decision and returns
the actual final successor. The horizon is configurable from 1 through 8.

Collection enumerates every binary action sequence, repeating each 16 times by
default. For horizon 3 that gives 128 episodes and 384 executed decisions from
the same starting conditions, with returns from 0 through 3 and a behavior mean
of 1.5. Repetition gives conflicting action labels for identical observations
while retaining examples at each return level. The dataset manifest lists the
exact collected files and their episode IDs; stale files are not added to the
current experiment.

The example configuration writes these files beneath `outputs/numeric-policy`:

- `dataset/sequence-*.json`: recorded episodes in original numeric units.
- `dataset/manifest.json`: exact source files, episode IDs, and raw returns.
- `dataset/return_labels.json`: the return definition and raw per-decision labels.
- `return_conditioned.pt` and `behavior_cloning.pt`: local actor artifacts.
- `evaluation/*.json`: trajectories executed using the reloaded actors.
- `report.json`: episode partitions, per-epoch action likelihood, checkpoint
  selection, greedy/sampled actions, achieved returns, and requested-return errors.

The report distinguishes raw episode reward sums from discounted achieved
returns. It compares both actors at the same requested returns, initial states,
and sampling seeds. The BC actor does not receive the requested return. This is
a controlled learning comparison; performance depends on training configuration
and is recorded rather than assumed.

## Policy alignment and causal windows

At decision position `t`, the policy receives `o_t`, `a_(t-1)`, and, for return
conditioning, `G_t`. The current action `a_t` is a separate target. The first
previous action is a dedicated start-token ID equal to the number of executable
actions. It cannot be selected by the action head.

For the sequence `[1, 0, 1]`, gamma 1, and the default task:

| Decision | Current observation | Previous action ID | Desired return | Action target |
| --- | --- | --- | --- | --- |
| 0 | `(position=0, remaining_steps=3)` | 2 (start) | 2 | 1 |
| 1 | `(position=1, remaining_steps=2)` | 1 | 1 | 0 |
| 2 | `(position=1, remaining_steps=1)` | 0 | 1 | 1 |

Offline and online execution use `build_policy_inputs` to construct the same
left-padded causal windows. Observations, previous actions, and returns are
trimmed together. Padding has a false attention mask. Only the last position
of each training example has a true action-likelihood mask, so every recorded
decision contributes once even when context windows overlap.

The actor uses numeric projection, previous-action embedding, a small causal GRU,
and one categorical head. Padded positions do not advance its hidden state.
Forward calls take `PolicyInputs`, which contain no action labels or observed
current rewards. A change to a current-action label cannot affect that position's
inputs; future observations are excluded from its window. Return-to-go is the
intentional reward-derived training condition, with requested budgets replacing
it online. Changing rewards can therefore change the return condition.

Observation/static normalization is fit on the training episode partition only.
Its fixed means/scales are model buffers, saved with the actor. The RC and BC
actors use the same architecture, initialization seed, episode split, batch
order, and optimization configuration. BC disables the return channel by
feeding zero; it retains the same parameter count.

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

The public entry points live in `sequifier.rl_api`. To train on compatible saved
policy episodes:

```python
from sequifier.rl_api import (
    CategoricalPolicyConfig, PolicyTrainingConfig, TrajectoryDataset,
    load_policy, save_policy, split_by_episode, train_policy,
)

spec = CategoricalPolicyConfig(
    observation_columns=["position", "remaining_steps"],
    action_column="action",
    action_values=[0.0, 1.0],
)
dataset = TrajectoryDataset(episode_paths, spec)
training, validation = split_by_episode(dataset, validation_fraction=0.2, seed=1010)
result = train_policy(training, validation, PolicyTrainingConfig())
save_policy(result.policy, "actor.pt")
actor = load_policy("actor.pt")
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

```python
from sequifier.rl_api import collect_policy_rollout
from sequifier.rollout.control import BinaryControlAdapter
from sequifier.rl.experiment import control_rollout_config, control_seed
from sequifier.rollout_api import save_trajectory

rollout = control_rollout_config("execution.json", horizon=3)
trajectory = collect_policy_rollout(
    actor, BinaryControlAdapter(horizon=3), control_seed(horizon=3),
    rollout, max_steps=3, desired_return=3.0, deterministic=True,
)
save_trajectory(trajectory, rollout.trajectory_path)
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
