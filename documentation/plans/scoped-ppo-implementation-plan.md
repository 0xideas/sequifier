# Scoped PPO implementation plan for the current policy-learning path

Status: proposed implementation plan, 2026-11-09. This is a follow-up to the
[broader rollout and reinforcement-learning plan](rollout-and-reinforcement-learning-implementation-plan.md),
especially sections 3, 4, 5.1, 5.4, and 6. The target is **online PPO at the
current behavior-cloning and return-conditioned imitation product boundary**:
one scalar-observation, single-categorical-action policy can be trained through
`sequifier train`, evaluated in a local environment, checkpointed, exported as
the ordinary actor model, loaded by `sequifier infer`, and executed through the
same rollout machinery. This does not claim completion of full Stage 3.

## 1. Baseline and scope

The diff since `2d01d407cd2134c8197b1a18f0034d01486f18a9` adds opt-in
single-environment scalar forecast rollouts; validated version-1 JSON episode
records; common columnar episode imports from CSV/Parquet, prepared scalar data,
and JSON recordings; causal policy windows; one categorical distributional
actor; BC and return-conditioned imitation; ordinary actor artifacts and
format-v2 imitation run checkpoints; and policy inference/rollouts. The latest
commit adds one fixed-dataset IQL phase with V and twin Q/target Q networks,
transition sampling, weighted actor extraction, an IQL learner checkpoint, and
actor-only export. The IQL phase is single-process and single-categorical; it
does not add an online environment loop or PPO infrastructure.

The current `collect_rollout` resets one seed-initialized episode, stops at an
episode end or budget, and always closes the adapter. `PolicyExecutor` builds
causal windows and samples actions, but its current execution path returns
domain action labels and records log-probabilities as side metadata. It does not
retain value estimates or a reusable live environment. `PolicyTrainConfig`
requires `dataset_training` and one offline phase; `load_train_config` rejects
PPO. The existing `HistoryCritic` can serve as a separate history-only V, while
the IQL Q networks and offline transition sampler are not PPO components.

**Included:** one persistent in-process adapter with one environment slot;
scalar real or categorical observation/static fields already supported by the
policy codec; one fixed categorical action vocabulary; fixed-length causal
context; single-process CPU and supported local torch devices; synchronous
collect/update iterations; local deterministic rollout evaluation; explicit
resume behavior; Python and CLI entry points; actor-only portable artifacts.
An unconditioned BC or IQL actor may initialize PPO after strict compatibility
checks. PPO from scratch uses an explicit, frozen observation codec and
normalization declaration, including a deliberate identity-normalization
choice; it never fits statistics from its changing on-policy data.

**Deferred:** vector/Gym/remote adapters, media/depth, continuous or joint
actions, DDP/FSDP, reward providers and delayed rewards, mixed phases or
same-run phase transfer, generalized rollout monitors/search, ONNX policy
execution, and sharded trajectory storage. Return-conditioned actors are
rejected for this PPO slice; dropping the desired-return input would silently
change the actor contract. Existing BC/return-conditioned/IQL behavior remains
available through their current routes.

## 2. Contracts to settle first

1. **Decision timing.** At decision `t`, actor and V read observations through
   `o_t` and actions only through `a_(t-1)`. Execute `a_t`, then record its
   reward, genuine `o_(t+1)`, `terminated`, and `truncated`. The successor V
   window includes `o_(t+1)` and `a_t` as its previous action. Never use an
   autoreset observation as that successor. Continue to permit both boundary
   flags, with termination taking precedence for bootstrap.
2. **Environment lifecycle.** Add an opt-in online local adapter protocol (or
   versioned extension to `EnvironmentAdapter`) whose `reset(seed)` returns the
   initial observation and static values, and whose `step(action)` returns the
   true successor, scalar finite reward, and boundary flags. One instance lives
   from training start through all collect/update iterations and closes in
   `finally`. Keep the existing seed-initialized, version-1 inference adapter
   callable unchanged; provide a narrow compatibility wrapper where possible.
   Reset only after a recorded boundary, before the next decision. A local
   adapter may optionally declare and implement exact state snapshots; merely
   having `state_dict` methods is insufficient.
3. **History and identity.** Move causal history maintenance into a per-slot
   state object shared by `PolicyExecutor`, collection, and evaluation. It
   stores the current observation, previous action/start token, static values,
   attention mask, episode ID, step ID, and reset count. A collection horizon
   cuts a live episode without resetting it. Derive seeds from run seed,
   evaluation/training stream, and reset count; do not mix evaluation resets
   into training RNG state. Assert that a policy version is constant within a
   collected buffer; a live episode may span versions across buffers.
4. **Frozen representation.** Pin the observation/action schema, action-value
   order, normalization statistics, context length, and deterministic input
   transforms before the first reset. Use domain values at the adapter edge and
   the existing codec for model tensors. Reject schema/shape/unit drift and
   non-finite values before insertion. Disable dropout and stochastic
   augmentation for collection and PPO scoring while retaining gradients in
   scoring. Make `evaluate_actions` and `act` use FP32 probability math.
5. **Rewards and boundaries.** This slice optimizes the adapter's one scalar
   reward with a fixed, checkpointed scale (default 1). Reject provisional or
   delayed rewards. A true terminal has no bootstrap; a time-limit truncation
   bootstraps only from its genuine final state; a collection cut is a fragment
   and bootstraps without a reset. Reject a requested bootstrap when the true
   successor is unavailable.

**Gate A:** hand-authored resets, first/interior/final decisions, truncations,
cuts, and episode restarts prove alignment. Replacing current action, reward,
or future observations cannot change step-`t` actor/V outputs. Existing
seed-based rollout behavior and version-1 JSON loading still work.

## 3. Dependency-ordered implementation packages

### B. Persistent collection and on-policy data

Build a `PPOCollector` over the shared history/adapter contract. At each
iteration freeze an actor/value snapshot and collect exactly `rollout_length`
decisions, except a declared final shorter iteration when `total_env_steps` is
not divisible by the horizon. The budget counts executed actions, including
terminal actions. A completed episode resets immediately for the next decision;
a live episode and its history continue into the next iteration. Collection
runs under `no_grad` and records one policy version for every decision.

Add a sealed, bounded `PPORolloutBuffer` with decision references to a compact
observation/action history ring rather than a copied context window per step.
Each entry holds domain/encoded action, old joint log-probability, old V,
reward, true successor/final observation reference, termination/truncation,
bootstrap eligibility, episode/step identity, and policy version. Retain the
pre-buffer context needed to reconstruct the earliest decision window. Store
old V of each true successor or calculate it from the frozen collection value
network before the buffer is sealed. Memory use is bounded by the authored
horizon and context; estimate required memory before collection. Random
minibatch access reconstructs complete causal windows and preserves their
within-window time order. The buffer is immutable after target calculation and
cannot be reused for a later policy generation.

Keep optional trajectory archival separate from the training buffer. Extend
the JSON episode/fragment record or introduce a small versioned collection
record so fragments retain preceding action/history, policy version, cutoff
reason, old log-probability/value when requested, and true final successor.
Write completed fragments atomically; preserve readers for existing version-1
records. Archival is never silently fed back into an on-policy update.

**Gate B:** collect across multiple short episodes and across a live episode
cut; verify exact executed-step budget, terminal reward, no lost/duplicated
actions, bounded storage, reproducible sampled actions, and reconstructed
windows equal collection windows before optimization. A buffer from an older
policy version is rejected.

### C. Value model, GAE, and PPO math

Reuse `CategoricalPolicy` and `CategoricalActionDistribution` for the actor.
Instantiate an independent history-only `HistoryCritic` for V; do not create
IQL's Q/target-Q networks. Validate disjoint actor/V trainable parameter IDs
and identical frozen observation codecs. Provide a single decision API that
returns domain action, encoded action, old log-probability, entropy, and V
from the same causal input. Re-scoring that exact action before the first
optimizer step must reproduce old log-probability and V within tolerance.

Compute GAE in FP32 over the sealed buffer. With `b_t` meaning bootstrap is
allowed and `c_t` meaning the next stored decision is in the same episode and
same collection segment:

```text
delta_t = reward_t + gamma * b_t * V_old(h_next) - V_old(h_t)
A_t = delta_t + gamma * lambda * c_t * A_(t+1)
target_t = stop_gradient(A_t + V_old(h_t))
```

Set `(b,c)` to `(0,0)` at true termination, `(1,0)` at a time-limit truncation
with valid final observation, `(1,0)` at a live collection cut, and `(1,1)`
only for an ordinary transition followed by the next decision of this episode
within the buffer. Normalize advantages once over valid decisions for an
iteration when configured, never per minibatch. Keep old log-probabilities,
old values, advantages, and targets fixed for all optimization epochs.

Implement clipped policy loss, independent value loss, and current-policy
categorical entropy. Value clipping around `V_old` is optional and off by
default. Calculate ratio from joint `logp_new - logp_old`; check it and all
losses for non-finite values. Implement optional target-KL early stopping and
gradient clipping. Use independent actor/V optimizers and schedulers with
explicit per-optimizer-step clocks. Correctly weight a smaller last minibatch
and gradient accumulation by valid decision count. Never construct a fake
supervised target or pass PPO loss through supervised `LossService`.

**Gate C:** hand-computed positive/negative advantage clipping, value clipping,
entropy, KL, and GAE match reference values. Gradient ownership is disjoint.
The identity ratio is one before any update, including after a checkpoint
round trip. A one-iteration run has finite losses and actually changes actor
and V weights.

### D. Online phase, execution, and evaluation

Add a discriminated `PPOPhaseConfig` and an online environment source to
`config/training_plan.py`/`config/train_config.py`. Route it through
`load_train_config`, `train.py`, `rl/learner.py`, and a `rl/ppo.py` algorithm
runtime. It must not require `dataset_training`, offline `sources`, `epochs`,
validation loaders, or fabricated `global_training` batch counts. Preserve
legacy supervised and imitation defaults. For this slice require exactly one
PPO phase; mixed-phase scheduling remains a separate parent-plan item.

The typed PPO schema includes `total_env_steps`, `rollout_length`,
`optimization_epochs`, minibatch and accumulation sizes, gamma, GAE lambda,
policy clip epsilon, optional value clip, value/entropy coefficients, optional
target KL, advantage normalization, actor/V optimizer and scheduler plans,
learning rates, gradient limits, evaluation/checkpoint intervals, adapter
factory/config, separate evaluation adapter factory/config, reset/seed policy,
normalization source, and resume mode. Validate ranges, field ownership,
unsupported distributions/devices/distribution modes, budget/remainder rules,
paths, and parameter ownership before a long collection starts. An optional
`initialize_from` actor must match schema, vocab, normalization, model
architecture, and context length exactly. Do not treat this as exact resume.

Run evaluation on a separate local adapter instance with fixed seeds and a
frozen actor snapshot. Default to deterministic mode actions; allow explicitly
named stochastic evaluation. Report mean episode return, spread, lengths,
completed/censored counts, and success only when the adapter defines it.
Select the exported best actor by a declared evaluation metric/direction,
defaulting to mean completed-episode return; never select by PPO loss. If no
episode completes, report that the selection metric is unavailable rather than
treating censored return as complete. The evaluation budget and timeout must be
explicit so a nonterminating task cannot hang training. Expose the same
evaluator to Python callers and to the PPO CLI workflow; this does not yet
require generalized supervised/rollout monitor or
hyperparameter-search integration.

**Gate D:** a clean PPO CLI run can start from a reset-only local environment,
evaluate without changing training RNG/history, choose a best actor by return,
write a report, and load that actor through the existing policy inference and
single-environment rollout APIs. Existing BC, return-conditioned, and IQL
configs still parse and dispatch as before.

### E. Checkpoint, resume, and operational validation

Write an algorithm-owned versioned PPO checkpoint at complete iteration
boundaries. Include actor/V specs and weights; both optimizer/scheduler states;
global environment steps, iteration and policy version; best actor/metric;
codec/statistics and config fingerprints; actor initialization provenance;
training/evaluation seed streams and all torch RNG states; and the live
history/episode counters. Checkpoint writes are atomic. Do not put V or
optimizers in the portable `sequifier_model` actor artifact. Keep reading the
existing supervised format-v2 and IQL checkpoints through their own routes.

Support two declared resume modes. `exact` requires an adapter capability
declaration and successfully restored simulator/RNG state plus history; an
unsupported adapter fails preflight. `reset_environments` restores learner
state but resets the adapter and causal history, drops incomplete-episode
metrics, and marks the continuation as non-trajectory-equivalent. For a crash
inside collection or optimization, restart from the last committed iteration;
do not pretend the partial buffer was resumed. Validate incompatible config,
action codec, model, reward scale, or adapter identity before restore.

Add a structured training report with counts, return/evaluation curves,
policy/value loss, entropy, clip fraction, approximate KL, explained variance,
gradient norms, ratio tails, invalid/censored episode counts, and separate
collection/update/evaluation timing. Add a minimal deterministic categorical
control adapter and PPO config to `documentation/example_inputs`, a `sequifier
make` template, Python API example, policy guide, config docs, consolidated
docs, and release notes. Document the exact scoped limitations and the two
resume semantics.

**Gate E / done for this slice:** the fixed-seed control task improves over
its random policy over multiple seeds; a short deterministic CPU job stays in
CI. A clean train → evaluate → checkpoint → reload → infer → rollout workflow
passes. Interrupted versus uninterrupted exact-snapshot runs match within
device-appropriate tolerance; non-snapshot adapters reject exact resume and
continue in reset mode. Run the existing repository and external test suite
where available, and test the old supervised/BC/return-conditioned/IQL paths.
Record seeds, task rules, hardware, context length, model size, throughput,
and peak memory. Do not mark the broad parent plan complete from this slice.

## 4. Implementation order and review boundaries

| Package | Depends on | Reviewable result |
|---|---|---|
| A. Online contracts and causal state | Existing policy/rollout path | Reset/step/boundary fixtures, shared history state, frozen codec |
| B. Collector and buffer | A | Persistent single-slot collection, exact windows, versioned fragments |
| C. Value, GAE, PPO update | B | Independent V, numerical formulas, identity ratio, optimizer transactions |
| D. Config and evaluation | A–C | One online PPO phase, CLI/Python parity, return-based best actor |
| E. Checkpoints and validation | A–D | Resume modes, artifact/export, learning and regression evidence |

The parent plan's work packages B/D/E/F/I/J remain broader than these scoped
packages. In particular, passing the gates above adds a usable local
categorical PPO path; it does not establish vector/remote/multimodal or
distributed PPO acceptance.
