# DPO, GRPO, and multi-phase policy training implementation plan

Status: proposed, 2026-11-09. This plan builds on the policy-learning changes
since `2d01d407cd2134c8197b1a18f0034d01486f18a9`, the
[IQL plan](offline-iql-implementation-plan.md), and the
[PPO plan](scoped-ppo-implementation-plan.md). It is an implementation plan,
not a claim that these features exist. All policy-learning changes since that
commit are unreleased and may be redesigned or removed. There is no requirement
to migrate their configs, APIs, artifacts, checkpoints, or adapter protocols.
The verification work described below is for a subsequent implementation; no
tests are to be written or run as part of producing this plan.

## 1. Outcome and scope

Support ordered, same-run combinations of `policy_imitation`, `iql`, `ppo`,
`dpo`, and `grpo` phases through the training CLI and Python API. Each phase
has an explicit budget, data/environment contract, evaluation, and selection
rule. A phase produces a selected actor and hands it to the next phase; the
final selected actor is a portable Sequifier model usable for policy inference
and rollout through the redesigned interfaces.

The first release targets this policy boundary: one fixed
categorical action, scalar/categorical observation and static fields, causal
fixed-length history, local single-process execution, and one selected policy
interface. DPO compares alternatives from the same initial history; GRPO samples
multiple trajectories from the same initial condition. This is a control and
sequence-policy implementation, not a claim of LLM-tokenizer, text-prompt,
continuous-action, vectorized, distributed, or arbitrary reward-model support.
The design includes imitation, return-conditioned imitation, IQL, and PPO.
Return-conditioned policies are available for imitation; IQL, PPO, DPO, and GRPO initially require an
unconditioned actor, with an explicit preflight error for incompatible plans.

A multi-phase plan is sequential, not a mixture of concurrently trained
algorithms. Supervised forecasting phases are not part of this policy plan:
reject mixing them with policy phases rather than implying an unspecified
conversion. Cross-run composition uses the same portable actor artifact contract.

## 2. Product and execution contracts

### 2.1 Authoring and dispatch

Add a canonical `PolicyRunConfig` with run-wide model, policy interface,
context length, device, seed, output paths, source catalog, and a discriminated
list of phase configurations. Offline phases reference source names; online
phases declare an environment, evaluation, and frozen normalization. Specify
budgets in their natural units: epochs, offline updates, PPO environment steps,
or GRPO groups. Validate unique phase names and all references before training.
Do not overload supervised `ResolvedTrainingPhase`, which assumes epoch-based
execution and supports only supervised/imitation algorithms.

Use `PolicyRunConfig` as the sole policy-training configuration, with a single
result type containing ordered phase results and the final actor. Route both CLI
and Python entry points through the same validation and coordinator. Delete or
rewrite the unreleased `PolicyTrainConfig`, `PPOTrainConfig`, their separate
dispatch paths, and redundant public entry points as needed. Design one coherent
CLI and `rl_api` around the new contracts; no old policy syntax, default, result
shape, output path, or import needs an adapter. Keep the pre-existing supervised
training path independent, and reject mixed supervised/policy plans.

Replace duplicated closed-set branching with a **built-in explicit registry**
that maps algorithm identifiers to phase config type and runtime factory.
Pydantic's static discriminated union still explicitly lists built-in phase
classes; the registry is not a promise of third-party dynamic config plugins.
One registration inventory drives dispatch and a consistency check verifies
that it agrees with the union. Algorithm-specific configuration and validation
remain in their own phase models; avoid a generic dictionary of arbitrary
options or an algorithm-independent `update()` abstraction that conceals
required inputs. Update typing, `rl_api`, docs and CLI examples together.

### 2.2 Frozen actor representation and phase handoff

Resolve one immutable actor representation before the first phase: architecture,
interface, context length, feature ordering/types, categorical vocabularies,
action IDs and ordering, normalization and units, and return conditioning.
The first offline phase may fit codecs and statistics on its training partition
only; persist the resulting complete representation before any update. An
online-first run must declare a frozen representation or initialize from a
compatible artifact. A later phase must reuse the frozen representation: never
refit from its own data or silently substitute identity statistics. Preflight
all explicitly declared phase representations and artifact contracts against it;
validate data-dependent details when the first phase has fitted them and before
starting later phases. Reject incompatible architecture, codec, normalization,
units, context, or conditioning with an actionable field-level diagnostic.

Each phase receives a deep-copied **selected** actor from the previous phase,
not a mutable handle to the previous trainer. The first phase starts from the
run's `initialize_from` actor if supplied, otherwise from a seeded fresh actor.
Each phase declares `handoff: best | last` (default `best`), its supported
monitor and direction, and a deterministic tie rule: retain the earliest best.
`best` requires a finite metric and an evaluation schedule capable of producing
one; fail rather than silently use `last`. `last` means the most recently
completed optimizer transaction. Save both selected provenance and metric.
The final export is the final phase's selected actor, not the globally best
actor across incomparable metrics.

Optimizers, schedulers, samplers, critics and mutable environment state are
phase-local and start fresh at a phase boundary. Do not transfer optimizer state
or reuse old on-policy samples between algorithms. Warm-starting critics or
optimizer state is outside this release. Use explicit phase indices and names in
paths and reports to prevent collisions. Protect all inputs, including
preference manifests and reference artifacts, from output overwrites.

### 2.3 Reference policies

DPO and, in this plan, GRPO require a frozen reference actor. Each declares
`reference: {source: phase_start}` or `{source: artifact, path: ...}`;
`phase_start` is the default. A reference is a deep copy of the incoming actor
*before* any update, in evaluation mode, with gradients disabled. For a fresh
first phase it is the initialized actor. Validate the full representation and
architecture against the trainable actor and fingerprint the actual weights,
not merely the path. Never silently retarget a reference at the end of an
epoch or phase. Reference weights or an immutable content-addressed artifact
and its verified hash must survive resume even if the original file changes or
disappears; use embedded checkpoint weights in the first implementation.

DPO's reference enters the chosen/rejected log-ratio. GRPO's reference is used
only for its explicitly configured KL penalty; it is distinct from the frozen
**old collection policy** whose scores define the importance ratio. GRPO still
records a reference identity when its KL coefficient is zero so its run
contract is unambiguous.

## 3. DPO data and objective

### 3.1 Preference contract

Add a versioned, inspectable preference-pair manifest and an offline dataset
view in `trajectory/` and `io/`. A pair has a unique ID, a declared common
initial history/context identity, and chosen and rejected alternatives, each
referencing an episode or a contiguous decision span in a supported
sequence source. Define span bounds as zero-based start-inclusive/end-exclusive;
require at least one scored action per side. Alternative histories must have
identical canonical initial observations, static inputs, valid masks and prior
action history at the branch point, under the frozen codec. After branching,
each alternative uses its own actual observations and prior actions. The pair
record need not invent rewards, successor states, or terminal flags for DPO;
design sequence sources to support reward-free preference data as well as
reward-bearing trajectories.
Never fabricate a reward-bearing `Trajectory` merely to satisfy its validator.
Reject missing actions, unsupported labels, ambiguous alignment, non-finite
features, duplicate pair IDs, overlapping chosen/rejected identities that make a
preference meaningless, and incompatible initial contexts. Permit unequal
alternative lengths; record exact decision masks and length diagnostics.

Split before fitting representation. Form connected components across shared
context/group IDs, episode IDs, and duplicate content, then put each component
entirely in train or validation. Explicit validation sources must satisfy the
same no-overlap checks. If too few independent components remain, fail with a
clear error. Avoid materializing all windows: reuse `PolicyWindowView` and
bounded episode caches; persist source identities, span definitions, content
fingerprints, split membership and codec identity for exact resume. Comparison
semantics are supplied by the data: do not infer preference from total reward.

### 3.2 Sequence likelihood and optimization

Add a differentiable sequence-scoring helper over `CategoricalPolicy`: score
the recorded executable action at each selected decision using the causal
window valid at that decision, then sum FP32 log probabilities per alternative.
Padding and unselected context positions contribute zero. Ensure start/special
tokens are never executable labels. Chunk windows to bound memory while
preserving gradients for trainable scores. Reference scoring uses no gradients,
identical input transformations, and evaluation mode. Stochastic network layers
remain disabled during both reference and trainable scoring for reproducible
likelihood ratios; gradients are still enabled on the trainable actor.

For each pair, with `s_theta(y)` the summed action log likelihood and `s_ref`
the frozen reference score, implement canonical sigmoid DPO:

```text
margin = (s_theta(chosen) - s_theta(rejected))
       - (s_ref(chosen)   - s_ref(rejected))
loss   = -logsigmoid(beta * margin)
```

Require finite `beta > 0`; average losses over pairs. Do not silently switch to
per-token averaging, add a supervised objective, or treat a trajectory reward
as a preference. A pair contributes equally even when alternatives have unequal
lengths; report lengths, margins and preference accuracy to expose that choice.
Support a phase-local optimizer/scheduler, gradient accumulation weighted by
actual pair count, clipping, deterministic sampler, checkpoint interval, and
held-out DPO loss as the default minimizing selection metric. Define evaluation
frequency and save/checkpoint boundaries in optimizer updates. Reference scores
may be cached only against content, span, codec and reference weight hashes;
a cache miss or mismatch must recompute, never silently reuse stale scores.

## 4. GRPO collection and objective

### 4.1 Reproducible group starts

Define an online environment contract with an explicit group-start capability. It must produce an opaque, serializable start token and initialize
independent episodes from that token, yielding the same initial observation,
static inputs and units. Specify whether the token captures simulator randomness:
independent rollout action/environment RNG streams must be derived from run
seed, group index and sample index, while the initial condition remains the
same. An adapter may implement this by restoring a captured initial snapshot or
by a deterministic scenario/reset factory, but must explicitly declare the
capability. A repeated seed alone is insufficient. Validate initial histories
and units across a group and reject unsupported adapters before collection.
Use one coherent adapter protocol for online algorithms, with group-start as an
explicit capability required by GRPO. Redesign the current unreleased adapter
protocol and bridge as needed; do not infer capabilities from method names alone.

One group contains `group_size >= 2` independent trajectories from the same
start token, generated sequentially in the first implementation under one
frozen old actor version. Use distinct action RNG streams. Each episode ends at
a real terminal or truncation, or at an explicit per-sample `max_steps` cutoff;
the scalar score is the finite sum of observed rewards up to that boundary,
with a declared fixed reward scale. A cutoff is an explicitly accepted
finite-horizon score, not an invented terminal or a bootstrap. If an adapter
fails, a score is non-finite, or a requested complete group cannot fit within
the configured memory/step budget, abort the whole uncommitted group and never
update from a partial group. Close adapters in `finally`. `total_groups` is the
hard authored budget; optional `max_env_steps` must be checked before starting
a group against the conservative `group_size * max_steps` bound. Report actual
executed actions as well as groups, including discarded actions if a failure is
reported. No automatic retry that silently changes data.

Store group/start identity, episode and step IDs, action histories/masks,
rewards, cutoff reason, fixed old per-action log probabilities, reference
identity, score, and actor generation in a bounded sealed group buffer. Retain
only enough causal frames to reconstruct each action history. Verify old scores
against the current actor before the first update; enforce one generation and
consume each sealed buffer only once. Keep archival optional and distinct from
training input, as with PPO. Groups are collected under `no_grad` and no actor
updates happen until all groups for an iteration have been sealed.

### 4.2 Chosen GRPO variant

Specify this exact first variant rather than treating all GRPO variants as
interchangeable. Collect `groups_per_update` complete groups. For group scores
`R_i`, compute FP32 population mean and standard deviation, and assign the same
advantage to every valid decision of trajectory `i`:

```text
A_i = (R_i - mean(R)) / (std_population(R) + advantage_epsilon)
r_it = exp(log pi_theta(a_it|h_it) - log pi_old(a_it|h_it))
K_it = exp(log pi_ref(a_it|h_it) - log pi_theta(a_it|h_it))
       - (log pi_ref(a_it|h_it) - log pi_theta(a_it|h_it)) - 1
loss = mean_over_trajectories(mean_over_valid_decisions(
       -min(r_it * A_i, clamp(r_it, 1-epsilon, 1+epsilon) * A_i)
       + kl_coefficient * K_it))
```

This sampled reverse-KL estimator is nonnegative per decision, but its
expectation is a KL only under the corresponding sampling distribution; after
multiple optimization epochs the samples are from `pi_old`, so report it as a
sampled reference penalty, not an exact current-policy KL. Scores from the old
actor and fixed reference remain detached for every optimization epoch.
`advantage_epsilon > 0` is explicit; identical group scores yield zero
advantages, though a nonzero reference penalty can still update the actor. This
variant weights each trajectory equally irrespective of length and each group
equally because group sizes are fixed. Minibatches contain whole trajectories;
gradient accumulation weights by actual trajectory count, including the final
partial minibatch. Compute ratio, penalty and loss in FP32 with explicit finite
checks and fail before applying an invalid update. `kl_coefficient >= 0` and
`0 < epsilon < 1` are phase fields. Do not reuse PPO's old-policy approximate
KL as the reference penalty; an optional old-policy KL early-stop is separately
named and documented. GRPO has no value model, GAE, or value optimizer.

Evaluation uses a separate adapter and independent deterministic evaluation
seeds/scenarios, never the training group RNG. Define score mean as the default
maximizing selection metric, evaluation interval in completed updates, and an
explicit final evaluation so `handoff: best` is meaningful. Report score
spread, zero-variance groups, sample lengths/cutoffs, KL penalty, clip fraction,
policy version, optimizer steps and real environment steps.

## 5. Runtime, checkpoints, and artifacts

Introduce a policy run coordinator responsible for validating the plan,
preparing the immutable actor representation, invoking a phase runtime,
committing phase results, and advancing to the next phase. Phase runtimes own
their natural lifecycle and algorithm-specific state. A narrow runtime
contract should accept a phase context and return a selected actor, typed phase
result, provenance and resumable state; it need not impose identical update
loops on imitation, IQL, PPO, DPO and GRPO. Factor shared actor loading and
compatibility, reference freezing, optimizer/scheduler construction, RNG
capture, safe atomic writes and report conventions only where their semantics
actually match. Keep PPO's collection-policy version separate from generic
phase/run IDs. Refactor imitation, IQL and PPO directly into the target runtime; simplify or
replace their current interfaces and implementation where this improves the
architecture. Specify their objectives, selection rules and defaults as part of
the new contracts rather than preserving incidental current behavior.

Create a versioned multi-phase run checkpoint envelope containing the resolved
plan and immutable fingerprint, actor representation, run RNG state, current
phase index and status, completed phase results and selected actors, current
phase's algorithm identifier/version and opaque typed state, phase-local RNGs,
and the last committed transaction. Each algorithm owns validation of its
state: DPO includes actor/reference, optimizer/scheduler, sampler, dataset/split
identity, clocks and best selection; GRPO includes actor/reference, optimizer,
scheduler, group/seed counters, evaluation state and clocks. Write to a
temporary file and atomically replace only after a completed optimizer update
or phase handoff. No partial group or partially applied update is a checkpoint
boundary. On resume, check plan, data, model, reference hashes and capabilities
before mutating outputs. Resume at the current phase without rerunning completed
phases or reselecting their actor; a committed handoff must be idempotent.

For GRPO, checkpoints are taken only after updates and with no live group.
Subsequent groups are derived from persisted group counters/start tokens and RNG
streams. Exact continuation requires the adapter's documented group-start
reproducibility and compatible implementation identity; validate both. If the
adapter cannot guarantee it, reject exact resume rather than claim equivalence.
An explicitly selected non-exact restart may begin new groups with new derived
seeds, report the discontinuity and preserve completed updates, but may never
reuse a partial group. PPO must likewise declare whether resuming restores
the environment exactly or resets it; exact continuation requires a validated
adapter snapshot, and a reset must be reported as a discontinuity.

Use the new checkpoint envelope as the only policy-training checkpoint format.
Remove the unreleased algorithm-specific readers and migration paths where they
are no longer needed; do not add legacy format detection or conversion.
Portable inference artifacts remain actor-only. The run report records phase
order/config fingerprints, source/reference provenance, chosen metric and actor,
phase clocks, continuation mode and final artifact path. Checkpoint payloads can
contain training-only networks and references; inference loading must not depend
on them.

## 6. Dependency-ordered delivery

1. **Target contracts:** define the policy run/phase schemas, algorithm
   registry, adapter capabilities, actor representation, phase-scoped paths and
   structured results. Remove redundant policy configuration and dispatch.
   Reject mixed supervised/policy plans and incompatible representations early.
2. **Coordinator and handoff:** implement the phase runtime contract, immutable
   phase context, selected actor handoff and checkpoint envelope. Restructure
   imitation, IQL and PPO to fit directly into the target architecture; define
   their objectives and selection behavior explicitly. Demonstrate imitation →
   IQL and imitation → PPO.
3. **DPO inputs and scoring:** implement pair manifest, source adapters,
   component-safe splits, causal sequence log scoring and frozen reference
   loading. Validate representation and prevent data leakage before training.
4. **DPO runtime:** add loss, optimizer, metrics, evaluation and resume; wire
   the phase through CLI/Python and export. Exercise imitation → DPO and DPO →
   another compatible algorithm as well as single-phase DPO.
5. **GRPO environment and buffers:** implement the shared online environment
   contract, explicit group-start capability, reproducible independent rollout
   streams, bounded complete-group collection and old/reference scoring.
6. **GRPO runtime:** add the specified group-normalized objective, actor-only
   optimizer, separate evaluation and checkpoint/resume. Exercise imitation →
   GRPO, DPO → GRPO, and a longer mixed plan with phase handoffs.
7. **Product integration:** document the new schemas, limitations, checkpoint
   behavior, monitor meanings and sample configs. Update CLI help, public Python
   exports and release notes; ensure final actors work with the redesigned
   inference and rollout interfaces. Remove superseded unreleased policy code,
   examples and documentation instead of maintaining parallel interfaces.

## 7. Acceptance and verification for implementation

The implementation is complete only when all items below have evidence. These
are future verification requirements, not tests executed for this plan. Use the
repository's actual test ownership and commands discovered at implementation
time; do not assume an external test checkout exists.

- **Algorithm coverage:** single-phase BC, return-conditioned imitation, IQL,
  PPO, DPO and GRPO work under the new contracts. Verify their stated objectives
  and selection behavior; unsupported combinations fail before training. The
  supervised training path predating this policy work continues to function.
- **DPO data and math:** hand-computed pair scores, causal masking, unequal
  lengths, objective and gradient direction match the equations; actor equal to
  reference gives zero margin and loss `log(2)`. Reference parameters never
  receive gradients. Reject invalid pairs and train/validation leakage.
- **GRPO data and math:** identical group starts and independent reproducible
  streams are established; analytic group normalization, clipping for both
  advantage signs, reference penalty, per-trajectory weighting and zero-spread
  groups match the declared variant. No critic is allocated, stale policy
  versions and incomplete groups are rejected, and buffers obey memory limits.
- **Lifecycle:** every supported phase transition transfers exactly the chosen
  actor and frozen codec; optimizers, samplers, reference policies and critics
  have the stated ownership. Best/last behavior, evaluation failures, source
  overlaps and output path collisions are explicit. A deterministic toy control
  task demonstrates usable DPO and GRPO learning without making a universal
  learning-quality guarantee.
- **Resume:** interruption at each committed update and phase boundary produces
  equivalent subsequent state under the supported exact-resume contract;
  changed data, reference weights, codec, config or adapter identity is
  rejected. Non-exact environment reset is labeled, and partial groups are
  never consumed. Only the new policy checkpoint format is required.
- **End-to-end:** run representative single- and multi-phase YAML configs via
  CLI and Python, check phase reports and counters, then load the final ordinary
  actor for inference and rollout through the redesigned interfaces. Verify deterministic reproducibility
  on a supported local device and document any backend-specific limits.

## 8. Deliberately deferred

LLM/prompt-completion support, continuous or multi-component action spaces,
remote/vectorized environments, distributed/mixed-precision execution,
learned reward models, automatic preference generation, online preference
collection, asynchronous groups, dynamic external algorithm plugins,
supervised-to-policy phases, cross-phase optimizer/critic inheritance, and
mid-group exact checkpointing require their own contracts. Their omission must
not be represented as general support for those workflows.
