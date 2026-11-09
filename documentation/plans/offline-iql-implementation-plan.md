# Offline IQL implementation plan for the current policy-learning slice

Status: implementation added, verification deferred at user request, 2026-11-09.
This is a scoped follow-up to
[the rollout and reinforcement-learning plan](rollout-and-reinforcement-learning-implementation-plan.md),
especially sections 3, 5.1, 5.3–5.5, and 8. The target is an **offline IQL path
as complete as the current scalar, single-categorical-action BC and
return-conditioned imitation path**: train from the existing supported sequence
sources, select and save an actor, resume training, load it for inference, and
execute it with the existing single-environment coordinator. The original plan's
continuous/joint actions, media, vector/remote environments, DDP, PPO, and
mixed-algorithm runs remain later work.

## 1. Baseline in the diff since `2d01d407cd2134c8197b1a18f0034d01486f18a9`

The diff provides the dependencies for this slice, but no IQL learner:

- `sequifier train` accepts one `policy_imitation` phase. It trains categorical
  BC or return-conditioned imitation using the ordinary model, optimization,
  evaluation, and format-v2 run-checkpoint services. The best actor is a normal
  portable model artifact. `sequifier infer` can make raw table decisions and
  execute a policy through the shared rollout coordinator.
- CSV/Parquet, prepared scalar histories, and version-1 JSON rollout logs adapt
  to `SequenceEpisode`. Optional `TransitionColumns` already represent rewards,
  boundary flags, successors, and bootstrap flags. Episode splitting, training-only
  fitting, bounded episode caching, and causal windows exist.
- Rollout collection has a persistent **single** local adapter, explicit input
  bindings, terminal action execution, scalar reward, final successor, and atomic
  one-episode JSON output. It is not a general vector simulator or an evaluation
  service.
- `CategoricalPolicy` and `CategoricalActionDistribution` support one categorical
  action. There is no V, Q, target-Q, transition sampler, IQL optimizer, or
  multi-network checkpoint. Policy training's config and command dispatch reject
  IQL and more than one phase.

No repository test suite is present in this checkout; the release notes say the
tests moved to `../sequifier-test/tests/sequifier-refactor-unit`. Verification
below must discover and run that suite where available, and add focused tests in
the owning test repository or an agreed in-repository location.

## 2. Scope and contract decisions

1. **Action and model scope.** One fixed categorical action vocabulary, existing
   scalar/categorical observation and static fields, finite context, one actor
   interface, and independent V/Q1/Q2 modules. The actor is unconditioned
   (`return_conditioned: false`). A return-conditioned actor is rejected by IQL
   until an explicit conditioning or distillation policy is implemented. The
   ordinary BC actor can warm-start IQL after exact schema, vocabulary,
   normalization, architecture, and context-length checks; critics start fresh.
2. **Data scope.** A fixed offline training partition and separate held-out
   partition from the current `SequenceSourceConfig` formats. IQL requires a
   reward, action, known termination/truncation and bootstrap decision at every
   sampled step, plus a genuine successor whenever bootstrap is allowed. BC's
   ability to use reward-free data stays intact. Collection cutoffs and fragments
   are eligible when their history and successor are known; unknown boundaries
   are rejected for IQL. No new online data is added during the phase.
3. **Runtime scope.** One `iql` phase via `sequifier train`, with Python entry
   point parity, single process on the currently supported policy devices,
   actor-only inference/rollouts, and exact offline resume from complete update
   boundaries. No same-run BC→IQL phase scheduling is required for this slice:
   BC artifact warm-start is the explicit bridge. Do not imply the full
   multi-phase or distributed claims of the parent plan are complete.
4. **Reward and evaluation scope.** Existing sources contain one scalar reward.
   Preserve it in raw units; expose an optional fixed, checkpointed scale only
   if needed. Select checkpoints using a declared held-out metric. Include
   held-out losses/diagnostics and a repeatable local rollout evaluation on a
   small control fixture, so learning quality is measured independently of
   Bellman loss. A general training-time rollout monitor is not part of this
   increment.

## 3. Dependency-ordered work packages

### A. Lock transition and successor-history semantics

Extend `io/sequence.py` and `trajectory/importers.py` with an IQL eligibility
validator. Require transition columns to be complete for selected IQL steps;
reject absent rewards or boundary/bootstrap flags, non-finite reward, ambiguous
action timing, bootstrap after termination, bootstrap with a missing final
successor, and a reset observation passed as a successor. Preserve the current
ability to record both `terminated` and `truncated`, with termination taking
precedence. Make validation errors include source, episode ID, and step.

Add an IQL transition view in `io/` that indexes decisions across episodes without
materializing all context windows. It returns `h_t`, recorded `a_t`, scalar
`r_t`, `h_next`, bootstrap mask, and valid/padding masks. Reuse
`PolicyWindowView` for `h_t`; add a successor-window method that appends the
**actual** successor observation and uses `a_t` as its previous action. For an
interior step, the explicit successor must equal the next row when both are
present. For a terminal step without a successor, return a masked placeholder
and never evaluate V on it. Do not cross episode or fragment boundaries by
looking up the next dataset item. Static context and trained codecs are shared
between both windows. Fit vocabularies and normalization on training episodes
only, then freeze them for validation and inference.

Sampling is uniform over eligible transitions, with a persisted RNG state and
dataset identity (content hash, episode/split IDs, codec hash). Keep the existing
bounded episode cache and construct only the two context-sized windows per
sample. Continue to guard against duplicate paths, IDs, content, and collection
groups across train/validation partitions.

**Gate A:** hand-authored episodes prove first/interior/final alignment, truncation
bootstrap from the final state, terminal no-bootstrap, fragment context, and
rejection of missing/contradictory capabilities. Changing `a_t`, reward, or future
observation cannot change actor/V inputs at time `t`.

### B. Add actor, V, and twin-Q model primitives

In `rl/models.py`, compose V, Q1, and Q2 from the existing ingestion/backbone
building blocks. V accepts only history. Each Q encodes history, then fuses the
**recorded current action** through a separate categorical action encoder before
its scalar head. Q1 and Q2 have independent learned parameters. Actor, V, Q1,
and Q2 share immutable codecs/specifications but no trainable parameters or
mutable normalization buffers. Make parameter ownership and optimizer assignment
explicit and validate disjointness before training.

Expose a small policy runtime boundary in `rl/policy.py`/
`rl/distributions.py`: deterministic/stochastic `act` and
`evaluate_actions` return the same action IDs/domain labels and joint log
probability for the recorded categorical action. Keep special/start IDs excluded
from actions. This is a one-component instance of the parent plan's contract;
continuous, ordered, and joint distributions are deferred. Keep probability
math in FP32 and reject invalid action labels.

**Gate B:** shape, action-codec, finite-output, parameter-disjointness, and
gradient-ownership tests. Actor/V are invariant to changing current action
labels and future observations; Q responds to its explicit action input.

### C. Add the IQL phase and optimizer transactions

Add a discriminated `IQLPhaseConfig` to `config/training_plan.py` and route it
through `config/train_config.py`, `train.py`, and `rl/learner.py`. Keep the
existing supervised and imitation defaults. Require one IQL phase for this
slice, named training/validation sources, actor/V/Q specs, `updates`, `batch_size`,
`gamma`, `expectile`, `inverse_temperature`, `max_weight`, target-Q Polyak rate,
separate optimizer/scheduler plans, clipping limits, evaluation/checkpoint
intervals, and optional BC artifact initialization. Validate all ranges,
references, action/schema compatibility, unsupported distribution modes and
distributed/FSDP settings **before** a long run starts. Do not invent epochs or
supervised targets for an IQL update budget.

Add `rl/algorithms/iql.py` behind an algorithm runtime that owns preparation,
one update, evaluation, checkpoint state, and named counters. Reuse the existing
optimizer transaction and random-state services where their semantics fit;
extend them to assign independent V, Q, and actor optimizers and scheduler
clocks. Controller/observer events must identify algorithm, network role,
algorithm update, and optimizer step so one-optimizer integrations cannot
silently act on a different network. The supervised loss path stays untouched.

For each sampled batch, take a consistent pre-update snapshot of detached
targets/advantages:

```text
qbar = min(Q1_target(h_t,a_t), Q2_target(h_t,a_t))
delta = stop_gradient(qbar) - V(h_t)
L_V = mean(|expectile - 1[delta < 0]| * delta²)
y = r_t + gamma * bootstrap_t * stop_gradient(V(h_next))
L_Q = mean((Q1(h_t,a_t)-y)² + (Q2(h_t,a_t)-y)²)
w = min(exp(inverse_temperature * stop_gradient(qbar - V(h_t))), max_weight)
L_actor = -mean(stop_gradient(w) * log pi(a_t|h_t))
```

Cap the exponent input at `log(max_weight)` for stability. Apply V, Q, and actor
optimizer transactions to those targets without gradient paths across networks.
Update target Qs by Polyak averaging **only after a successful Q step**. Define
one IQL update as all three attempted transactions and record applied/skipped
counters separately; handle AMP skips without advancing the wrong scheduler or
target. Stop on non-finite targets/losses instead of silently clipping them.

**Gate C:** hand-computed expectile asymmetry, Bellman bootstrap, twin minimum,
actor weights/cap, and target updates match reference numbers. Only the intended
network receives gradients in each transaction. A one-batch deterministic run
produces finite metrics and saves an actor that `load_policy` can execute.

### D. Bundle IQL state without changing portable actor artifacts

Add an algorithm-owned checkpoint bundle in `rl/checkpoint.py` or a dedicated
module. It contains actor/V/Q1/Q2 and target-Q weights, three
optimizer/scheduler/scaler states, update and applied-step counters, sampler and
all device RNG states, frozen normalization/codecs, source/split/content hashes,
phase config, and artifact-initialization provenance. Save atomically after a
complete update. Restore strictly; missing critics or mismatched data/specs
must fail exact resume. Retain reading of existing supervised/imitation
format-v2 checkpoints, and do not reinterpret one as an IQL resume bundle.

Export **only** the actor through the current `sequifier_model` path. Record IQL
provenance without embedding critics or optimizers. Inference, `load_policy`,
and the single-environment coordinator should execute it with the existing
codec/action mapping. Artifact warm-start and exact IQL resume remain distinct.

**Gate D:** interrupted versus uninterrupted fixed-seed training matches
sample order, counters, and weights within device-appropriate tolerances;
corrupt/mismatched bundles fail; existing format-v2 resume and model loading
still work; actor outputs match before and after export.

### E. Evaluate, document, and release the slice

Write a structured IQL report with source/split IDs, exclusions, update counts,
actor NLL, V/Q ranges, Bellman residual, advantage and weight quantiles,
weight-cap fraction, best-checkpoint criterion, and artifact paths. Held-out
diagnostics must not fit codecs or update networks. Provide a deterministic
small local control environment and mixed-quality offline behavior dataset.
Compare IQL against BC on fixed seeds using rollout return and action quality;
report mean, spread, dataset-generation rules, and tolerances. The local
rollout evaluation can initially be a separate explicit evaluation command/job
that consumes each saved actor and the same adapter protocol; it need not be a
generalized training monitor.

Add one complete `train-iql` YAML example, Python API example, and documentation
for required transition columns, cutoff/truncation handling, actor warm-start,
resume, inference, unsupported combinations, and result interpretation. Update
`documentation/policy-learning.md`, config docs, `sequifier make` templates,
consolidated docs, and release notes. Run lint/type checks and the existing
suite, plus the focused gates above and a clean CLI train → checkpoint → reload
→ infer → rollout smoke run.

**Gate E / done:** a user can import or point to an existing supported scalar
trajectory source, train a fixed-dataset categorical IQL actor with a named
phase, resume it exactly, inspect learning and held-out behavior, load the
actor as an ordinary model artifact, and execute it through the current local
coordinator. BC and return-conditioned workflows still pass their existing
checks. This completes **this IQL slice**, not all Stage 1/2 acceptance criteria
of the parent plan.
