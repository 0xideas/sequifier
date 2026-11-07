# Rollouts, return-conditioned policies, IQL, and PPO

Status: implementation specification. Date: 2026-11-07.

First implementation slice: opt-in CPU numeric forecast collection with explicit
bindings, a persistent local single-environment adapter, shared prediction/step
ordering, seed history and fixed preprocessing, and validated JSON trajectories.
See [local forecast rollouts](../rollout.md) for configuration and public APIs.
The broader stages below remain planned. No tests or acceptance fixtures were
written or run for this slice, as requested.

Second implementation slice: episode-split trajectory datasets, causal
current-observation/previous-action windows, discounted return labels, a single
categorical actor, return-conditioned supervised training and BC, local actor
artifacts, and execution through the same coordinator. The tiny binary-control
workflow records achieved returns after reloading both actors. See
[policy learning](../policy-learning.md). This slice has not been executed or
acceptance-tested; no tests were written or run, as requested. IQL and the
remaining broader stages are still planned; RC is an independent route.

Deliver this as one coordinated update, built and verified in the three dependency-ordered stages below. Completion means all three stages work together; stage boundaries are engineering checkpoints, not separate proposed releases. This document specifies new behavior and proposed API names, not functionality already present.

## 1. Outcome and scope

The update must enable:

1. Autoregressive inference when some next-step inputs come from predictions and others come from external dependent-variable calculations or simulation. Support scalar features, categorical features, image views, static conditioning, and depth inputs through the existing execution schema.
2. A persistent, batched environment interface shared by inference, training-time rollout evaluation, trajectory collection, and PPO. A separate optional reward provider can supply rewards from external software.
3. Versioned trajectories that can be collected online, imported from existing data, evaluated, reward-labeled, and consumed by offline learners without temporal leakage.
4. Return-conditioned sequence modeling (RCSM), complete offline Implicit Q-Learning (IQL), and full synchronous clipped PPO with generalized advantage estimation (GAE).
5. One training plan that can run supervised/return-conditioned pretraining, offline IQL, and online PPO phases, with explicit state transfer and optimizer lifecycle rules.
6. PyTorch learning and PT/ONNX policy deployment, including consistent preprocessing and action distributions. Training algorithms and external environments remain outside the exported model graph.

Supported RL actions: ordinary categorical variables, bounded and unbounded continuous vectors, and joint policies combining these. Include independent categorical heads and explicitly ordered autoregressive categorical heads. All learnable action components have a joint likelihood. Existing ordinary regression heads remain available to supervised modeling and dependent-variable inference; RL phases require an explicit probability distribution.

The release includes single-process CPU/CUDA execution and synchronous DDP learning with separate environment slots per rank. Existing supervised distributed strategies must keep working. New IQL/PPO phases reject FSDP initially with a specific error until multi-network optimizer/checkpoint semantics are implemented for it; do not advertise FSDP RL support through inheritance from supervised training.

Not part of this update: SAC/TD3/AWAC, online IQL fine-tuning, learned world models, differentiable simulators, asynchronous actors, policy-gradient methods for diffusion/flow policies, constrained/multi-objective RL, dynamic action spaces, multi-agent learning, or hard real-time robot scheduling. Hash-decoded categorical outputs remain supported in existing inference but are rejected as RL actions unless they expose a tested normalized joint probability; candidate ranking scores are insufficient.

One RL decision executes one action vector and receives one transition. The adapter may hold that action for a configured fixed number of physics substeps and aggregate its reward. Temporal action chunks with intervening replanning or variable-duration decisions need explicit semi-Markov discounts and are deferred. Ordered action dimensions at one instant are supported and are distinct from temporal chunks.

## 2. Repository integration points

Current constraints established by source inspection:

- `infer.py:get_probs_preds_from_dict` owns a NumPy generation loop, assumes each changing input exists in `preds_for_step`, and rejects future image frames. Extract orchestration rather than adding independent loops to each consumer.
- `config/infer_config.py` separately enforces identical inputs/targets in autoregression and generation-step validators; depth autoregression is rejected elsewhere. Replace these restrictions with source coverage and execution-schema checks for supported rollout paths. Preserve forward-looking, one-step generative constraints where still required.
- `model/network.py` exposes encoding/decoding primitives and `ModelOutput`. Preserve the supervised output contract; add policy/value/Q wrappers rather than changing every scalar output into an RL distribution.
- `io/batch.py` and `training/loss.py` describe supervised batches and masked supervised losses. RL needs dedicated transition/rollout batches; do not overload target masks with termination semantics.
- `training/engine.py`, `training/runtime.py`, `training/optimization.py`, and `train.py` already separate orchestration from model, data, and update services. Extend that separation with algorithm runtimes.
- `integration/specifications.py` already defines persistent `module:attribute` factories and state policies. Extract reusable factory loading/identity validation; environments are not `TrainingObserver` or `TrainingController` instances and must not be forced into those protocols.
- `evaluation/service.py` and its monitor currently describe supervised loss. Extend results/monitors with named metrics rather than placing returns in `total_loss`.
- `artifacts/run_checkpoint.py` currently stores one model and optimization state in format version 2. Introduce an explicit compatible checkpoint migration and an algorithm-owned bundle for additional networks/optimizers.
- `hyperparameter_search.py` launches `evaluation_script` once and reads named JSON metrics. Keep that contract. Share factory conventions with hooks, not its subprocess-per-evaluation execution model.

Proposed new modules:

```text
config/rollout_config.py           Environment/reward factories, bindings, budgets
config/rl_config.py                Policy specs and discriminated algorithm phases
rollout/{contracts,bindings,history,coordinator,returns}.py
rollout/{factories,dependent,rewards,remote}.py
rollout/adapters/{gymnasium,tensor_vector}.py
trajectory/{schema,store,importers,dataset,relabel}.py
rl/{policy,distributions,models,normalization,metrics}.py
rl/{runtime,checkpoint,transfer}.py
rl/algorithms/{return_conditioned,iql,ppo}.py
rl/{ppo_buffer,offline_sampling}.py
rollout_api.py                    Stable external adapter/collector API
rl_api.py                         Stable policy and learner API
```

Names can be adjusted during implementation, but the ownership boundaries and contracts below are requirements.

## 3. Common contracts to settle before implementation

### 3.1 Time and observation/action alignment

Use the following canonical transition everywhere:

```text
h_t = observations through o_t, actions only through a_(t-1), validity and time metadata
a_t ~ policy(h_t)
environment.step(a_t) -> o_(t+1), reward_t, terminated_t, truncated_t
transition = (h_t, a_t, reward_t, o_(t+1), boundary flags)
```

The policy and V must never see `a_t`, its reward, or next observation when computing their step-t outputs. Q may receive `a_t` through a separate action encoder. Introduce an explicit policy window view that puts current observations and previous actions at one decision position and supervises the current action there. Do not reinterpret ordinary `target_offset` silently. For existing forecast models, keep their training alignment and use an explicit forecast binding; do not claim that arbitrary next-item checkpoints can be converted into current-observation policies without retraining.

For robotics, the row contains current images/proprioception and the previous motor command; the output is the next command to execute. For order-book forecasting, a forecast binding appends predicted level-4 features and adapter-computed level-2 features into the same next context position. Both use the coordinator, with different bindings.

### 3.2 Environment and reward protocols

Use typed dataclasses/protocols with a protocol version. Illustrative signatures:

```python
class EnvironmentAdapter(Protocol):
    def describe(self) -> EnvironmentSpec: ...
    def reset(self, request: ResetRequest) -> ObservationBatch: ...
    def step(self, request: StepRequest) -> TransitionBatch: ...
    def close(self) -> None: ...

class RewardProvider(Protocol):
    def score(self, transitions: TransitionBatch) -> RewardBatch: ...
    def finalize_episode(self, episode: EpisodeSummary) -> EpisodeOutcome: ...
    def close(self) -> None: ...
```

`EnvironmentSpec` declares ordered fields, shapes, dtypes, units, bounds, batch/device support, partial-reset/autoreset semantics, seed support, and snapshot capabilities. `StepRequest` includes slot IDs, episode IDs, transition indices, an idempotency/request ID, actions, and relevant context. The coordinator steps only active slots. GPU adapters use fixed slots plus an active mask; adapters unable to step subsets must freeze inactive worlds or keep all slots active during collection. Never silently advance a live trajectory without recording it.

`TransitionBatch` contains true next observations, named rewards if supplied, separate `terminated`/`truncated` masks, optional declared privileged critic observations, elapsed simulated time, and diagnostics. Return final pre-reset observations explicitly. Autoreset adapters normalize their outputs into this contract and separately expose reset observations. A state that was reset is never used as the successor of the preceding episode.

Every hook is instantiated once per rank/run or evaluation session and closed in `finally`. Partial reset is explicit and changes only selected slots. Episode IDs are unique across ranks/restarts. A stateless dependent-variable adapter uses the same protocol, seeded from the initial history, but can simply calculate a deterministic function of predictions without a physics engine.

Environment and reward factory specifications share `factory`, `config`, `id`, and version fields. Use `module:factory` as the CLI/config path and allow supplied instances for local Python runs. Construct instances inside workers for DDP. Adapter configuration specifies connection/launch details; no per-step process startup.

The public value representation is domain values: original categorical labels, physical continuous units, and image arrays/tensors. One shared codec maps to/from model representation using fixed training metadata. Provide a numeric tensor fast path with declared integer vocabularies and scales; use a metadata handshake once to avoid decoding labels through Python on every GPU step. Reject changed units/vocabularies/shapes and non-finite values. Never refit preprocessing on rollout/evaluation data.

Provide optional adapter `state_dict`/`load_state_dict` and explicit capabilities for exact snapshots, including remote simulator RNG and reward state. Exact snapshots are optional; never infer snapshot support from the existence of a Python method alone.

### 3.3 Bindings and field ownership

Each dynamic model-input field, image view, and structured input has exactly one next-step source: `prediction`, `environment`, or `provided` (an explicitly scheduled exogenous input). Static fields remain constant within an episode. Return conditioning is owned by the coordinator. Reject missing coverage, duplicate producers, ambiguous prediction/environment overrides, and exhausted exogenous schedules before the affected transition.

Mappings include prediction-to-action names and codecs, input update names, and structured masks. A model can predict outputs that are only actions and never fed back, or outputs that are appended directly. Environment outputs can include full corrected state when simulator state is authoritative. This choice must be declared per field; do not automatically overwrite predicted values.

Image frame-valid masks, attention masks, depth occupancy masks, and time coordinates advance atomically with their features. Reuse `ExecutionSchema` for shape/dtype checks. Depth updates replace a complete next-position structured value plus its mask. Prefix/preconditioning images remain static unless the policy interface explicitly models them as dynamic image views.

### 3.4 Rewards, outcomes, and discounts

Store raw named reward channels and episode metrics separately. Algorithms optimize one scalar reward selected by explicit fixed channel weights and transforms. Version and persist this scalarization; multiple channels do not imply a multi-objective learner. Avoid double-counting a channel produced by both the simulator and reward provider.

Support sparse terminal rewards: the episode outcome callback runs on termination/truncation before the terminal transition is sealed and may supply an explicitly configured terminal reward adjustment. Once emitted to a learner, a transition is immutable. Scores available only after a whole collection run are evaluation metrics unless an offline relabeling job constructs a new dataset. PPO rejects unresolved delayed reward providers; it cannot update from provisional rewards and revise them later.

Distinguish task termination from time-limit truncation and collection boundaries. Store a `bootstrap_allowed` flag validated against those boundaries: ordinary true termination disables bootstrap; time-limit truncation with a valid final observation permits it. If task semantics make a time limit genuinely terminal, encode that explicitly. Collector horizon cuts are fragments, not artificial episode ends.

If an adapter reports both termination and truncation on the same transition, preserve both facts and give true termination precedence for bootstrap and GAE. Import validation rejects contradictory discount/next-observation claims, not the simultaneous flags by themselves.

Use fixed gamma per decision for this release. Record physics substep count and duration, and reject variable-duration RL decisions rather than applying a misleading fixed discount. Raw rewards remain unchanged in artifacts; any learner reward scaling is a fixed, checkpointed transform.

## 4. Stage 1 — shared rollout and trajectory infrastructure

### 4.1 Coordinator and history storage

Implement a coordinator independent of CLI, model format, and optimizer. It owns slot identities, bindings, codec calls, context windows, masks, environment/reward lifecycle, collection budgets, and sink writes. It calls a one-step predictor interface; PT/ONNX forecasting and RL policy runtimes implement that interface.

Use a per-slot history ring buffer with explicit decision positions and padding. Each collection starts from either validated historical seed windows or adapter reset observations. Default for reset-only starts: left padding with false attention-valid masks, then one real observation and a declared previous-action start token/value. Longer warm-up requires an explicit source/controller and is recorded. Never fabricate valid historical camera frames or prior actions.

Budget modes are explicit: maximum transitions, completed episodes, and per-episode maximum decisions, plus inference `generation_steps`. Prediction output budgets count emitted predictions, even when trajectories end early. Ragged results carry episode/step identity; padded tensors carry valid masks. For fixed-shape GPU collection, collect the configured decisions per slot and reset ended episodes in place.

Terminal predictions must still be executed and rewarded. Refactor the existing generation loop's early break so the final requested action/forecast gets a transition and outcome even though no next model call is needed. In pure legacy prediction-feedback mode use a no-op adapter and avoid externally visible extra work.

Build one coordinator path for eager PT/ONNX and media/tensor inputs. Avoid eager CPU conversion for the torch policy path. Separate output formatting from collection so existing flat prediction/probability output remains unchanged when there are no episode boundaries. Include episode IDs and masks in the new trajectory outputs.

### 4.2 Adapters and service transport

Ship these adapters/examples:

- A dependent-feature adapter for level-4 to level-2 order-book feature updates, with deterministic synthetic data and no trading-system dependency.
- A native torch vector adapter for small analytic control environments, with CPU/CUDA support and partial resets. This is the reference for high-throughput learning.
- An optional Gymnasium single/vector adapter covering discrete, continuous Box, MultiDiscrete, and supported dictionary observations. Pin/test the optional dependency and normalize supported autoreset modes explicitly.
- A deterministic toy robotics adapter with RGB views and motor commands, demonstrating image/mask updates without a heavyweight simulator installation.
- A generic persistent remote adapter and example server using versioned, framed requests/responses, request IDs, deadlines, handshake schemas, and bounded payload sizes. Numeric arrays use a binary representation preserving shape/dtype; avoid JSON pixel lists. Support start/connect/close and demonstrate crash handling. No transparent retry of a non-idempotent step after an ambiguous timeout; require server deduplication or terminate collection with a clear error.

Third-party robotics integrations implement the same adapter protocol. The release claims complete simulator integration plumbing, with tested local/vector/remote examples; it does not claim bundled production integrations for every robotics engine. Keep Gymnasium and media dependencies optional and lazy-loaded. Timeouts are specified per reset/step/reward request; a synchronous local call cannot be forcibly interrupted in-process, so adapters requiring hard deadlines must use process/service isolation.

### 4.3 Trajectory artifact version 1

Define a manifest plus atomic, sharded records. Use Arrow/Parquet for tabular columns and indexed binary/tensor shards for images and depth payloads; do not require all media in RAM. Include a final committed manifest, shard checksums, counts, codec/schema hashes, and immutable source provenance. Incomplete shards are recoverable but never silently treated as a finished dataset.

Required logical fields:

| Scope | Fields |
|---|---|
| Dataset | Format version, field/units/action specs, preprocessing/codec hash, reward definition and scalarization, task ID, split IDs, simulator/adapter versions, collection policy artifact ID |
| Episode | Globally unique ID, slot/rank, seed/reset conditions, static context, start/end status, outcome metrics, simulated duration |
| Transition | Episode ID, t, current observation reference, action, next/final observation reference, named rewards, terminal/truncation/bootstrap flags, time, validity, behavior policy version |
| Optional learner data | Joint old log-probability, value estimate, pre-transform continuous action, action masks, privileged critic observations, collection generation |
| Fragment | Starting history references, continuation/end reason, preceding action, whether complete episode outcome/return is known |

Store observations once and reference adjacent frames where practical. References preserve true terminal successor observations even when the next stored record is a reset. Serialize action component order, allowed categories, bounds, transforms, and masks as part of compatibility hashes. Sinks support configurable flush intervals and bounded queues; backpressure cannot drop transitions.

Add strict importers for existing tabular/PT histories with explicit mappings for episode, observation, action, reward, and boundary fields. Reject gaps, conflicting terminal flags, missing successors for bootstrapped records, and ambiguous action timing. Never assume every file is one complete episode. For final terminal steps without observable successors, permit an explicit missing-next mask only when no bootstrap is used. Split by whole episode (and optionally task/collection group), not windows; detect duplicate episode IDs across train/evaluation splits.

Offline reward relabeling produces a new manifest with parent dataset ID and new reward version. Episode metrics are not automatically copied into every step as rewards. Add return/transition validation utilities and inspectable trajectory summaries.

### 4.4 Inference, evaluation, and configuration

Add shared rollout config referenced by `infer`, new `sequifier rollout --config-path`, and training rollout evaluation sources. `infer` keeps its existing arguments and output behavior when rollout config is absent. `rollout` can start from environments without a file-backed seed dataset.

Replace input/target equality validation with binding coverage only when a rollout binding is present; legacy feedback paths keep their existing checks until routed through an equivalent generated binding. Keep BERT/embedding and multi-position forecast restrictions explicit. No relaxing a validator without implementing the associated image/depth history behavior.

Training-time rollouts run against an immutable evaluation snapshot, separate environment slots and RNG streams, and no learner gradients. Restore model mode after evaluation. Metrics include channel returns, scalarized return, success rate when defined, episode length, completed/censored counts, simulated time, and collection/policy/transport timing. Aggregate weighted sums/counts across ranks; report incomplete episodes separately.

Extend evaluation sources to a discriminated supervised/rollout union and monitors to select source + metric + direction. Preserve legacy total-loss selectors by normalization. Training schedules use epoch/batch triggers for supervised phases and update/environment-step triggers for RL phases. Hyperparameter search can run rollout evaluation using `evaluation_inference_config` pointing to an inference config with rollout settings, or consume native named evaluation metrics; keep legacy `evaluation_script` JSON output intact. Pruning uses the configured comparable metric, not an accidental PPO loss. Disallow incompatible per-trial metric schedules.

Replace loss-specific best-model bookkeeping in `RunState` with a typed monitor identity/value/direction and a migration for existing `best_validation_loss` state. Update best-checkpoint selection, export naming/selection, early stopping, visualization, and search consumers together. Continue to emit legacy supervised loss metrics alongside the new named metric stream so existing integrations remain usable.

### 4.5 Stage 1 acceptance

- An order-book model with level-4 outputs and level-4/level-2 inputs generates multiple steps; synthetic adapter outputs prove that the next model call sees the derived features.
- Camera observations returned after a motor command become the next policy input, including masks, on CPU and the optional GPU path.
- Different episode lengths, partial resets, terminal final observations, truncations, and collection fragments survive collection → storage → loading unchanged.
- PT and ONNX forecast rollouts have matched deterministic results within documented tolerances; all existing inference tests pass.
- A terminal reward is recorded for the final emitted action, adapters close on success/error, and remote timeout/retry behavior is tested.
- Rollout evaluation can select a best checkpoint and return custom hyperparameter-search metrics.

## 5. Stage 2 — return conditioning and complete offline IQL

### 5.1 Policy distributions and model wrappers

Implement these before either IQL policy extraction or PPO:

```python
PolicyRuntime.act(history, *, deterministic, generator) -> PolicyStep
PolicyRuntime.evaluate_actions(history, actions, auxiliary) -> ActionEvaluation
```

`PolicyStep` includes domain action, internal action, joint log-probability, optional value, action mask, and any continuous latent needed for numerically stable likelihood evaluation. `ActionEvaluation` evaluates exactly the same recorded joint action. Its joint log-probability sums over action components, never averages them.

Distributions:

- Categorical and independent MultiDiscrete components, excluding special tokens and respecting state-dependent masks. Reject an all-masked component. Store masks used during collection.
- Ordered autoregressive categorical actions: sample in declared component order; evaluate recorded actions by teacher forcing that same prefix; sum conditional log-probabilities. No sampling a new prefix when scoring a recorded action.
- Unbounded diagonal Gaussian continuous actions with learned/clamped log standard deviation.
- Affine tanh-transformed diagonal Gaussian for bounded continuous actions. Account for the tanh and affine Jacobians and retain pre-tanh actions for PPO. Never externally clip a sampled Gaussian and then score it as if unclipped. Offline actions exactly at bounds require an explicit, reported inverse-transform epsilon convention or rejection; no silent arbitrary clamping.
- Joint mixed actions: explicit factorization and component order. Initially continuous blocks are independent conditional on history; any dependence on sampled categorical actions must be explicitly represented and identically evaluated.

Use FP32 probability math even with lower-precision networks. Analytic entropy is available for independent categorical/unbounded Gaussian policies; for transformed or autoregressive distributions use a documented current-policy Monte Carlo estimator, not entropy evaluated on stale PPO actions. Temperature, masks, normalization, and action bounds are fixed within a PPO collection/update generation.

Freeze dropout and stochastic augmentation for RL action collection and likelihood evaluation, while retaining autograd during optimization (`eval()` does not mean `no_grad()`). Freeze mutable observation-normalization state within a PPO iteration. Use deterministic per-record transforms where any image preprocessing is required. Before the first optimizer update, old/new log-probabilities must match numerically.

Compose separate actor, V, Q1, and Q2 networks using existing ingestion/backbone/encode primitives. Default to no shared trainable parameters between actor, V, and Q networks; this prevents unintended optimizer interaction. Q encodes history first, then combines its representation with the recorded current action using an action encoder. V receives history only. Q1/Q2 may be implemented as a validated twin module but must have independent learned estimates. Share immutable preprocessing specifications, not mutable normalization buffers.

Allow explicitly separate actor and critic observation schemas for asymmetric simulator critics. A privileged critic input must be available in collection, its genuine successor state, offline data when used for IQL, and resume snapshots. It must never enter the actor codec, actor encoder, or exported actor artifact. Validate and test this isolation. Default all networks to the ordinary observation history to keep dataset-only workflows straightforward.

Initialize actor from a compatible supervised policy artifact; initialize critics independently or from explicitly copied encoders. Register all parameters in the parameter catalog, freezing/optimizer routing, initialization provenance, and checkpoint bundle. Ordinary point-regression actor transfer can initialize Gaussian means only under an explicit conversion policy, with new standard-deviation parameters recorded.

### 5.2 Return-conditioned sequence modeling

Implement a named algorithm runtime `return_conditioned` using the policy-aligned window view. This is Decision-Transformer-style return conditioning using Sequifier's multivariate rows; do not claim identical interleaved return/state/action token architecture.

For complete recorded episodes compute:

```text
G_t = reward_t + gamma * G_(t+1), with G_T = 0
```

Gamma is explicit in `(0, 1]`, default 1 for this mode. Compute rewards with the configured fixed scalarization; persist return scale and definition. For incomplete/truncated episodes, require a declared finite-horizon objective or exclude them from exact return-to-go training. Never present an unknown future tail as a known zero return. IQL may still consume bootstrappable fragments from the same dataset.

Add `desired_return` as a dynamic input at the current decision position; actions remain targets, with only previous actions in features. Use the existing supervised optimization services for masked action likelihood loss. Optional deterministic regression loss can be exposed only for the standalone return-conditioned learner; such an artifact is not directly PPO/IQL-compatible until converted into a distributional actor. Default to the shared distributional policy for seamless phase transfer.

Inference requires an initial desired return in raw configured reward units and updates it after each observed reward:

```text
desired_return_(t+1) = (desired_return_t - reward_t) / gamma
```

Apply scaling only at the encoder boundary and reset the budget per episode. Optional clamps must be authored configuration and logged; do not infer attainable targets from evaluation episodes. Unknown future realized returns are never used online.

Compare with behavior cloning and success-filtered behavior cloning on a controlled dataset with conflicting actions and known return differences. Test gamma/scaling arithmetic, episode boundaries, and absence of future-action/reward leakage.

### 5.3 IQL data and update implementation

Add an `iql` runtime that samples fixed-dataset transitions plus sufficient history for both h_t and h_(t+1). Sampling a transition includes up to the configured context length of history and one successor observation; never reconstruct successor context by concatenating arbitrary independently sampled windows. Respect masks, previous-action alignment, static context, and image/depth payload references.

Implement the canonical losses using scalarized rewards, learned V, twin Qs, and slowly updated target Qs. At each update, compute detached targets/advantages from a consistent pre-update snapshot:

```text
q_bar = min(Q1_target(h_t, a_t), Q2_target(h_t, a_t))
delta = stop_gradient(q_bar) - V(h_t)
L_V = mean(abs(expectile - indicator(delta < 0)) * delta^2)

y = reward_t + gamma * bootstrap_allowed_t * stop_gradient(V(h_next))
L_Q = mean((Q1(h_t,a_t)-y)^2 + (Q2(h_t,a_t)-y)^2)

advantage = stop_gradient(q_bar - V(h_t))
weight = min(exp(inverse_temperature * advantage), max_weight)
L_actor = -mean(stop_gradient(weight) * log policy(a_t | h_t))
```

Evaluate the exponential stably by capping its input at `log(max_weight)`. Reject invalid expectile/temperature/target-update settings. Target Qs are frozen copies, updated by Polyak averaging only after a successful critic optimizer step. Define one algorithm update as the configured V/Q/actor transactions and record individual update counters. Recompute forward graphs as necessary; do not backpropagate through detached targets, target networks, or the weight calculation. If AMP skips a transaction, dependent counters/target updates follow the actual applied steps; checkpoint only after the complete update boundary.

Keep initial scope faithful to offline IQL: a fixed training dataset, an independently held-out evaluation environment/dataset, uniform transition sampling, and no fresh data injected during the phase. Expose expectile, inverse temperature, maximum weight, gamma, Polyak factor, batch size, gradient clipping, and separate optimizer/scheduler plans. Reward transforms are fixed. Log Q/V ranges, Bellman residuals, advantage/weight distributions, weight-cap frequency, actor likelihood, update counts, and held-out rollout performance.

RC actor to IQL transfer requires a declared conditioning policy: either retain desired-return conditioning with identical semantics for all actor/critic contexts, or initialize a separately specified unconditioned actor via an explicit conversion/distillation step. Never silently remove an input. The straightforward default workflow is behavior-cloned unconditioned actor → IQL; return-conditioned training is a separate supported route, not a mandatory IQL prerequisite.

### 5.4 Training-plan changes

Replace resolved phase branching with a discriminated union normalized from authored config:

- `supervised`: existing sources, epochs, and source scheduling, unchanged defaults.
- `return_conditioned`: trajectory sources, policy interface, epochs, return definition.
- `iql`: fixed trajectory sources, updates, actor/V/Q specifications, algorithm settings.
- `ppo`: environment source, policy/value specifications, environment-step budget, rollout and update settings (Stage 3).

Legacy phases without an algorithm normalize to supervised. Give online phases their own source type and budget; they must not require dummy `dataset_training` files, criteria, epochs, or validation loaders. Adjust singleton normalization, canonical search config, authored/resolved validation, phase compilation, docs, and CLI routing together. Search spaces can address new discriminated config fields after ordinary canonical composition rules.

Introduce `AlgorithmRuntime` with explicit collect/prepare/update/evaluate/checkpoint boundaries and named counters. Reuse optimization transaction mechanics, random-state handling, distributed strategy, metrics, and integration events. Keep supervised `LossService` intact; algorithm loss functions return their own structured result with accounting counts. Do not implement RL as observer callbacks changing gradients behind the engine's back.

Extend update event identities with algorithm, network/optimizer role, algorithm-update index, and environment-step count. Keep existing supervised identity fields/behavior compatible. Controller directives apply only to the named optimizer transaction; observer/controller state is saved once per declared instance. Validate execution requirements per algorithm so a controller assuming one optimizer cannot silently change the wrong network. Define scheduler clocks explicitly per optimizer update, not in fabricated online epochs.

Phase transitions retain actor weights and compatible codecs, but reset optimizer moments by default when algorithms change. Copy/reset value state only under explicit compatible specs. Record transfer provenance and validate action/observation schema, context length, return conditioning, and bounds before copying. Artifact warm-start and exact run resume are different operations.

### 5.5 Stage 2 acceptance

- Return-conditioned policies change action choices with requested return on a deterministic fixture; online return-budget arithmetic matches offline labels.
- IQL losses, gradient ownership, expectile asymmetry, weight caps, and target updates match hand-computed examples.
- A tiny tabular/analytic control task demonstrates learned value ordering and policy improvement against a deliberately mixed-quality behavior dataset over fixed seeds; include tolerances and report variance rather than asserting monotonic improvement per update.
- Actor/V outputs are invariant to replacing current-action labels or future observations; Q changes appropriately with its explicit action input.
- Distribution sampling/scoring round trips, categorical prefixes, action masks, and bounded Gaussian Jacobians pass numerical tests.
- Exact offline resume reproduces uninterrupted sampling and updates within device-appropriate deterministic tolerances; actor-only exports reproduce pre-export policy outputs.

## 6. Stage 3 — full synchronous PPO

### 6.1 Collection and rollout buffer

Freeze a behavior policy version per iteration. Collect `num_envs × rollout_length` decisions per rank using the Stage 1 coordinator. Run collection without autograd; store observations/history references, sampled actions and latent values, joint old log-probabilities, old V estimates, scalar/raw rewards, final observations, terminal/truncation flags, masks, and policy version.

A completed episode resets immediately in its slot and starts a new history. A collection cut preserves live context for the next iteration. PPO may change the policy between fragments of the same episode; each transition retains its behavior version and only the current iteration's buffer is optimized. Treat environment-step count as the total across ranks. Validate a total budget compatible with fixed-size synchronous collection or explicitly define a final smaller masked rollout; never silently overshoot the authored budget.

Preallocate device rollout tensors/rings for numeric/image payloads and avoid duplicating full history windows per transition. Retain enough pre-rollout context to reconstruct every training history exactly. If storage is insufficient, fail before collecting, with a memory estimate. Permit CPU/pinned staging where configured; benchmark transport separately. Seal the buffer before calculating targets or updating. Reuse each buffer only for its configured PPO optimization epochs, then discard it from on-policy training (optional trajectory archival is independent).

### 6.2 Bootstrap and GAE

Evaluate V on each genuine successor/final observation using the corresponding successor history, with the same frozen value snapshot used during collection. Let b_t indicate whether bootstrap is permitted; let c_t indicate whether GAE recursion continues into the next stored transition of this same episode and collection segment:

```text
td_t = reward_t + gamma * b_t * V_old(h_next) - V_old(h_t)
A_t = td_t + gamma * lambda * c_t * A_(t+1)
value_target_t = stop_gradient(A_t + V_old(h_t))
```

True terminal: b=0, c=0. Time-limit truncation with valid final state: b=1, c=0. Ordinary transition inside the segment: b=1, c=1. Last live transition at a collection cut: b=1, c=0, so it bootstraps without accessing the next collection's advantages. Never bootstrap using reset observations or carry GAE across episode IDs.

Compute GAE in FP32. Normalize advantages once over globally valid transitions per iteration using DDP sums/counts (configurable off); do not normalize rewards implicitly or re-normalize each minibatch. Keep returns and old values fixed through optimization.

### 6.3 PPO loss and optimization

Evaluate each recorded action on the exact same observation history, mask, codecs, and frozen preprocessing state. Joint ratio and losses:

```text
ratio = exp(logp_new - logp_old)
L_policy = -mean(min(ratio * A,
                     clip(ratio, 1-epsilon, 1+epsilon) * A))
L_value = 0.5 * mean((V_new - value_target)^2)
L_total = L_policy + value_coefficient * L_value
          - entropy_coefficient * entropy_current
```

Expose optional value clipping around `V_old` as a documented maximum-of-clipped/unclipped loss, disabled by default. Implement optimizer gradient clipping and optional target-KL early stopping. Estimate KL consistently (for example `mean((ratio-1)-log_ratio)`) and aggregate globally; every rank stops at the same boundary. Log clipping fraction, ratio tails, entropy, KL, gradient norm, explained variance, value loss, returns, episode success, steps/sec, and timings. Detect non-finite ratios/losses and fail the iteration; do not hide them with arbitrary ratio clamping outside PPO's specified clipped objective.

Use separate actor and value modules/optimizers by default, with no shared parameters. Combined mathematical loss can be split into separate transactions; both optimize the same sealed targets. Support configurable optimization epochs, minibatch size, and gradient accumulation with correct valid-sample weighting. Randomly sample decision indices and reconstruct complete fixed-length causal contexts; do not shuffle time positions inside a context. This is a finite-context transformer policy, not a persistent recurrent state requiring truncated BPTT. Any future hidden-state/KV-cache optimization must prove equivalence after parameter updates and resets.

The complete PPO schema includes collection horizon/environment count, total budget, optimization epochs, minibatch/accumulation sizes, gamma, lambda, clipping epsilon, optional value-clip range, value/entropy coefficients, optional target KL, advantage normalization, actor/value optimizer and scheduler plans, gradient limits, action standard-deviation initialization/bounds, evaluation intervals, and checkpoint/resume policy. Validate positivity/ranges, actor/value parameter disjointness, batch coverage, divisibility or explicit remainder masks, and supported execution/distribution combinations before instantiating expensive simulators. Do not hide these controls in free-form adapter configuration.

Use the shared distributions for discrete, continuous, ordered categorical, and mixed policies. A density-free deterministic regression head is invalid for PPO. Default evaluation uses mode/mean actions and no exploration; report separately if stochastic evaluation is requested. Deployment returns the same bounded/domain actions as collection.

### 6.4 Distributed synchronization and simulator execution

Each DDP rank has its own simulator batch, unique episode/slot IDs, and derived seeds based on run, rank, slot, and reset count. Actor/value parameters are synchronized before collection. All ranks perform compatible collect/update schedules, use globally weighted valid-sample losses, and agree on KL stopping, errors, pruning, and checkpoint boundaries. Avoid rank-zero-only environment collection feeding a nominal DDP learner.

For IQL, partition/sample offline data with deterministic rank-specific samplers and synchronized update counts; target networks update from synchronized Q parameters. For all new runtimes, distinguish valid transition counts from padded context lengths in loss denominators and metric reductions.

Remote adapters use batched RPC calls and one connection/session per declared rank namespace. A slow service may reduce throughput but must not silently mix policy generations. Support GPU tensor adapters without mandatory CPU synchronization per field. Simulator APIs, transport, rewards, and preprocessing have separate timings so performance problems are diagnosable.

### 6.5 Checkpoints and resume

Version the algorithm checkpoint payload and retain a reader/migration for existing supervised format-v2 artifacts. Save model specs/weights for actor, V, Q/target Q where relevant; separate optimizer/scheduler/scaler states; algorithm counters; dataset samplers; policy version; random streams; normalization; codec/action spec hashes; phase transfer provenance; and environment/reward state capabilities.

For PPO, write durable checkpoints at complete iteration boundaries after optimizer updates, target computation, and metric accounting have finished. Exact resume additionally requires snapshots of live simulator states, histories, reset counters, reward state, and all rank RNG streams. If the adapter cannot snapshot, provide an explicit `reset_environments` resume mode: preserve learner state, reset environments/history, discard incomplete episode metrics, and label the run as a continuation without trajectory-equivalent resume. `exact` mode must fail rather than claim reproducibility. Mid-collection/mid-minibatch resume is not required in this release; a crash restarts from the last committed boundary.

Keep actor-only model artifacts small and independent of critic state. Learner checkpoints contain the complete training bundle. Validate every network and optimizer assignment during restore; no missing-critic fallback in exact resume. Atomic manifests must not point to partially written rank state.

### 6.6 Stage 3 acceptance

- Hand-computed PPO clipping tests cover positive/negative advantages, joint action likelihoods, optional value clipping, entropy, and KL stop behavior.
- GAE tests cover true terminals, truncations, collection cuts, autoresets, nonuniform episode lengths, and missing invalid successor observations.
- The identity-ratio invariant holds before any update, including image inputs, ordered action heads, and checkpoint round trips.
- Discrete and bounded continuous control environments show reliable learning over multiple seeded runs. Keep short deterministic CPU learning checks in CI; run longer multi-seed quality benchmarks separately and save results.
- An RGB toy control policy completes collection → PPO update → evaluation → checkpoint → reload → rollout. A structured/depth observation fixture proves the same schema path.
- DDP results agree with single-process global-batch loss calculations within tolerance; one rank hitting KL/error/termination does not hang others.
- Exact-snapshot adapters reproduce an uninterrupted continuation; non-snapshot adapters correctly reject exact resume and pass explicit reset-mode continuation.
- Remote vector environments can train PPO using the same algorithm runtime as in-process tensor environments.

## 7. Authoring surface and operational workflow

All names below are proposed new fields. Implement schema-validated examples with the complete existing model configuration in documentation/examples; these snippets show the new surface only.

```yaml
rollouts:
  robot:
    environment:
      id: toy_robot
      factory: examples.toy_robot:make_environment
      config: {num_envs: 32}
    reward:
      scalarization: {task_success: 1.0, effort: -0.01}
    bindings:
      actions: {motor_command: motor_command}
      inputs:
        camera: {source: environment, field: camera}
        joint_position: {source: environment, field: joint_position}
        previous_command: {source: prediction, field: motor_command}

training_plan:
  phases:
    - name: offline_improvement
      algorithm: iql
      updates: 100000
      trajectory_sources: [recorded_robot_training]
      policy: robot_policy
      value: robot_value
      critics: robot_q_twins
      gamma: 0.99
      expectile: 0.7
      inverse_temperature: 3.0
      max_weight: 100.0
      target_update_rate: 0.005
    - name: simulated_refinement
      algorithm: ppo
      environment_source: robot
      policy: robot_policy
      value: robot_value
      total_environment_steps: 1048576
      rollout_length: 128
      optimization_epochs: 4
      minibatch_size: 512
      gamma: 0.99
      gae_lambda: 0.95
      clip_epsilon: 0.2
```

Stage 2 adds named trajectory sources and policy/value/critic registries to the canonical config. Existing model interface definitions are referenced where suitable; avoid embedding unrelated interface authoring syntax in each algorithm config. Model registry validation must catch accidental parameter sharing and incompatible copied interfaces. `rollouts` is a shared named registry consumed by inference, rollout collection, and evaluation. Expose consistent factory resolution relative to the project root/import environment, with explicit error messages for missing modules.

Commands:

```text
sequifier rollout --config-path configs/collect.yaml
sequifier preprocess --config-path configs/import-trajectories.yaml
sequifier train --config-path configs/train-return-conditioned.yaml
sequifier train --config-path configs/train-iql-ppo.yaml
sequifier infer --config-path configs/infer-with-dependent-features.yaml
```

Extend preprocess with an explicit trajectory import/relabel operation union rather than treating transition metadata as ordinary selected features. A Python API supports the same operations and direct local adapter instances. Update `sequifier docs`, `make` templates, consolidated docs generation, documentation/configs, and release notes. Publish examples for dependent scalar variables, RGB control, offline return conditioning, IQL warm-start, and PPO with local and remote simulators.

Clarify numerical versus behavioral compatibility: ONNX exports expose policy parameters with action-spec/codec metadata, and the shared runtime performs sampling/scoring. Exported stochastic trajectories need not match PT draw-for-draw across backend RNGs; distribution parameters and deterministic actions must match within tested tolerances. Store ordered categorical decoding dependencies and implement equivalent stepwise ONNX sampling, or reject the particular export at preflight. Never label an exported logits tensor as a complete bounded-action policy without its transforms.

## 8. Implementation order within the single update

| Work package | Depends on | Concrete deliverable |
|---|---|---|
| A. Contracts and schema | None | Typed rollout/trajectory/action specs, phase union design, alignment and boundary fixtures |
| B. Coordinator and adapters | A | Dependent-variable inference, tensor/Gym/remote adapters, history/mask handling |
| C. Trajectory data | A/B | Atomic store/import/relabel, episode splits, history-aware loading |
| D. Evaluation integration | B/C | Rollout metrics, monitors, search/pruning support, inference regressions |
| E. Policy primitives | A | Distributions, actor/V/Q wrappers, codecs, log-probability tests, actor exports |
| F. Learner orchestration | C/E | Algorithm runtime, config union, optimizer/checkpoint bundle and migration |
| G. Return conditioning | C/E/F | Return labels, training, inference budget updates and examples |
| H. Offline IQL | C/E/F | Complete learner, targets, weighted actor extraction, resume and evaluation |
| I. PPO | B/D/E/F | Collection buffer, GAE, PPO optimization, DDP and resume semantics |
| J. Integrated release validation | G/H/I | End-to-end examples, numerical/learning/performance checks, docs and migration notes |

Build contracts and small mathematical fixtures first; add learning tests once the corresponding algorithm exists. Do not combine algorithm validation with heavy simulator installation. Stage gates should be reviewable commits within the same update, with the release gated on J.

## 9. Verification, performance, and release criteria

Preserve all existing supported preprocessing, supervised learning, inference, artifact, and export behavior. Run the existing required suite and targeted tests for every changed subsystem. New tests must prove mathematical/boundary behavior rather than mirror private implementation details.

Maintain three validation tiers:

1. Fast CPU tests: schema validation, temporal alignment, codec round trips, distributions/gradients, return arithmetic, IQL/PPO/GAE formulas, partial resets, artifact migration, remote failure handling.
2. Integration tests: CLI/Python parity, file/tensor/media sources, actor export, exact/reset-mode resume, training phase transfer, rollout monitors/hyperparameter search, two-process CPU DDP and optional CUDA DDP.
3. Reproducible learning/performance jobs: fixed datasets/tasks/seeds, BC and return-conditioned and IQL comparisons, PPO discrete/continuous/image tasks, local versus remote adapter equivalence, and measured throughput/memory.

Before optimization, record an agreed baseline on specified hardware with context length, model size, image resolution, simulator count, and DDP topology fixed. Measure policy inference, environment/reward stepping, transfer, serialization, update time, peak memory, and total valid transitions/sec separately. Require no accidental per-field GPU-to-CPU copies in the native tensor collection path and bounded memory over long runs. Set quantitative throughput regression limits after the baseline is measured; do not invent a hardware-independent performance promise.

The update is complete when all Stage 1/2/3 acceptance criteria pass, the old test suite remains green, the five documented workflows run from clean optional-dependency installations, checkpoints/exports have tested compatibility, and unsupported algorithm/model combinations fail before long collection or training begins. Publish benchmark configuration and results, including seeds and uncertainty, without claiming general robotics SOTA from toy tasks.

## 10. Algorithm references and interpretation

- [Decision Transformer](https://arxiv.org/abs/2106.01345): return-conditioned causal action modeling. The proposed multivariate-row representation is a Sequifier-specific implementation choice.
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169): expectile value learning, Q backups, and advantage-weighted policy extraction. The separate-network, runtime, and storage designs above are proposed integration decisions.
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347): clipped policy objective and repeated minibatch optimization of recent rollouts.
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): value-based advantage estimation. The explicit adapter boundaries, checkpoint contracts, and terminal-observation handling are requirements of this implementation plan.
