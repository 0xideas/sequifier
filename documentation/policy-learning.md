# Policy learning

Use `sequifier train --config-path configs/train-policy.yaml`. All algorithms use
`training_plan.phases`; model interfaces and datasets follow the
[training reference](configs/train.md). Start from the shipped
[imitation](../src/sequifier/templates/train-policy.yaml),
[IQL](../src/sequifier/templates/train-iql.yaml), [PPO](../src/sequifier/templates/train-ppo.yaml),
[DPO](../src/sequifier/templates/train-dpo.yaml), or [GRPO](../src/sequifier/templates/train-grpo.yaml)
template. [Artifact transfer](supervised-policy-learning.md) covers joint real/categorical
actions and mixed supervised/policy plans.

## Authoring and optimization

`policy_bindings` maps interface names to action declarations, including runs
with one interface. `model.execution` owns context geometry; interface input
contracts own frozen normalization; datasets own raw split policy. `run` owns
resources, reproducibility and recovery. Phases own learning, trainability,
optimization, evaluation, stopping, snapshots and `model_selection.handoff`.
See the [ownership reference](configs/ownership.md).

| Algorithm | Required budget/data | Trainable roles |
| --- | --- | --- |
| `policy_imitation` | `epochs`, `sources` | `actor` |
| `iql` | `iterations`, transition `sources` | `actor`, `value`, `critic` |
| `dpo` | `iterations`, `preferences` | `actor` |
| `ppo` | `total_env_steps`, `rollout_env_steps`, `environment`, `evaluation` | `actor`, `value` |
| `grpo` | `total_groups`, `max_steps`, `environment`, `evaluation` | `actor` |

Each role defaults to Adam, learning rate `0.0003`, gradient clip `1.0`, and
`ConstantLR` with factor `1.0`, total iterations `1`. Optimizers accept constructor
arguments; grouped optimizer plans set their own rates. Policy schedulers advance
on optimizer steps; plateau/compound schedulers are unsupported. `OneCycleLR.total_steps` is derived from the phase budget;
an explicitly supplied value must match that budget. Unknown roles fail validation.

Imitation, DPO and GRPO default to `batch_size: 32`, `accumulation_batches: 1`,
`evaluation.cadence: {iterations: 1}` and `checkpoints.cadence: {iterations: 1}`. Accumulation uses actual
example counts, including partial batches. Network stochastic layers are disabled
while gradients remain enabled. Best handoff keeps the earliest finite best on ties;
last handoff keeps the latest transaction. Final updates always evaluate.
See [continuation rules](configs/train.md#optimization-across-phases) for
`optimizer_state: continue`; its default is `reset`.

## Offline data

Sequence parts accept CSV/Parquet tables, JSON trajectories, checksummed sequence
manifests and prepared histories. At row `t`, observations and lagged actions form
inputs; the recorded current action is the label, and reward/successor describe
its resulting transition. Tables use `sequenceId` and `itemPosition`.
Observations and static inputs use the same names in the table, metadata and model.
When importing without a model, declare those names in the
`source.columns.observations` and `source.columns.static_inputs` lists.

Current actions use their policy field names or a structured `__executed_action`
column. Previous actions use `__previous_action_<action>` scalar columns or a
structured `__previous_action` column. Optional transition columns are `reward`,
`terminated`, `truncated`, and `bootstrap_allowed`. When successors are present,
provide `__next_<observation>` for every observation. Use `episodeId` for a
stable episode ID and `__group` for a collection group.
Rename external table columns to these names before importing or training.
Fragments require an explicit preceding action; file boundaries alone imply no
termination. Forecast trajectories require `action_semantics: executed_predictions`.

Automatic splitting keeps transitive episode/group/duplicate-content components
together and requires at least two components. Explicit `evaluation.sources`
undergo the same overlap checks. Codecs and moments are fitted only on training
coverage. Indexed storage and bounded episode caches are shared across phases;
windows are constructed on demand. `import_sequences` exports canonical raw
Parquet shards and a checksummed manifest; see the
[import example](example_inputs/configs/import-policy-sequences.yaml).

Imitation minimizes joint action NLL. Return conditioning requires rewards and
a known tail: true termination or an explicit finite return horizon. The label at
`t` includes `r_t`; unknown tails are excluded. Offline monitors are fixed to
`{metric: validation_action_nll, direction: min}` for imitation/IQL and
`{metric: validation_dpo_loss, direction: min}` for DPO. Only
`evaluation.sources` and `evaluation.monitor` are accepted; alternate spellings
are rejected.

IQL requires complete reward, terminated, truncated and bootstrap columns, plus
true successors for bootstrapped transitions. It samples uniformly with replacement,
fits expectile V and twin Q-functions, updates target Q-functions with Polyak
averaging, and weights actor NLL by
`min(exp(inverse_temperature * (min(target_Q1,target_Q2)-V)), max_advantage_weight)`.
Defaults: `batch_size: 64`, `gamma: 0.99`, `expectile: 0.7`,
`inverse_temperature: 3`, `max_advantage_weight: 100`, `target_update_rate: 0.005`,
evaluation/checkpoint intervals `100`. Held-out action NLL selects the actor;
critic diagnostics are separate. Critics are fresh per phase and are not exported.

## DPO preferences

A preference manifest declares `format: sequifier.preference-pairs`, `version: 126` (`ContractVersions.PREFERENCE_PAIR_MANIFEST`)
and `pairs`. Each pair has `pair_id`, `context_id`, optional `group_id`, and
`chosen`/`rejected` spans `{source, episode_id, start, end}`. Bounds are zero-based,
start-inclusive/end-exclusive within loaded episode rows. Initial executable
histories must match. Duplicate IDs, overlapping/identical alternatives and
incompatible contexts fail before training. Rewards do not imply preferences.

DPO minimizes `-logsigmoid(beta * ((s_actor_chosen-s_actor_rejected) -
(s_ref_chosen-s_ref_rejected)))`, where each score sums valid decision log likelihoods.
Pairs have equal weight, regardless of length. Defaults: `beta: 0.1`,
`scoring_chunk_size: 64`. Two-pass chunked backward preserves the scalar derivative
while bounding activation memory. Optional `evaluation.sources` selects held-out
preference manifests.

DPO and GRPO default to `reference: {source: phase_start}`; an artifact reference
uses `{source: artifact, path: ...}`. References are independent frozen snapshots.
Embedded checkpoint references survive removal of their original files;
a still-present changed reference is rejected.

## Online adapters

PPO, GRPO, evaluation and policy rollout use `ContractVersions.ONLINE_POLICY_ADAPTER`:
`environment: {factory: module:factory, implementation_id: ..., config: {}, units: {}}`.
Factories receive `config` as keyword arguments. Implementation identity and units
must match the declaration. `OnlineEnvironmentAdapter`, `EnvironmentCapabilities`,
`OnlineResetResult` and `OnlineStepResult` are exported by `sequifier.rl_api`.

Adapters implement `reset(seed)`, `step(named_domain_actions)` and `close()`;
policy rollout also requires `reset_from_observation(initial, seed)`.
Reset results provide observations/static inputs/units. Step results provide reward,
separate termination/truncation booleans, successor observation and optional success.
Never autoreset or substitute a reset observation for a final successor. A true
terminal policy step may omit its successor; other steps must retain it.

Capabilities declare seeded reset, exact snapshot support and group-start support.
Exact PPO recovery requires `snapshot_state()`/`restore_state()` including simulator
RNG. GRPO requires `create_group_start(seed)` and `reset_group(token, rollout_seed)`;
all group members restore the same portable token with independent future streams.
Exact GRPO recovery requires reproducible group starts. Evaluation factories must
create independent adapters. The [control adapter](../src/sequifier/rl/control.py)
is a complete local example.

## Initialize PPO and IQL from a pretrained history encoder

Set `history_initialization: phase_start_actor` on a PPO or IQL phase with
`optimizer_state: reset`. PPO initializes its value encoder from the incoming
actor. IQL initializes the value and both Q encoders from that same actor. Each
network owns an independent copy of the transformer and input encoders. The
scalar heads and the two Q action encoders start fresh, and each target Q starts
as a copy of its corresponding critic.

With `initialize_from`, the first phase receives the pretrained actor. Later
phases use the actor selected at the preceding handoff. The default
`history_initialization: random` uses fresh value and Q encoders. A phase with
`optimizer_state: continue` restores the preceding networks and optimizer state.

The transferred history parameters follow the phase’s usual trainability rules.
An optimizer plan can fine-tune the backbone at a lower rate than the heads.
For example, put the following under a PPO or IQL phase's `optimization` and
use the same grouping for `actor` or IQL's `critic` role as needed.

```yaml
value:
  optimizer:
    groups:
    - id: backbone
      select: {module: backbone}
      optimizer: {name: Adam}
      learning_rate: 0.00003
    - id: remaining
      select: otherwise
      optimizer: {name: Adam}
      learning_rate: 0.0003
```

See [supervised models and policy learning](supervised-policy-learning.md) for
initialization from artifacts whose decoder predicts different targets.

## PPO and GRPO

PPO seals one rollout generation, computes boundary-aware FP32 GAE, then optimizes
clipped actor and independent V objectives. Truncation bootstraps from the final
successor without carrying GAE across reset. Collection horizons retain live history.
Defaults: `optimization_epochs: 4`, `batch_size: 64`, `accumulation_batches: 1`,
`gamma: 0.99`, `gae_lambda: 0.95`, `clip_epsilon: 0.2`, `value_coefficient: 0.5`,
`entropy_coefficient: 0.01`, `normalize_advantages: true`, `reward_scale: 1`.
Optional `value_clip_delta` clips V changes; `old_policy_target_kl` stops old-policy drift.
`allow_partial_rollout: true` executes the remaining budget exactly; false requires
divisibility. Optional `archive_directory` writes independent collection records;
`archive_scores` defaults to false. Archives are not update sources.

GRPO seals complete same-start groups before updating an actor-only model.
Scores sum scaled rewards through termination, truncation or `max_steps`.
`A_i = (R_i-mean(R))/(population_std(R)+advantage_epsilon)`.
Its loss averages equally over trajectories, then over their decisions:
`-min(r*A, clip(r)*A) + reference_kl_coefficient*(exp(d)-d-1)`, where
`r=exp(log_pi-log_pi_old)` and `d=log_pi_ref-log_pi`.
Defaults: `group_size: 4`, `groups_per_iteration: 1`, `optimization_epochs: 4`,
`advantage_epsilon: 1e-6`, `clip_epsilon: 0.2`, `reference_kl_coefficient: 0.04`,
`reward_scale: 1`, `scoring_chunk_size: 64`. Optional `old_policy_target_kl` limits
old-policy drift separately. The sampled reference penalty is not exact current-policy KL.
`max_env_steps` uses a conservative complete-group bound. Failed/incomplete groups
are discarded without retry; errors report issued actions. Both online algorithms
default to `max_buffer_bytes: 536870912` and reject stale/reused buffers.

Image encoders accept `microbatch_size` (default `16`) independently of GRPO's
`scoring_chunk_size`. Smaller image microbatches reduce peak frozen-backbone
activation memory without changing policy optimizer-step boundaries. GRPO
collection runs sequentially in the training process. Iteration metrics report
collection seconds, collected environment steps per second, update seconds and
evaluation seconds.

GRPO defaults to `simulator_workers: 1`, preserving the in-process sequential
collector. Setting a larger value explicitly enables persistent spawned simulator
processes and batched actor/reference encoding. Active workers are capped at
`groups_per_iteration * group_size`; `rollout_inference_batch_size` (default `32`)
caps each inference call. `worker_threads` (default `1`) bounds PyTorch intra-op
threads per simulator, and `worker_timeout_seconds` (default `120`) sets response
deadlines from dispatch. Worker factories/configurations must be importable and
picklable; custom Python entry points require a `__main__` guard.

An adapter must explicitly declare `supports_parallel_group_start = True` before
workers can be enabled. This promises that a portable group token restores the
same scenario in an independent adapter instance and that `create_group_start`
depends on its seed/configuration, not the live trajectory. Existing adapters
remain valid with one worker. Construction/worker errors abort instead of silently
falling back or retrying issued actions. Pending groups are never checkpointed.
Pools close on all exit paths; unresponsive processes are terminated after a
bounded cleanup period.

Every sample retains its independent action and environment seed streams. GRPO
training does not calculate entropy or sample unused entropy-estimator actions,
in either collector. Seeded trajectories therefore differ from older versions
that consumed those extra draws; PPO entropy computation is unchanged. All samples
use a single actor version and all complete groups are sealed before optimization.
Group ordering, trajectory weighting, optimizer boundaries and old-score checks
are unchanged.
Floating-point results may vary with inference batch shape; cross-device or
cross-configuration bitwise equivalence is not promised.

Frozen image feature caching is independently opt-in through
`image_feature_cache_bytes`; see the training reference. Caches are transient,
memory-bounded and scoped to a phase. Metrics include actual simulator count and,
when caching is active, cumulative cache hits/misses and retained feature bytes.
The default disabled options are omitted from serialized contracts to preserve
existing plan/artifact identities. Enabled options are part of the resume plan.

These options do not guarantee a speedup for every workload or machine. Compare
collection steps/second, update time and peak memory against the default paths.
Increasing groups per iteration changes update frequency; compare equivalent
learning budgets. Additional workers and caches require CPU/GPU memory headroom.

Evaluation uses independent seeds/adapters, restores training modes and RNG, and
requires `environment`, `seeds`, `max_episode_steps`, `timeout_seconds` and the
main-thread POSIX timer. `action_selection` defaults to deterministic. PPO's default
monitor is `mean_completed_return`/`max`; `mean_completed_length` and `success_rate`
are supported. Censored episodes and unavailable metrics are explicit. GRPO uses
`score_mean`/`max`, including declared horizons and the fixed reward scale.
Online intervals count completed collection/update iterations.

## Checkpoints and inference

Policy runs use the shared `RUN_CHECKPOINT` envelope with a `POLICY_RECOVERY`
payload and a `POLICY_PHASE_STATE` record for the live phase. They retain representation, plan/source/partition/adapter
identities, selected phase summaries and checksums, RNG and live phase
networks/optimization/sampler/selection state. Training and evaluation histories
are journaled under each native phase directory. Keep the journal and committed
phase models with the checkpoint so reports can be regenerated from the recorded
positions. Mixed plans also retain their phase completion receipts. See
[Training artifacts](training/artifacts.md).

Source files and output-path conflicts are rechecked at phase boundaries.
Atomic publication precedes checkpoint advancement, so completed phases never rerun.

`resume.mode: exact` validates simulator snapshots or reproducible group starts.
`reset_environments` preserves updates but reports a trajectory discontinuity.
Offline resume restores sampler/order and optimizer transactions. Neither online
mode reuses partial groups or stale samples. Final portable artifacts contain
actor execution, codecs, normalization, policy and units, without critics or references.

`load_train_config(path, {}, False)` loads/resolves the common schema;
`sequifier.train.run_training` dispatches its phases. Python APIs and environment
contracts are exported by `sequifier.rl_api` and `sequifier.training_api`.
For prediction, task routing, probability outputs and policy rollout, use the
[inference reference](configs/infer.md#policy-model-interfaces) and
[rollout guide](rollout.md). Deterministic execution, action selection and RNG seed
are independent settings; reproducibility also depends on runtime/device/order.

Model scale parameters and policy action behavior use separate schemas; see
[model prediction parameters and policy actions](configs/train.md#model-prediction-parameters-and-policy-actions).
Normalization is global or inherited from the artifact and cannot be configured
on an individual phase.
