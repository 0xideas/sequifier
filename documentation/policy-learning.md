# Policy learning

Use `sequifier train --config-path configs/train-policy.yaml`. All algorithms use
`phases`; model interfaces and datasets follow the
[training reference](configs/train.md). Use `sequifier make` to create standalone
training configs for imitation, IQL, PPO, DPO and GRPO. The three control examples
combine a [shared actor and binding](../src/sequifier/templates/control/actor.yaml)
with the settings for [PPO](../src/sequifier/templates/control/train-ppo.yaml),
[DPO](../src/sequifier/templates/control/train-dpo.yaml), or
[GRPO](../src/sequifier/templates/control/train-grpo.yaml).
[Artifact transfer](supervised-policy-learning.md) covers joint real/categorical
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
| `policy_imitation` | `budget.epochs`, `sources` | `actor` |
| `iql` | `budget.iterations`, transition `sources` | `actor`, `value`, `critic` |
| `dpo` | `budget.iterations`, preference `sources` | `actor` |
| `ppo` | `budget.env_steps`, `collection.env_steps_per_iteration`, `collection.environment`, `evaluation` | `actor`, `value` |
| `grpo` | `budget.groups`, `collection.max_steps`, `collection.environment`, `evaluation` | `actor` |

Each role defaults to Adam, learning rate `0.0003`, gradient clip `1.0`, and
`ConstantLR` with factor `1.0`, total iterations `1`. Optimizers accept constructor
arguments; grouped optimizer plans set their own rates. Policy schedulers advance
on optimizer steps; plateau/compound schedulers are unsupported. `OneCycleLR.total_steps` is derived from the phase budget;
an explicitly supplied value must match that budget. Unknown roles fail validation.

Imitation, DPO and GRPO default to `batch_size: 32` and `accumulation_batches: 1`.
Accumulation uses actual example counts, including partial batches. Network
stochastic layers are disabled while gradients remain enabled.

Set `checkpoints.cadence: {iterations: N}` on a phase to retain a snapshot every
`N` committed iterations. The default is 10 for imitation, DPO, PPO and GRPO,
and 100 for IQL. Phase completion retains a snapshot even when the final
iteration falls between cadence points. Evaluation has its own cadence and
still runs on the final update.

Best handoff keeps the earliest finite best on ties. Consecutive policy phases
carry that candidate forward when they share an actor representation and an
evaluation contract. Offline phases compare the held-out data and scoring
settings, including reference weights and beta for DPO. Online phases compare
the evaluation environment, seeds, episode limits, action selection, and score
definition. Set `model_selection.scope: phase` to start a fresh best history.
`handoff: last` still passes the latest weights, and stopping patience remains
phase-local. The report's `provenance.selected_phase_index` identifies the phase
that produced the selected weights.

See [continuation rules](configs/train.md#optimization-across-phases) for
`optimizer_state: continue`; its default is `reset`.

## Offline data

Sequence sources accept CSV or Parquet observation rows, a checksummed sequence
manifest, or prepared histories. Declare the observation and static column names
under `source.columns` when importing a table without a model. Each sequence
starts at zero and ends with the genuine final observation. Every position
after zero carries the complete action that produced its observation. Optional
`reward`, `terminated`, and `truncated` columns describe that same transition.
The history through row `t` predicts the action on row `t+1`. Termination disables
bootstrap. A truncation retains its observed successor.

Use `source.action_columns` to map logical actions to physical columns,
`sequenceName` for a stable episode label, and `groupId` for related episodes.
Forecast predictions used as actions require `action_semantics: executed_predictions`.

Setting `partition.validation_fraction` generates explicit train and validation views. The partition keeps transitive episode/group/duplicate-content components
together and requires at least two components. Explicit `evaluation.sources`
undergo the same overlap checks. Codecs and moments are fitted only on training
coverage. Indexed storage and bounded episode caches are shared across phases;
windows are constructed on demand. `import_sequences` exports a canonical raw
Parquet observation table and its checksummed manifest. See the
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

A preference manifest declares `format: sequifier.preference-pairs` and
`pairs`. Each pair has `pair_id`, `context_id`, optional `group_id`, and
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

For direct training from raw sequences, declare bounded categorical observation or
static inputs under `model.interfaces.<name>.input_contract.cardinality_config`.
For example, `{place: {min_count: 3, hashing: {num_buckets: 16}}}` keeps frequent
labels and hashes the rest. The first offline fitting phase owns the vocabulary.
Prepared sources carry their fitted rules in metadata.

Every label in a categorical action's `values` must have the same scalar type.
An explicit sequence source `units` mapping must match the policy binding, including
an explicitly empty mapping. Rollout data carries the producing policy's units.

Declare each training adapter in `environments.<name>` with `factory`, `config` and `units`. PPO and GRPO select an adapter by name in `collection.environment` and `evaluation.environment`. A standalone inference rollout supplies the adapter declaration in `rollout.environment`.
Factories receive `config` as keyword arguments. Adapter-reported units must match
the configuration. Custom adapters subclass `OnlineEnvironmentAdapterBase` from
`sequifier.rl_api` and supply a nonempty `implementation_id` on the adapter.
Sequifier checks that identity when resuming a run. The same module exports
`EnvironmentCapabilities`, `OnlineResetResult`, and `OnlineStepResult`.

Adapters implement `reset(seed)`, `step(named_domain_actions)` and `close()`;
policy rollout also requires `reset_from_observation(initial, seed)`.
Reset results provide observations/static inputs/units. Step results provide reward,
separate termination/truncation booleans, successor observation and optional success.
Every step, including termination, must return its actual successor before resetting the environment.

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
divisibility. Set `collection.output.manifest_path` to publish the decision table.
Set `collection.output.include_scores: true` to include PPO scores in its
`__collection` column.

GRPO seals complete same-start groups before updating an actor-only model.
Scores sum scaled rewards through termination, truncation or `max_steps`.
`A_i = (R_i-mean(R))/(population_std(R)+advantage_epsilon)`.
Its loss averages equally over trajectories, then over their decisions:
`-min(r*A, clip(r)*A) + reference_kl_coefficient*(exp(d)-d-1)`, where
`r=exp(log_pi-log_pi_old)` and `d=log_pi_ref-log_pi`.
Defaults: `collection.group_size: 4`, `collection.groups_per_iteration: 1`, `optimization_epochs: 4`,
`advantage_epsilon: 1e-6`, `clip_epsilon: 0.2`, `reference_kl_coefficient: 0.04`,
`reward_scale: 1`, `scoring_chunk_size: 64`. Optional `old_policy_target_kl` limits
old-policy drift separately. The sampled reference penalty is not exact current-policy KL.
`budget.max_env_steps` uses a conservative complete-group bound. Failed/incomplete groups
are discarded without retry; errors report issued actions. Both online algorithms
default to `collection.max_buffer_bytes: 536870912` and reject stale/reused buffers.

Image encoders accept `microbatch_size` (default `16`) independently of GRPO's
`scoring_chunk_size`. Smaller image microbatches reduce peak frozen-backbone
activation memory without changing policy optimizer-step boundaries. GRPO
collection runs sequentially in the training process. Iteration metrics report
collection seconds, collected environment steps per second, update seconds and
evaluation seconds.

GRPO collects in-process by default. Set `collection.parallel.workers` to two or more to spawn persistent simulator processes and batch actor/reference encoding. Active workers are capped at
`groups_per_iteration * group_size`; `parallel.inference_batch_size` (default `32`)
caps each inference call. `parallel.threads_per_worker` (default `1`) bounds PyTorch intra-op
threads per simulator, and `parallel.timeout_seconds` (default `120`) sets response
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

## ONNX export

To publish the selected actor after imitation, IQL, PPO, GRPO, DPO, or a mixed
training plan, add this setting to the training YAML.

```yaml
run:
  export:
    prediction:
      onnx: {}
```

With the default model path, a single policy interface produces
`models/<model_name>.onnx` alongside the canonical PT model. Multiple bound
interfaces produce one graph per interface, named `<model_stem>-<interface>.onnx`.
The run manifest lists the published files. Select the graph under
`artifact.paths` in a `task: policy` inference or rollout config. Both
`action_selection: deterministic` and `action_selection: stochastic` use the
same evaluation-mode graph. PT and ONNX use different sampling procedures, so
the same seed can produce different trajectories across backends.

For direct ONNX Runtime use, supply model-encoded histories in the input order
and types described by `sequifier.execution_schema` in the graph metadata.
The additional Float64 input `policy_draws` has shape `[batch, 1 + 2 * actions]`.
For deterministic inference, fill it with zeros. For stochastic inference,
set column zero to one. For each action at zero-based index `i` in the binding's
declared order, supply an independent uniform draw in `[0, 1)` at column
`1 + 2*i` and a standard normal draw at column `2 + 2*i`. Sequifier's CLI
prepares these inputs automatically.

Output `action_i` is a zero-based index into the field's declared values for a
categorical action, or a real value in physical units. The graph also emits the
joint log probability, conditional categorical probability vectors, and real
latents and mixture components for replay. The model metadata contains the
binding, field order, units and input transforms.

## Checkpoints and inference

Policy runs use the shared `RUN_CHECKPOINT` envelope with a `POLICY_RECOVERY`
payload and a `POLICY_PHASE_STATE` record for the live phase. The embedded model owns
the execution configuration. Recovery records the active interface and verifies
the representation fingerprint on load. Checkpoints also retain plan, source,
partition and adapter identities, selected phase summaries and checksums, RNG and live phase
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
