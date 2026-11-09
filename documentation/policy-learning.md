# Categorical policy learning

`sequifier train` and `sequifier.rl_api.run_policy_training` execute the same
`PolicyRunConfig`. Its ordered `phases` may contain `policy_imitation`, `iql`,
`ppo`, `dpo`, and `grpo` in any compatible order. Each phase receives a deep copy
of its predecessor's selected actor. Optimizers, schedulers, critics, samplers,
and environments start fresh at phase boundaries.

This release supports one fixed categorical action, scalar or categorical
observations/static fields, causal fixed-length histories, FP32, and local
single-process CPU/CUDA/MPS execution. Return conditioning is available only
for imitation. A run containing another algorithm requires an unconditioned
actor throughout. Forecasting and policy phases cannot share a training plan.
Text tokenizers, continuous/joint actions, images/depth, distributed training,
vectorized environments, and reward models are outside this contract.

## Authoring a run

Start with the [multi-phase example](example_inputs/configs/train-multiphase-control.yaml),
[single-phase DPO](example_inputs/configs/train-dpo-control.yaml), or
[single-phase GRPO](example_inputs/configs/train-grpo-control.yaml).
Generate their curated data with:

```shell
python documentation/example_inputs/policy_control_data.py data
sequifier train --config-path documentation/example_inputs/configs/train-multiphase-control.yaml
```

The example defines imitation → IQL → DPO → GRPO → PPO. Remove or reorder phase
entries to author smaller runs, such as imitation → IQL, imitation → PPO,
imitation → DPO, imitation → GRPO, or DPO → GRPO. The input preferences come from
an explicitly supplied sign-matching rule; Sequifier never derives preferences
from recorded reward. These are illustrative configurations, without a claim
that every algorithm order improves every task.

```yaml
project_root: .
model_name: control
policy_interface: actor
context_length: 4
seed: 1010
device: cpu
# model uses the ordinary backbone/interfaces architecture schema.
model: { ... }
sources:
  decisions:
    paths: [data/control-decisions.csv]
    columns:
      episode_id: episodeId
      group: contextId
preferences:
  rankings: data/control-preferences.json
phases:
  - name: imitate
    algorithm: policy_imitation
    sources: [decisions]
    epochs: 4
  - name: rank
    algorithm: dpo
    preferences: rankings
    updates: 100
    beta: 0.1
    reference: {source: phase_start}
resume: {policy: if_exists, mode: exact}
```

Run-wide fields:

| Field | Meaning / default |
| --- | --- |
| `model`, `policy_interface`, `context_length`, `model_name`, `phases` | Required architecture, selected interface, history length, run name, ordered phases |
| `project_root`, `device`, `seed`, `enforce_determinism` | `.`, `cpu`, 1010, true; seed in `[0, 2**32)` |
| `sources` | Named `SequenceSourceConfig` objects; paths/manifest, format, explicit column mappings, prepared metadata |
| `preferences` | Named paths to version-1 preference-pair JSON manifests |
| `units` | Frozen observation/static units; `{}` means unitless; online training/evaluation adapters must match |
| `initialize_from` | Optional portable actor with matching full representation and architecture |
| `normalization` | Optional explicit frozen `source: identity` or `source: statistics` with exact real-field `statistics: {column: {mean: ..., std: ...}}` |
| `validation_fraction`, `episode_cache_size` | 0.2 and 8; split independent components before fitting codecs/statistics |
| `resume` | `{policy: never, mode: reset_environments}`; policy also `if_exists`/`required`, mode also `exact` |
| `model_path` | `models/<model_name>/actor.pt` |
| `report_path` | `outputs/training/<model_name>/report.json` |
| `checkpoint_path` | `checkpoints/<model_name>/run.pt` |
| `phase_directory` | `outputs/training/<model_name>/phases`; subdirectories `000-<phase_name>`, etc. |

The model must contain exactly `policy_interface`. Its input columns list the
policy observations followed by static fields; its target is the action column.
Use a scalar categorical decoder with `support: 1`, `prediction_length: 1`.
Declare `return_conditioned: false` for BC or any improvement algorithm. Standard
ingestion/backbone/decoder configuration remains available. Previous executable
actions enter the input as `__policy_previous_action`; its start token is
input-only. Return-conditioned imitation additionally uses `__policy_return`.

The first offline phase fits missing categorical vocabularies and normalization
on its training episodes only. A declared normalization takes precedence over
fitting statistics. An online-first run must supply frozen normalization and
categorical counts/codecs, or a compatible `initialize_from` actor. Later phases
reuse the complete frozen representation. They may optionally declare
`representation` and online `normalization` as assertions; any disagreement
fails with a field-specific diagnostic before training. An online phase after
an offline phase normally omits `normalization` to reuse the fitted values.

All input files, preference manifests, metadata, initial actors and references
are protected from output collisions. Source bytes, partitions, codecs, model,
units, reference weights, and adapter identities are checked on resume.

## Selection and optimization

Every phase has a unique identifier `name` and `handoff: best | last` (default
`best`). Best selection retains the earliest finite best on ties. The final
update always evaluates. Missing finite evaluation for `best` fails explicitly.
`last` selects the most recently completed optimizer transaction. The final
export uses the last phase's selected actor, with metric, transaction and weight
hash recorded; metrics from different phases are never compared globally.

Imitation, DPO and GRPO share these actor optimization fields:

| Field | Default |
| --- | --- |
| `batch_size`, `accumulation_steps` | 32, 1; DPO counts pairs, GRPO counts whole trajectories |
| `actor_learning_rate`, `actor_optimizer` | 0.0003, `{name: Adam}` |
| `actor_scheduler` | `{name: ConstantLR, factor: 1.0, total_iters: 1}` |
| `gradient_clip` | 1.0 |
| `evaluation_interval`, `checkpoint_interval` | 1; optimizer updates for imitation/DPO, completed collection/optimization updates for GRPO |

Optimizers support Adam, AdamW, SGD and RMSprop. Their constructor arguments
belong alongside `name` (or in `arguments`); `lr` and parameters belong to the
phase. Schedulers advance on completed optimizer steps; plateau/compound
schedulers are unsupported. `OneCycleLR.total_steps` must equal the authored
maximum optimizer-step budget. Accumulation weights by the actual number of
examples, including final partial batches. Network stochastic layers are disabled
while gradients remain enabled; likelihood ratios use reproducible scores.

## Offline sequences and imitation

Sources accept CSV/Parquet tables, JSON rollout logs, committed sequence
manifests, or scalar prepared histories. A row has current observations and its
executed action; optional rewards/successors describe the resulting transition.
`PolicyTableColumns` exposes explicit observation/static/action/reward/group/ID
mappings. Temporal positions must be contiguous, zero-based at episode starts.
Fragments require a known preceding action. Reward-free sequences are ordinary
`SequenceEpisode` objects with absent transition capabilities, rather than
fabricated reward-bearing trajectories.

`policy_imitation` uses `sources`, optional disjoint `validation_sources`, and
`epochs`. It minimizes action negative log likelihood at each valid causal
decision. The monitor is `validation_action_nll`, direction `min`. BC requires no
rewards. Return conditioning requires recorded rewards and either true task
termination or an explicit finite return horizon, using the declared discount
and scale. Unknown return tails are excluded. The input return label at decision
`t` includes reward `r_t`; the current action is never an input at that decision.

Episode/group/duplicate-content components remain wholly in training or
validation. Explicit validation sources undergo the same overlap checks.
Episodes load through bounded caches; windows are constructed on demand.

## Offline IQL

`iql` requires complete rewards, termination/truncation and bootstrap flags,
with genuine successors for bootstrapped decisions. Nonterminal transitions
bootstrap; true terminal transitions do not. Never substitute a reset
observation for a successor.

The phase has `updates` (required), `batch_size` (64), `gamma` (0.99),
`expectile` (0.7), `inverse_temperature` (3), `max_weight` (100), and
`target_update_rate` (0.005). Independent value and twin-Q networks use
expectile regression and one-step Bellman regression with Polyak target-Q
updates. Actor extraction minimizes action NLL weighted by
`min(exp(inverse_temperature * (min(target_Q1,target_Q2)-V)), max_weight)`.
Each update samples transitions uniformly with replacement. Held-out unweighted
`validation_action_nll` is the minimizing selection monitor; Bellman residual,
values, Q and weight diagnostics are reported separately.

`actor_`, `value_`, and `critic_learning_rate` default to 0.0003; each role has
its own optimizer/scheduler. `gradient_clip` defaults to 1.0, evaluation and
checkpoint intervals to 100 updates. Critics are fresh even after another IQL
phase and are absent from the portable actor.

## DPO preferences

The inspectable manifest is separate from its sequence sources:

```json
{
  "format": "sequifier.preference-pairs",
  "version": 1,
  "pairs": [{
    "pair_id": "pair-1",
    "context_id": "initial-condition-1",
    "chosen": {"source": "decisions", "episode_id": "good", "start": 0, "end": 8},
    "rejected": {"source": "decisions", "episode_id": "bad", "start": 0, "end": 6}
  }]
}
```

Span bounds index the loaded episode/fragment rows, zero-based, start-inclusive
and end-exclusive. Both sides need
at least one executable action. Unequal lengths are permitted. Canonical
observations, static fields, prior-action histories and valid masks must match
at the branch point. Each subsequent score uses that alternative's actual
observations and prior actions. Duplicate pair IDs, identical/overlapping
alternatives, unsupported labels, non-finite features and incompatible initial
contexts fail before updates.

Splitting joins shared context/group IDs, episode IDs and duplicate learning
content transitively; explicit validation manifests obey the same constraints.
At least two independent components are required for automatic splitting.
Source identities, exact spans, fingerprints and membership persist in the run
checkpoint. The manifest must state the preferences; reward is not interpreted
as a preference.

`dpo` requires `preferences` and `updates`; optional `validation_preferences`
selects an explicit validation manifest. `beta` defaults to 0.1 and must be
finite positive. `scoring_chunk_size` defaults to 64 causal windows. For the
summed FP32 action log likelihood `s`, it minimizes equally weighted pair losses:

```text
margin = (s_actor(chosen)-s_actor(rejected)) - (s_ref(chosen)-s_ref(rejected))
loss = -logsigmoid(beta * margin)
```

Padding and unselected context positions contribute zero. Losses are averaged
over pairs, without length normalization or an added supervised term. Reports
include lengths, margin and preference accuracy. Default selection minimizes
`validation_dpo_loss`. Reference scores are recomputed, avoiding stale caches.
A two-pass backward calculation uses the exact scalar margin derivative and
recomputes one chunk's differentiable likelihood at a time, bounding activation
memory without changing the summed sequence objective.

DPO and GRPO each declare `reference: {source: phase_start}` (default) or
`{source: artifact, path: models/reference/actor.pt}`. A phase-start reference is
a deep copy before any phase update. References stay in evaluation mode with
gradients disabled. Full representations and actual weight hashes must match.
Embedded checkpoint reference weights survive removal of the original file;
a still-present artifact with changed weights is rejected on resume.

## Online adapters

PPO and GRPO share the version-3 `OnlineEnvironmentAdapter` contract. Configure
`environment: {factory: module:factory, implementation_id: ..., config: {}, units: {}}`.
Factories must create fresh independent adapters for evaluation. An adapter
explicitly declares:

```python
from sequifier.rl_api import EnvironmentCapabilities, OnlineResetResult, OnlineStepResult

class Adapter:
    protocol_version = 3
    implementation_id = "my-simulator.v1"
    capabilities = EnvironmentCapabilities(
        exact_snapshot=True,
        group_start="scenario",
        reproducible_group_start=True,
        group_start_randomness="independent",
    )

    def reset(self, seed: int) -> OnlineResetResult: ...
    def step(self, action: dict) -> OnlineStepResult: ...
    def close(self) -> None: ...
    def snapshot_state(self) -> dict: ...
    def restore_state(self, state: dict) -> None: ...
    def create_group_start(self, seed: int) -> object: ...
    def reset_group(self, token: object, rollout_seed: int) -> OnlineResetResult: ...
```

Reset returns the current observation, static inputs and units. Step returns
immediate scalar reward, actual successor observation, boundary flags, units
and optional boolean success. Only true termination may omit the successor.
`close()` runs in `finally`. `implementation_id` must match the authored config
and change whenever simulator/reset/RNG semantics change.

GRPO requires `group_start: snapshot | scenario`, explicit methods, and an opaque
finite JSON-serializable token. Resetting a sample must reproduce the token's
initial condition while independently seeding future environment randomness.
`group_start_randomness` documents whether the token captures initial simulator
randomness or only the scenario. Group/sample indices and run/phase seeds derive
separate environment and action streams. Repeating a reset seed alone does not
establish this capability. Exact GRPO continuation additionally requires the
adapter's documented `reproducible_group_start` guarantee. The bundled
`CategoricalControlAdapter` demonstrates a deterministic scenario token.

`SeededAdapterBridge` can adapt an existing seeded forecast-rollout adapter with
an explicit initial-state factory; it declares no group-start or exact-snapshot
capability. Forecast rollout bindings keep their separate version-1 contract.

## Online categorical PPO

`ppo` requires `environment`, `evaluation`, `total_env_steps` and `rollout_length`.
Normalization is inherited from the run representation; a phase assertion is
optional. One persistent training environment collects compact sealed horizons.
True successors bootstrap fixed-length horizon/truncation boundaries; termination
disables bootstrap. FP32 GAE feeds the clipped policy objective, entropy bonus,
and separate value regression, with optional value clipping. Actor and value
optimizers share no parameters. Old collection scores are checked before update,
and collection generations are independent of phase/run indices.

Defaults: `optimization_epochs: 4`, `minibatch_size: 64`, `accumulation_steps: 1`,
`gamma: 0.99`, `gae_lambda: 0.95`, `clip_epsilon: 0.2`,
`value_coefficient: 0.5`, `entropy_coefficient: 0.01`,
`normalize_advantages: true`, `reward_scale: 1`. Optional positive `target_kl`
stops optimization against the old collection policy; `value_clip` is optional.
`actor_learning_rate`/`value_learning_rate` default to 0.0003, with independent
optimizers/schedulers and `actor_gradient_clip`/`value_gradient_clip` of 1.0.
`allow_partial_rollout: true` executes exactly the remaining action budget;
false requires divisibility. `max_buffer_bytes` defaults to 536870912.
`archive_directory` optionally writes separate versioned collection records;
`archive_scores` defaults to false. Archived records are not training input.

`evaluation_interval` and `checkpoint_interval` default to one complete
collection/optimization iteration. Evaluation uses a separate adapter,
independent seeds and a whole-evaluation POSIX/main-thread deadline. It requires
`environment`, `seeds`, `max_episode_steps`, and `timeout_seconds`;
`selection: deterministic | stochastic` defaults to deterministic. Default
`metric: mean_completed_return`, `direction: max` selects true completed returns.
`mean_completed_length` and `success_rate` are also supported. Censored episodes
and unavailable metrics are explicit; best selection fails when no finite
monitored value is available. The final iteration always evaluates.

## GRPO groups and objective

`grpo` requires `environment`, `evaluation`, `total_groups` and `max_steps`.
Defaults: `group_size: 4` (at least 2), `groups_per_update: 1`,
`optimization_epochs: 4`, `advantage_epsilon: 1e-6`, `clip_epsilon: 0.2`,
`kl_coefficient: 0.04`, `reward_scale: 1`, `scoring_chunk_size: 64`.
`old_policy_target_kl` optionally limits old-policy drift independently of the
reference penalty. No value network, value optimizer, or GAE is allocated.

Each group contains independent sequential trajectories from the same opaque
start token under one frozen old actor generation. All groups in an iteration
are sealed before any update. Scores sum observed rewards times the fixed
reward scale, through termination, truncation or the explicit `max_steps`
finite-horizon cutoff. Cutoffs do not invent termination or bootstrapped rewards.
Incomplete/failed/non-finite groups are discarded and never optimized. Errors
report issued environment actions, including actions discarded with the group;
there is no retry. Optional `max_env_steps` is checked before collection using
the conservative `group_size * max_steps` bound for every requested group.
Memory is bounded by `max_buffer_bytes` (default 536870912), checked before
collection and as frames are appended. Optional archives are separate from the
training buffer.

The precise first variant is:

```text
A_i = (R_i - mean(R)) / (std_population(R) + advantage_epsilon)
r_it = exp(log pi_actor(a_it|h_it) - log pi_old(a_it|h_it))
d_it = log pi_ref(a_it|h_it) - log pi_actor(a_it|h_it)
K_it = exp(d_it) - d_it - 1
loss = mean_trajectories(mean_valid_decisions(
    -min(r_it*A_i, clamp(r_it,1-epsilon,1+epsilon)*A_i) + kl_coefficient*K_it))
```

Each trajectory has equal weight regardless of length; group sizes are fixed.
Minibatches contain whole trajectories; accumulation uses actual trajectory
counts. FP32 ratios/objectives/gradients are checked before applying an update.
Equal group scores give zero advantages; the reference penalty can still update
the actor. Reference and old-policy scores stay detached across optimization
epochs. The reported `sampled_reference_penalty` is a sampled estimator under
old-policy samples, not exact current-policy KL.

Evaluation requires `metric: score_mean`, `direction: max` and scores every
independent evaluation episode through its real boundary or declared horizon.
The evaluation score uses the same fixed reward scale. Evaluation/checkpoint
intervals count completed updates, with an explicit final evaluation. Reports
include score spread, zero-variance groups, lengths, cutoff reasons, start
tokens, sampled penalty, clip fraction, generations, optimizer steps and issued
environment actions. Sealed buffers are consumed once; stale generations fail.

## Checkpoints, resume, and artifacts

The only policy-training checkpoint is `sequifier_policy_run`, format version 1.
It contains the resolved plan/fingerprint, immutable representation, data and
adapter identities, run RNG, current phase/index/status, completed results and
selected actors, algorithm/version and phase-owned state, and the last committed
transaction. DPO/GRPO reference weights are embedded. Optimizer, scheduler,
sampler, selection, evaluation and phase counters remain phase-local.

Writes use a temporary file and atomic replacement at completed updates or
handoffs. No partial group is checkpointable. A committed handoff advances the
phase index exactly once; resume never reruns completed phases or reselects their
actor. Phase artifacts/reports can be republished idempotently from the stored
selection. Representation and data/adapter/reference mismatches fail before
output mutation.

`resume.mode: exact` requires validated PPO simulator/RNG snapshots or GRPO's
reproducible group-start capability and matching implementation identity. PPO
checks the restored snapshot exactly; GRPO derives subsequent group tokens and
independent streams from persisted counters. `reset_environments` explicitly
resets PPO environments or restarts GRPO with a new derived stream, preserving
completed updates and reporting a trajectory discontinuity. Neither mode reuses
partial groups or old on-policy samples. Offline continuation restores its
sampler/order and optimizer transaction exactly.

Portable artifacts are ordinary actor-only `sequifier_model` bundles. They
carry normalization, codecs, architecture, policy and unit provenance, with no
critics, references or trainer dependency. The final artifact comes from the
final phase's selected actor. The run report records order/config fingerprints,
source/reference provenance, selections, clocks and continuation modes.

## Python and inference

```python
from sequifier.config.train_config import load_train_config
from sequifier.rl_api import PolicyRunConfig, run_policy_training, load_policy

config = load_train_config("configs/train-policy.yaml", {}, False)
if not isinstance(config, PolicyRunConfig):
    raise TypeError("expected a policy run")
result = run_policy_training(config)
actor = load_policy(result.model_path)
for phase in result.phases:
    print(phase.name, phase.selected_metric, phase.clocks)
```

`PolicyRunConfig.model_validate` accepts the same mapping directly. The single
`PolicyRunResult` exposes `policy`, ordered `phases`, `representation` and output
paths. `rl_api` also exports the phase schemas, preference records, online
capabilities, `PolicyWindowView`, and model loading/execution helpers.

Use `sequifier infer` with the final artifact for categorical sequence inference
or rollout. `PolicyExecutor` and `collect_policy_rollout` build the same causal
histories; return-conditioned actors require an explicit desired return.
See [policy inference](configs/infer.md), [rollout](rollout.md), and the
[Python PPO workflow](example_inputs/ppo_toy_control.py).

CPU determinism is the reference execution contract. CUDA and MPS reproducibility
is limited by the selected PyTorch backend/operators and matching device/RNG
topology. This implementation was statically checked; no tests or training runs
were written/executed for this change, so numerical and end-to-end acceptance
verification remains outstanding.
