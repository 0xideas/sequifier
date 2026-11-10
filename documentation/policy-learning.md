# Policy learning

For supervised checkpoint attachment, mixed real/categorical heads, full model
input schemas, and mixed supervised/policy plans, see the
[unified learning guide](supervised-policy-learning.md). The configuration below
documents the compatible single-categorical authoring surface.

`sequifier train` accepts one `SequifierConfig` with ordered
`training_plan.phases`. Supervised, imitation, IQL, PPO, DPO and GRPO use the same
model/interface and artifact contracts. A policy binding selects named real or
categorical action heads. See the [training guide](configs/train.md) for phase
ownership, scheduling budgets, optimizer groups and transition validation.
Policy execution currently uses a local CPU/CUDA/MPS process; standalone
supervised training also supports distributed execution.

## Authoring a run

Use the [policy template](../src/sequifier/templates/train-policy.yaml),
[PPO template](../src/sequifier/templates/train-ppo.yaml), or
[multiphase example](example_inputs/configs/train-multiphase-control.yaml).
The latter contains imitation → IQL → DPO → GRPO → PPO. Generate its data with:

```shell
python documentation/example_inputs/policy_control_data.py data
sequifier train --config-path documentation/example_inputs/configs/train-multiphase-control.yaml
```

All sources live in `dataset_training.<dataset>.parts.<part>`. Use `type: sequence`
for raw tables or exported sequence manifests, prepared metadata parts for
supervised data, and `type: preferences` with `manifest` for preference pairs.
Phase source references are objects such as `{source: decisions.default}`;
DPO `preferences` uses the same reference shape. Every data-bearing dataset
selects its model interface explicitly when more than one is available.

Run-wide `context_length`, normalization, validation fraction (0.2), episode
cache size (8), determinism and resume settings live in `global_training`.
`resume` defaults to `{policy: never, mode: reset_environments}`; use `if_exists`
or `required` with a checkpoint path, and `exact` for supported online adapters.
`policy_binding` applies to a single model interface; use `policy_bindings` keyed
by interface name for multiple interfaces. Environment units must match the binding.
No optimizer defaults live in this run-wide block.

Default outputs are `models/<model_name>/model.pt`,
`outputs/training/<model_name>/report.json`, and
`outputs/training/<model_name>/phases`. The resolved configuration records
role defaults and execution contracts. Fingerprints and checkpoints include
phase settings and validate source, reference and environment identities on resume.

## Selection and optimization

Every phase has a unique identifier `name` and `handoff: best | last` (default
`last`). Best selection retains the earliest finite best on ties. The final
update always evaluates. Missing finite evaluation for `best` fails explicitly.
`last` selects the most recently completed optimizer transaction. The final
export uses the last phase's selected actor, with metric, transaction and weight
hash recorded; metrics from different phases are never compared globally.

Each phase configures trainable roles under `optimization`. Imitation, DPO and
GRPO have `actor`; PPO has `actor` and `value`; IQL has `actor`, `value` and
`critic` (the twin Q networks share that optimizer). Omitted roles use defaults.
For example:

```yaml
optimization:
  actor: {learning_rate: 0.005, optimizer: {name: AdamW}, gradient_clip: 1.0}
```

Every role defaults to `learning_rate: 0.0003`, `optimizer: {name: Adam}`,
`scheduler: {name: ConstantLR, factor: 1.0, total_iters: 1}`, and
`gradient_clip: 1.0`. Unknown roles fail validation.

Imitation, DPO and GRPO share these batching and scheduling fields:

| Field | Default |
| --- | --- |
| `batch_size`, `accumulation_steps` | 32, 1; DPO counts pairs, GRPO counts whole trajectories |
| `evaluation_interval`, `checkpoint_interval` | 1; optimizer updates for imitation/DPO, completed collection/optimization updates for GRPO |

Optimizers support Adam, AdamW, SGD and RMSprop. Their constructor arguments
belong alongside `name` (or in `arguments`); `lr` and parameters belong to the
role configuration. Schedulers advance on completed optimizer steps; plateau/compound
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
Episode fitting and decision indexing are shared independently of imitation
batches, IQL transitions, and preference pairs.

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

The `optimization.actor`, `optimization.value` and `optimization.critic` roles
have independent optimizers and schedulers. All gradients are validated and
clipped once before any optimizer steps. Evaluation and checkpoint intervals
default to 100 updates. Critics are fresh even after another IQL phase and are
absent from the portable actor.

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
an independent weight snapshot before any phase update. Artifact references are
validated during preflight and reuse cached CPU weights; their files are checked
again before phase startup. References stay in evaluation mode with
gradients disabled. Full representations and actual weight hashes must match.
Embedded checkpoint reference weights survive removal of the original file;
a still-present artifact with changed weights is rejected on resume.

## Online adapters

PPO, GRPO, evaluation and policy rollout share the version-3 `OnlineEnvironmentAdapter` contract. Configure
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

Policy inference from supplied observation rows additionally requires
`EnvironmentCapabilities(seeded_reset=True)` and
`reset_from_observation(initial: OnlineResetResult, seed: int) -> OnlineResetResult`.
It must initialize the simulator without executing an action and return exactly
the supplied observation, static inputs and units. This is distinct from the
ordinary `reset(seed)` used during training. The bundled control adapter
implements both. Forecast rollout bindings retain their separate version-1
contract.

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
The `optimization.actor` and `optimization.value` roles have independent
optimizers/schedulers and the shared learning-rate and clipping defaults.
`allow_partial_rollout: true` executes exactly the remaining action budget;
false requires divisibility. `max_buffer_bytes` defaults to 536870912.
`archive_directory` optionally writes separate versioned collection records;
`archive_scores` defaults to false. Archived records are not training input.

`evaluation_interval` and `checkpoint_interval` default to one complete
collection/optimization iteration. Evaluation uses a separate adapter,
independent seeds and a whole-evaluation POSIX/main-thread deadline. It evaluates
the live actor synchronously and restores every module's training mode and the
training RNG state afterward. It requires
`environment`, `seeds`, `max_episode_steps`, and `timeout_seconds`;
`action_selection: deterministic | stochastic` defaults to deterministic. Default
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

Named sources share indexed episode storage across phases and partitions. Table
sources are parsed once into temporary sorted Parquet storage with direct episode
row-group access. The episode cache stays bounded; partition views retain metadata
and validated transition indices. Imitation shuffles episodes and then decisions
within each episode to preserve storage locality.

Inputs are resolved once into a run-owned catalog. Source resolution returns
verified file checksums, including the checksums checked against sequence manifests;
preparation reuses those resolved sources without resolving or hashing them again.
Every phase boundary explicitly rehashes all input files, manifests, metadata,
static data and preference files to detect modifications. Output and archive path
conflicts are checked separately before output mutation and at phase boundaries.
Prepared data supplies its fitting view and frozen-representation validation;
algorithm-specific transition checks remain with the IQL dataset.

The policy-training checkpoint is `sequifier_policy_run`, format version 6.
Version-5 checkpoints are read and upgraded on the next save.
Completed results determine the phase boundary; optional version-2 phase state
records the latest committed transaction. It contains the resolved plan, frozen
representation, source/partition and adapter identities, run RNG, completed
phase records pairing each result with its actor checksum and result fingerprint.
Actor/report paths come from the result provenance. Live phase state contains
role-keyed networks and optimizers/schedulers/step counts, sampler, selection, metrics and
algorithm-specific progress. DPO/GRPO references are embedded. Earlier selected
actors remain in their phase artifacts. Checkpoints older than version 5 are
not supported.

Writes use temporary files, flush them, and atomically replace their destinations.
At a handoff, the selected `actor.pt` is published with its result metadata, then
the run checkpoint commits its checksum and result. The checkpoint is the
authoritative phase boundary. An interrupted, uncommitted export is replaced on
retry; a committed handoff advances exactly once. Completed phases never rerun
or reselect their actor.

Retain phase actor artifacts alongside the checkpoint. Resume checks their
checksums and result identities before output mutation, without loading or
rewriting earlier models. Phase `report.json` files are derived from committed
results and regenerated on resume; they are not recovery dependencies. No
separate phase manifest is needed. Representation and data/adapter/reference
mismatches still fail before output mutation.

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

The Python API starts from the same authored schema as YAML:

```python
from sequifier.config.train_config import SequifierConfig
from sequifier.config.training_resolution import resolve_training_run
from sequifier.train import run_training
import yaml

with open("configs/train-policy.yaml") as stream:
    authored = SequifierConfig.model_validate(yaml.safe_load(stream))
resolved = resolve_training_run(authored)
result = run_training(resolved)
```

`run_training` selects the supervised engine, native policy runner, or mixed-plan
coordinator from the resolved phase algorithms. `load_train_config(path, {}, False)`
combines YAML loading and resolution. `run_policy_training` takes an internal
resolved policy run, rather than a separate authored configuration. Direct Python
inference can validate `PolicyInferenceConfig`, resolve it with
`policy_inference_config(request.model_dump(exclude_unset=True))`, and call
`sequifier.rl.inference.infer_policy`; it uses the same schema as the YAML loader.
`rl_api` exports phase schemas, policy bindings, action declarations, environment
capabilities, sequence records and `load_policy`.

Use `model_type: policy` with a selected portable PT artifact for raw CSV/Parquet
observation rows. `enforce_determinism` requests deterministic execution;
`action_selection: deterministic | stochastic` selects the policy mode or sampling.
The seed is independent of both. Seeded sampling can reproduce a sequence under
the same supported runtime, device and input order. Online evaluation uses
`action_selection`, defaulting to deterministic, with the run's execution settings.
Evaluation reports record that choice.

Policy `rollout.environment` declares the version-3 environment, `max_steps`
sets the collection budget, and `seed_index` selects a zero-based raw input row.
`trajectory_path` is required; `max_episode_steps` is optional. Policy rollout
rejects forecast bindings and generation settings. It requires seeded observation
reset; successors come from actual environment transitions. True termination may
omit a successor; truncation and collection cutoffs retain it. Forecast rollout
instead selects a prepared model window and follows the separate version-1
protocol. See [inference](configs/infer.md), [rollout](rollout.md), and the
[Python PPO workflow](example_inputs/ppo_toy_control.py).

CPU tests cover the reference execution contract. CUDA and MPS reproducibility
also depends on backend/operators and matching device/RNG topology.
