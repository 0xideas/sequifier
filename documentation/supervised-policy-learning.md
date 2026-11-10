# Supervised models and policy learning

Policy learning uses the same model interfaces, input tensors, decoder branches,
categorical codecs, and portable artifacts as supervised learning. A policy
binding identifies the output heads that form an environment action. The
binding does not automatically add inputs or replace a pretrained decoder.

## Post-train a supervised artifact

An ordinary supervised PT model, or a supported supervised run checkpoint, can
initialize a policy run. Its interface does not need existing policy metadata.

```yaml
project_root: .
model_name: posttrained
initialize_from: models/supervised/model.pt
policy_binding:
  actions:
    mode: {type: categorical, values: ["off", "on"]}
    price: {type: real, scale: 0.5}
dataset_training:
  demonstrations:
    parts:
      recorded: {type: sequence, paths: [data/demonstrations.parquet], format: table}
training_plan:
  phases:
    - name: imitate
      algorithm: policy_imitation
      sources: [{source: demonstrations.recorded}]
      epochs: 5
      batch_size: 64
      optimization:
        actor: {learning_rate: 0.0003}
```

Use the actual interface and target names from the source artifact. Quote YAML
labels such as `"off"` and `"on"` if the YAML parser interprets them as booleans.
Omitting categorical `values` selects the artifact's entire decoder vocabulary,
including any executable special tokens. Explicit values select a subset and
are reported as an adaptation when that changes the original support.

The artifact provides the architecture, context length, target positions,
normalization, and class maps. A separately supplied model must agree with it.
Original weights load strictly. Only explicitly introduced exploration parameters
are initialized. The run report records the transfer and its new/reused parameters.

## Real and categorical heads

Every real action head requires an explicit positive `scale`, measured in the
head's physical units. Its pretrained scalar output is the Gaussian location;
categorical heads retain their logits and persisted decoder-ID semantics.

| Setting | Meaning |
| --- | --- |
| `scale` | Fixed Gaussian standard deviation by default. |
| `learn_scale: true` | Learn a global log scale, bounded by `min_scale` and `max_scale`. |
| `lower` / `upper` | Transform the Gaussian onto a positive, negative, or finite open interval using a softplus or logistic bijection. |
| `atoms: {0.0: 0.05}` | Assign fixed probability mass to an exact real value, alongside the continuous component. Probabilities must sum to less than one. |
| `active_when: {mode: [on]}` | Activate a field only for these values of an earlier action field. |
| `inactive_value: 0.0` | Deterministic value used when a field is inactive. |

Bounds change the deterministic prediction function and are reported as an
adaptation. Samples are never clipped and then scored as unclipped Gaussians.
Collection retains transform latents for stable replay near numerical boundaries.
Externally supplied exact endpoints require a declared point mass. Inactive
fields are point masses with zero log-probability contribution, rather than
Gaussians with zero variance.

The existing decoder determines factorization. Ordinary branches predict
independent fields; an autoregressive branch samples in its existing target
order, embedding each sampled prefix value. Recorded actions are scored with
their recorded prefix. Selecting a later autoregressive target requires selecting
all its predecessors. Composite branches and tied weights are retained. Hashed
categorical outputs are normalized over canonical executable candidates.

Likelihoods sum over active action fields once per decision. Supervised loss
weights do not alter PPO ratios or DPO likelihoods. Independent categorical and
unbounded Gaussian entropy is exact; dependent/transformed entropy uses a
score-function estimator that includes the sampled prefix distribution. These
estimates can have higher variance. The GRPO sampled reference penalty retains
its existing interpretation.

All five algorithms consume this joint action contract: imitation, IQL, PPO,
DPO, and GRPO. IQL projects real action fields and embeds categorical fields
before evaluating its scalar twin Q-functions.

## Inputs, history, and output positions

The full persisted execution schema describes scalar, static, depth, image, and
mask tensors. Online adapters can return nested JSON values, or array/tensor
payloads that can be converted to portable values. Supply images with the
model's existing crop, channel, frame-capacity, and byte conventions. Depth
arrays and their named masks must match the source execution schema.

Prepared sources reuse depth masks and the existing image store. Table sources
can provide nested Parquet payloads. Use `columns.actions` to map logical action
names to table columns; `columns.previous_actions` maps preceding recorded actions.
Exported sequence shards preserve named action records and optional real latents.

No previous-action input is inserted by default. To feed an existing model input
from a lagged action, declare, for example:

```yaml
policy_binding:
  previous_action_inputs: {last_quantity: quantity}
  initial_inputs: {last_quantity: 0.0}
  actions:
    quantity: {type: real, scale: 0.5}
```

`initial_inputs` supplies the first decision's lagged value. The existing input's
codec and normalization are retained. Temporal padding, action activity, and
missing training labels are distinct. Complete relevant action tuples are
required for likelihood training; missing predecessors are not silently masked.

`decision_position` defaults to `-1`. Other negative positions select existing
decoder positions while preserving support and prediction geometry. A model
with several prediction positions does not automatically execute an action block.
Only the selected current decision is executed by these runners.

Forward attention is persisted separately from the learning objective. A
bidirectional encoder can consume an entirely observed history. Each historical
decision is reconstructed at its actual availability boundary, so later
observations cannot leak into its likelihood. An input unavailable at action time
must be supplied through an explicit task adaptation.

Return conditioning is opt-in for attached models with a `__policy_return`
input. Offline conditioning needs a known reward tail. Online conditioning needs
`initial_desired_return`; subsequent budgets follow the configured return update.

## Mixed training plans

Supervised and policy phases use the same `training_plan.phases` structure.
See the full [supervised-to-policy template](../src/sequifier/templates/train-supervised-to-policy.yaml).
Supervised prepared parts and policy sequence parts both belong to
`dataset_training`; each phase references the appropriate parts. No additional
run discriminator is needed. `global_training` owns execution and model/data
contracts. Each phase owns batching and `optimization.<role>`:

```yaml
training_plan:
  phases:
    - name: supervised
      algorithm: supervised
      epochs: 10
      mode: sequential
      sources: [{source: observations.train}]
      batch_size: 64
      accumulation_steps: 1
      optimization:
        model: {optimizer: {name: Adam}, learning_rate: 0.001}
    - name: imitation
      algorithm: policy_imitation
      epochs: 5
      sources: [{source: demonstrations.recorded}]
      batch_size: 64
      accumulation_steps: 1
      optimization:
        actor: {optimizer: {name: Adam}, learning_rate: 0.0003}
```

The coordinator preflights model, source and environment contracts and commits
incoming artifacts before native phase execution. Phase-local checkpoints recover
active work; atomic receipts recover completed handoffs. Policy `model_interface`
selects an interface when source references are insufficient to identify it.
All interfaces remain in the model artifact; a shared backbone update can change
other interfaces' predictions.

`optimizer_state` defaults to `reset`. Mixed-plan continuation is rejected because
artifact handoffs do not preserve optimization state. Native supervised-only and
compatible policy-only plans support explicit continuation under the
[training transition rules](configs/train.md#optimization-across-phases).
`handoff: last | best` selects phase weights; continuation requires the predecessor's
`last` handoff. Mixed plans execute locally. Exact online recovery requires the
adapter's snapshot capabilities.

The Python API exposes `SequifierConfig`, `ResolvedSequifierConfig`, and
`run_learning_plan` from `sequifier.training_api`. The common entry point
`sequifier.train.run_training` dispatches a resolved plan to its appropriate
runner. `load_policy(..., binding=...)` attaches a task directly to a supervised
artifact; policy bindings never author a second model architecture.

## Distribution-aware supervised losses and inference

Existing MSE, MAE, cross-entropy, and hashed-component losses retain their
semantics. An interface may declare `prediction_distributions` for real targets
and select `GaussianNLLLoss` in `dataset_training.criterion`. This uses the same
real distribution implementation as policy likelihoods and stores learned scales
in the model artifact.

Artifacts retain ordinary point-prediction execution. For an artifact containing
policy metadata, set `execution_mode: prediction` to select ordinary forecast
inference, or `execution_mode: policy` for action sampling. Joint policy
inference writes every executed field; `output_probabilities: true` also writes
the joint action log probability. Deterministic generation uses sequential
locations/argmax choices; it does not claim a global mode of an autoregressive
joint distribution.

New model artifacts use format version 3; versions 1 and 2 remain readable.
Joint trajectories and PPO collection records use version 2. Existing
single-categorical configuration and inference APIs remain available as
compatibility surfaces. ONNX remains a prediction export, not a general training
checkpoint or an automatically exported stochastic sampler.

This implementation has not been exercised with tests or training experiments
yet. Probability, architecture-parity, learning, and interruption acceptance
checks remain to be performed before release.
