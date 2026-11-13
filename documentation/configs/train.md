# Train Command Guide

For trajectory-backed categorical BC, return-conditioned imitation, and offline
IQL and online PPO through the same commands, see [policy learning](../policy-learning.md).

`sequifier train` trains one shared transformer backbone through one or more
named model interfaces. An interface is an ingestion module, its generated
adapter, and a decoder. Phases own batching, learning, trainability and lifecycle controls; datasets own sources; `model` owns architecture and execution contracts. See the [ownership reference](ownership.md) for the complete field map and migration command.

```console
sequifier train --config-path configs/train.yaml
```

## Minimal named configuration

Use the same named structure for one or many interfaces, datasets, parts, and phases:

```yaml
model:
  backbone:
    architecture:
      model_dim: 128
      max_context_length: 512
      num_layers: 6
      attention:
        type: mha
        num_heads: 8
        num_kv_heads: 8
        output_projection: true
      feed_forward:
        activation: swiglu
        hidden_dim: 512
      normalization:
        type: rmsnorm
        norm_first: true
      position_encoding:
        type: rope
        theta: 10000
      dropout: 0.1
      shared_layer_groups: []
  interfaces:
    event_prediction:
      input_columns:
      - event
      target_columns:
      - event
      categorical_decoder_special_tokens:
        event:
        - other
      ingestion:
        type: embedding
        output_dim: 128
      decoder:
        type: linear
        prediction_length: 1
        support_length: 1
  execution:
    context_length: 128
    target_offset: 1
    training_objective: causal
dataset_training:
  events:
    model_interface: event_prediction
    parts:
      original:
        metadata_config_path: configs/metadata_configs/events.json
        read_format: parquet
      increment:
        metadata_config_path: configs/metadata_configs/events-increment.json
        read_format: parquet
    split:
      validation_fraction: 0.2
training_plan:
  phases:
  - name: incremental_finetuning
    epochs: 2
    mode: sequential
    sources:
    - source: events.increment
    batch_size: 64
    accumulation_batches: 4
    optimization:
      model:
        learning_rate: 0.0001
        optimizer:
          name: AdamW
          arguments:
            weight_decay: 0.01
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
        gradient_clip_norm: 1.0
    optimizer_state: reset
    evaluation:
      sources:
      - source: events
      cadence:
        epochs: 1
      batch_size: 64
      before_phase: true
      on_snapshot: true
    algorithm: supervised
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      window_stride: 1
      datasets:
        events:
          criterion:
            event: CrossEntropyLoss
          loss_weights:
            event: 1.0
    trainability:
      by_dataset:
        events:
          roles:
            model: &id001
              freeze:
              - component: backbone
                exclude_modules: []
                semantic_groups:
                - attention.qkv
  - name: complete_retraining
    epochs: 5
    mode: interleaved
    selection: round_robin
    sources:
    - source: events
      batches_per_selection: 4
    batch_size: 64
    accumulation_batches: 4
    optimization:
      model:
        learning_rate: 0.0001
        optimizer:
          name: AdamW
          arguments:
            weight_decay: 0.01
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
        gradient_clip_norm: 1.0
    optimizer_state: reset
    evaluation:
      sources:
      - source: events
      cadence:
        epochs: 1
      batch_size: 64
      before_phase: false
      on_snapshot: true
    algorithm: supervised
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      window_stride: 1
      datasets:
        events:
          criterion:
            event: CrossEntropyLoss
          loss_weights:
            event: 1.0
    trainability:
      by_dataset:
        events:
          roles:
            model: *id001
run:
  project_root: .
  model_name: event-model
  reproducibility:
    seed: 1010
    enforce_determinism: false
  execution:
    device: cuda
    distribution:
      strategy: local
      world_size: 1
      backend: nccl
    precision: {}
    data_loading: {}
    compilation:
      torch_compile: outer
  recovery: {}
  logging: {}
  outputs: {}
  export:
    generative_model: true
    embedding_model: false
    onnx: true
    pt: false
    with_dropout: false
    batch_size: 256
```

All configurations use `model.interfaces`, `dataset_training`, dataset `parts`,
`policy_bindings` where applicable, and `training_plan.phases`. A unique interface
and source may be inferred when omitted. See the [naming reference](naming.md).

To condition every model window on features from a separate per-sequence
table, preprocess with `static_data_path` and configure the interface:

```yaml
model:
  interfaces:
    default:
      input_columns:
      - event
      static_conditioning:
        columns:
        - account_type
        - account_age
        feature_embedding_dims:
          account_type: 16
          account_age: 8
      target_columns:
      - event
      ingestion:
        type: embedding
        output_dim: 128
      decoder:
        type: linear
        prediction_length: 1
        support_length: 1
```

The static columns are separate from temporal `input_columns` and targets.
They are looked up by `sequenceId` for every sampled window, including windows
from the middle of a sequence.
`feature_embedding_dims` assigns a width to each static feature; real features
with width greater than one use a learned projection. Alternatively,
`total_embedding_dim` distributes a total width automatically when all static
features are categorical or all are real. For mixed types, specify
`feature_embedding_dims`. If neither option is set, categorical widths use
the existing cardinality-based default and real features use one channel.
The concatenated features are projected to the backbone `model_dim`.
The model creates one prefix token at runtime
and removes its output before decoding. Set backbone `max_context_length` to
at least `model.execution.context_length + 1`.

## Canonical configuration

```yaml
model:
  backbone:
    architecture:
      model_dim: 128
      max_context_length: 512
      num_layers: 6
      attention:
        type: mha
        num_heads: 8
        num_kv_heads: 8
        output_projection: true
      feed_forward:
        activation: swiglu
        hidden_dim: 512
      normalization:
        type: rmsnorm
        norm_first: true
      position_encoding:
        type: rope
        theta: 10000
      dropout: 0.1
      shared_layer_groups: []
  interfaces:
    event_prediction:
      input_columns:
      - event
      target_columns:
      - event
      categorical_decoder_special_tokens:
        event:
        - other
      ingestion:
        type: embedding
        output_dim: 128
      decoder:
        type: linear
        prediction_length: 1
        support_length: 1
  execution:
    context_length: 128
    target_offset: 1
    training_objective: causal
dataset_training:
  events:
    model_interface: event_prediction
    parts:
      original:
        metadata_config_path: configs/metadata_configs/events.json
        read_format: parquet
      increment:
        metadata_config_path: configs/metadata_configs/events-increment.json
        read_format: parquet
    split:
      validation_fraction: 0.2
training_plan:
  phases:
  - name: incremental_finetuning
    epochs: 2
    mode: sequential
    sources:
    - source: events.increment
    batch_size: 64
    accumulation_batches: 4
    optimization:
      model:
        learning_rate: 0.0001
        optimizer:
          name: AdamW
          arguments:
            weight_decay: 0.01
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
        gradient_clip_norm: 1.0
    optimizer_state: reset
    evaluation:
      sources:
      - source: events
      cadence:
        epochs: 1
      batch_size: 64
      before_phase: true
      on_snapshot: true
    algorithm: supervised
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      window_stride: 1
      datasets:
        events:
          criterion:
            event: CrossEntropyLoss
          loss_weights:
            event: 1.0
    trainability:
      by_dataset:
        events:
          roles:
            model: &id001
              freeze:
              - component: backbone
                exclude_modules: []
                semantic_groups:
                - attention.qkv
  - name: complete_retraining
    epochs: 5
    mode: interleaved
    selection: round_robin
    sources:
    - source: events
      batches_per_selection: 4
    batch_size: 64
    accumulation_batches: 4
    optimization:
      model:
        learning_rate: 0.0001
        optimizer:
          name: AdamW
          arguments:
            weight_decay: 0.01
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
        gradient_clip_norm: 1.0
    optimizer_state: reset
    evaluation:
      sources:
      - source: events
      cadence:
        epochs: 1
      batch_size: 64
      before_phase: false
      on_snapshot: true
    algorithm: supervised
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      window_stride: 1
      datasets:
        events:
          criterion:
            event: CrossEntropyLoss
          loss_weights:
            event: 1.0
    trainability:
      by_dataset:
        events:
          roles:
            model: *id001
run:
  project_root: .
  model_name: event-model
  reproducibility:
    seed: 1010
    enforce_determinism: false
  execution:
    device: cuda
    distribution:
      strategy: local
      world_size: 1
      backend: nccl
    precision: {}
    data_loading: {}
    compilation:
      torch_compile: outer
  recovery: {}
  logging: {}
  outputs: {}
  export:
    generative_model: true
    embedding_model: false
    onnx: true
    pt: false
    with_dropout: false
    batch_size: 256
```


For a causal objective, `target_offset` sets the target position relative to
each input position. With `target_offset: 1`, every decoded position predicts
its next token. A decoder `prediction_length` greater than 1 selects several
consecutive positions for inference; it does not make one position predict
several future tokens as in LLM multi-token prediction. Training currently
computes loss over all valid decoded positions, regardless of
`prediction_length`.

For `model.execution.training_objective: next_occurrence`, declare the trigger
in each supervised phase's `learning` settings:

```yaml
learning:
  next_occurrence_config:
    column_name: event_type
    target_values: [purchase]
```

The trigger column must be a categorical target in every dataset used by that
phase, and its values must exist in that dataset's preprocessing metadata.
Different phases may select different triggers. These settings define training
and evaluation targets; they are not needed to execute an exported model.

## Ownership and resolution

- `run` owns execution resources, reproducibility, recovery, logging, outputs and export.
- `model.execution` owns immutable window/objective/attention facts; named interfaces own architecture and input contracts.
- `dataset_training` owns sources, interface references and split policy.
- Phases own losses, trainability, optimization, scheduling, evaluation, model selection, stopping and snapshots.
- Preprocessing metadata owns frozen storage, codec and normalization facts; explicit assertions must agree.

For categorical hard-label targets, set a target's criterion to
`AccurateCrossEntropyLoss` to preserve small loss values and target-logit
gradients at high confidence. `CrossEntropyLoss` retains the existing PyTorch
behavior. For example:

```yaml
training_plan:
  phases:
  - name: fit
    learning:
      datasets:
        events:
          criterion:
            event: AccurateCrossEntropyLoss
```

The choice is per target and is checked when resuming a run. Class
weights and `loss_weights` retain their usual meaning: the target class weight
multiplies each example's loss, then the target loss weight scales the masked
sum divided by the number of valid positions. The criterion also works with
hashed target heads, which divide the target loss weight among their heads;
class weights remain unsupported for hashed targets.

Every part of a dataset must resolve to the same schema, categorical semantics,
normalization contract, per-split window widths, alignment settings, and
file/folder storage form. Training validates the model view against each
selected training or evaluation split width. A source
named `events` iterates all parts in declaration order; `events.increment`
iterates only that part. Only parts selected by `evaluation.sources` require a
validation split.

### Categorical multi-hash embeddings and targets

Add `categorical_hashing` to a model interface, keyed by categorical column
name. The setting applies to an input column's embedding and, if the column is
also a target, to its prediction heads. For example, add these fields to a
training config with a backbone and dataset part:

```yaml
model:
  interfaces:
    default:
      input_columns:
      - accountId
      - merchantId
      target_columns:
      - accountId
      categorical_hashing:
        accountId:
          type: multi_hash
          num_buckets: 50000
          num_hashes: 4
          seed: 1010
        merchantId:
          type: qr
          num_buckets: 1000
      ingestion:
        type: embedding
        output_dim: 128
      decoder:
        type: linear
        prediction_length: 1
        support_length: 1
dataset_training:
  default:
    model_interface: default
    criterion:
      accountId: CrossEntropyLoss
    loss_weights:
      accountId: 1.0
```

Here `accountId` uses both hashed input embeddings and hashed target heads;
`merchantId` uses hashed input embeddings only. `multi_hash` creates
`num_hashes` independently seeded tables of `num_buckets` rows and adds their
outputs. `num_hashes` must be positive; `seed` defaults to 0. `qr` uses the
quotient and remainder of the category ID as two table indices and multiplies
their embeddings. Its hash count is always two. Do not add `hashing` under
`ingestion`: the embedding ingestion reads the interface setting, including
when used by `temporal_conv` with `base_ingestion: embedding`.

When `feature_embedding_dims` is omitted for a categorical-only embedding
ingestion, each column receives at least two dimensions. The remaining
`output_dim` positions are split in proportion to the logarithm of each
column's cardinality, rounding down first and assigning leftover positions
by largest fractional remainder (ties follow column order). For
`multi_hash`, the effective cardinality is the smaller of the original
cardinality and `num_buckets` per table. Other categorical columns use their
original cardinality. The output width must be at least twice the number of
categorical columns. Explicit `feature_embedding_dims` still controls each
column's width directly.

A hashed target requires `CrossEntropyLoss` or `AccurateCrossEntropyLoss`. Its
loss weight is divided equally across the hash heads; `class_weights` cannot be
used for that target. The
stored data still contains the original categorical IDs, without extra hash
columns. Hash codes for classes eligible for prediction must be unique; config
resolution reports a collision if the chosen bucket count and hash count do
not distinguish them.

In an `autoregressive_transformer` decoder branch, a hashed target must be the
last target in that branch's `target_columns` order. Earlier targets use a
full-size feedback embedding inside the decoder.

Folder dataset parts accept `file_order: shuffled` (the default) or
`file_order: name`. For curriculum training, these respectively reshuffle file
blocks each epoch or keep them in lexicographic path order; curriculum order is
local to each file. Without curriculum training, eager `shuffled` loading keeps
the ordinary global sample shuffle. The setting has no effect on single-file
parts.
`training_plan` also accepts `curriculum_training: false` (the default). When
set to `true`, `curriculum_column` must name one column available in the
preprocessed data. Training visits lower values first inside each physical file
and reshuffles equal-valued samples each epoch. Curriculum ordering applies only
to training loaders for parts selected by the training plan; validation keeps
its ordinary unshuffled order. Ordering is intentionally not synchronized
across files, loader workers, or distributed ranks.
`curriculum_column` defaults to `null` and cannot be set when curriculum
training is disabled. When disabled, curriculum values are ignored and the
ordinary sample shuffle is preserved.

```yaml
training_plan:
  phases:
  - epochs: 5
    name: train
    mode: sequential
    algorithm: supervised
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      datasets:
        default: {}
    curriculum:
      column: difficulty
dataset_training:
  default:
    parts:
      default:
        metadata_config_path: configs/metadata_configs/events.json
        file_order: name
        read_format: parquet
    split:
      validation_fraction: 0.2
run:
  project_root: .
  model_name: training
  reproducibility:
    seed: 1010
    enforce_determinism: false
  execution:
    device: cpu
    distribution:
      strategy: local
      world_size: 1
      backend: nccl
    precision: {}
    data_loading: {}
    compilation:
      torch_compile: none
  recovery: {}
  logging: {}
  outputs: {}
  export:
    batch_size: 1
```

`loss_weights` scales each target's contribution to the training and reported
aggregate loss. A weight of `0.0` disables that target's backward-loss
component while retaining its output and per-target accounting. At least one
target in each dataset must have a positive weight.

## Optimizer plans

The single-optimizer form applies one optimizer to all
parameters and uses the role’s `learning_rate`. To select parameters or
use multiple optimizers, replace it with an ordered plan. Each group has its
own learning rate; use each group’s `learning_rate` in this form:

```yaml
training_plan:
  phases:
  - name: supervised
    algorithm: supervised
    optimization:
      model:
        optimizer:
          groups:
          - id: hidden_matrices
            select:
              semantic_groups:
              - attention.*
              - feed_forward.*
              component: backbone
              parameter_kind: weight
              ndim: 2
            optimizer:
              name: Muon
              arguments:
                momentum: 0.95
            learning_rate: 0.02
          - id: remainder
            select: otherwise
            optimizer:
              name: AdamW
              arguments:
                weight_decay: 0.01
            learning_rate: 0.0003
```

Each supplied selector condition must match. `semantic_groups` matches the
parameter catalog's semantic names using shell-style `*` patterns; a parameter
matches the list if it matches any pattern. Other available conditions are
`component` (`ingestion`, `backbone`, or `decoder`), `parameter_kind` (`weight`,
`bias`, or `other`), and `ndim`. At least one condition is required. The final
group must be `otherwise`, which receives every parameter not selected above.
Routes must be disjoint, all groups must receive parameters, and each trainable
parameter is assigned once. Group IDs are available to integration directives.

The plan also supports different settings for one optimizer, for example two
AdamW groups with different weight decay. The named optimizer must be available
in Sequifier's optimizer registry. Muon availability depends on the installed
PyTorch version. The configured scheduler applies to every group; training
metrics report the first group's learning rate. With `OneCycleLR`, each group's
`learning_rate` is its peak rate. Omit `scheduler.max_lr` in plan mode;
Sequifier supplies the group rates in plan order. `OneCycleLR` derives the
starting rates from those peaks and `div_factor`.

With `CyclicLR`, each group's `learning_rate` is also its peak rate. Add a
`base_learning_rate` below that peak to every optimizer group, and omit
`scheduler.base_lr` and `scheduler.max_lr`. For the plan above, the two base
rates could be `0.002` and `0.00003`, respectively. Configure the scheduler
with `name: CyclicLR`, `step_size_up: 2000`, and `scheduler_step_on: optimizer_step`.
Momentum cycling defaults to off for both cycle schedulers in plan mode;
explicit `cycle_momentum: true` is unsupported. `base_learning_rate` is only
valid with `CyclicLR`.

## Canonical training settings

Batching, role-keyed optimization and trainability belong to phases. Evaluation
uses `evaluation.cadence: {epochs: N}` for supervised phases and
`{iterations: N}` for policy phases. Retained checkpoint cadence is
`checkpoints.cadence` with the same explicit units. Final progress always
evaluates when enabled and checkpoints.

Evaluation owns `before_phase`, `on_snapshot` and `batch_size`. For supervised
phases, an omitted or null evaluation `batch_size` inherits the phase training
batch size. An explicit positive value overrides it. Model selection
uses `model_selection.handoff: best | last`; stopping uses
`stopping: {patience: N, min_delta: 0, on_trigger: end_phase | end_plan}`.
Patience counts scheduled finite observations. In supervised training, initial
evaluation is diagnostic and cannot select best weights; snapshot evaluations can
select best weights without consuming patience. Supervised phases preserve best
weights across phase boundaries when their resolved evaluation contracts match.
The contract includes validation sources, model interfaces, loss definitions,
window views, effective evaluation batch size and precision. Optimizer settings,
freezing and training duration do not affect compatibility. BERT phases have
different evaluation-mask seeds and therefore start separate histories. Set
`model_selection.scope: phase` to explicitly start a new history; the default is
`compatible_phases`. Stopping patience remains phase-local.
Best handoff, stopping and plateau
scheduling require an evaluation monitor. See [ownership](ownership.md#phase-settings).

Frozen normalization is an interface input contract or inherited artifact fact.
Recovery policy is `run.recovery`; phase snapshots belong to `checkpoints`.

## Model prediction parameters and policy actions

Model interfaces own prediction parameters. Policy bindings select targets as
actions and define their support, transforms, conditional activity and scale use.
They never create, overwrite or remove model parameter declarations.

```yaml
model:
  interfaces:
    actor:
      prediction_distributions:
        throttle:
          scale:
            initial: 0.2
            trainable: true
            min: 0.0001
            max: 100
policy_bindings:
  actor:
    actions:
      throttle:
        type: real
        scale: model
        lower: 0
        upper: 1
```

A real action must explicitly choose `scale: model` or `scale: {fixed: 0.1}`.
The latter bypasses the model scale for that policy without deleting its learned
parameter. A fixed policy scale does not require a model distribution entry.
Categorical actions declare `values` and cannot configure real scales or bounds.
`GaussianNLLLoss` uses the model's Gaussian distribution and its scale; policy
bounds, point masses and activity conditions apply only to policy likelihoods.

When initializing from an artifact, a full `model` declaration may add prediction
distribution entries. The architecture and existing distribution declarations
must match; only newly declared scale parameters receive fresh initialization.

## Optimization across phases

Each phase owns `batch_size` (32 by default), `accumulation_batches` (1), and
`optimization.<role>`. Supervised phases have the single role `model`, meaning
the trainable model parameters. Imitation, DPO and GRPO have `actor`; PPO also
has `value`; IQL has `actor`, `value`, and `critic`. Unknown roles fail validation.
Omitted roles use Adam, learning rate 0.0003 and gradient clip 1.0. Set
`gradient_clip_norm: null` to disable clipping. Optimizer plans take their rates from
their groups. Supervised roles default to StepLR (`step_size: 1`, `gamma: 0.99`);
policy roles default to ConstantLR (`factor: 1`, `total_iters: 1`).

```yaml
training_plan:
  phases:
  - name: supervised
    algorithm: supervised
    epochs: 10
    mode: sequential
    sources:
    - source: observations.train
    batch_size: 64
    accumulation_batches: 1
    optimizer_state: reset
    optimization:
      model:
        optimizer:
          name: Adam
          arguments: {}
        learning_rate: 0.001
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
        gradient_clip_norm: 1.0
```

`optimizer_state: reset | continue` describes the transition **into** this phase;
its default is `reset`. Reset creates new optimizer, scheduler and scaler state.
Continue preserves these states from the immediately preceding phase and requires
that phase's `model_selection.handoff: last`, identical algorithm, role configurations, parameter
ownership and freeze selections. The first phase cannot continue. Supervised
continuation requires the same dataset ownership; policy continuation requires
the same interface. Mixed plans reject continuation because their artifact
handoffs do not preserve optimizer state. Policy OneCycleLR continuation is
also rejected. Native checkpoints restore the active lifecycle in either mode. Supervised
`model_selection.handoff: best` requires an evaluation monitor and selects the preceding phase’s
best weights before starting the next phase; its monitor state is phase-local.

Supervised `scheduler_step_on: epoch` advances once after a completed phase epoch;
ReduceLROnPlateau receives the configured evaluation monitor at that boundary
and requires the phase's `evaluation.monitor` and `evaluation.cadence: {epochs: 1}`. `optimizer_step` advances once per applied optimizer
update, after accumulation, rather than once per microbatch. Policy roles only
accept `scheduler_step_on: optimizer_step`, their default. Policy plateau and
compound schedulers are unsupported. Constructor arguments are validated before
training; the shared optimizer shape does not imply shared scheduler capabilities.

For epoch OneCycleLR, the resolved `total_steps` is the phase's epoch count, or
the sum of epochs in its contiguous continuation chain. An explicit value must
match. Batch OneCycleLR derives `total_steps`. An explicit value is an assertion. Preflight counts loader
batches using the same sequential/interleaved source scheduler, phase seeds and
accumulation rules as training, including partial updates flushed at dataset
boundaries. The budget sums over a contiguous continuation chain. This calculation
does not consume samples or advance loader RNG. Early stopping and explicitly
skipped updates may consume less than the authored maximum.
Policy OneCycleLR derives the maximum optimizer-update budget below. An explicit
`total_steps` must match it:

| Algorithm | Optimizer updates per role |
| --- | --- |
| Imitation | epochs × ceil(training examples / (batch_size × accumulation_batches)) |
| IQL / DPO | iterations |
| PPO | optimization_epochs × sum(ceil(rollout size / (batch_size × accumulation_batches))) |
| GRPO | optimization_epochs × sum(ceil(group trajectories / (batch_size × accumulation_batches))) |

PPO `rollout_env_steps` counts collected transitions; `batch_size` counts
transitions in each optimization microbatch. They are distinct budgets. IQL
uses `batch_size: 64` by default and one update per sampled batch; it does not
accept accumulation. DPO counts preference pairs; GRPO counts trajectories.
PPO defaults to `batch_size: 64`; other batched algorithms default to 32.
Early stopping, skipped updates or KL limits can consume less than the maximum.

The authored schema is separate from resolved training phases. Checkpoints and
plan fingerprints contain resolved roles, batching and scheduler budgets;
reproduction does not depend on global optimizer values. Run execution policy and
durable model execution facts have separate owners; see [ownership](ownership.md).

## Training plans

A sequential phase exhausts each source in listed order. An interleaved phase
uses `round_robin` or `weighted_random`. Each selection consumes at most
`batches_per_selection`; every source is still exhausted once per phase epoch.
Source `selection_weight` values affect order and burst frequency, not the amount of data consumed.

Gradient accumulation may cross part or source boundaries only while the
dataset stays the same. A dataset transition flushes a partial window using its
actual microbatch count. Dataset-specific frozen gradients are removed before
the optimizer step, preventing momentum and weight decay from changing frozen
parameters.

With several evaluation sources, configure an explicit monitor when
validation-based saving or early stopping is enabled:

```yaml
evaluation:
  sources:
  - source: events
  - source: telemetry.main
  monitor:
    source: events
    metric: loss
    direction: min
```

Phase `checkpoints.cadence` and snapshot triggers create retained checkpoints.
`run.recovery.latest_checkpoint_interval_minutes` refreshes rolling recovery.
`evaluation.on_snapshot` requests validation when a snapshot is due. Checkpoints
are written after flushing partial accumulation, or at a committed policy transaction.

## Composable YAML files

An entry file may declare complementary fragments with
`additional_config_paths`. Relative paths resolve against the entry file's
`project_root`. Fragments can contribute disjoint children under containers
such as `run`, `model.interfaces`, and
`dataset_training`; duplicate fields are rejected. CLI overrides are
applied after composition and before metadata resolution.

Fragments use the same named paths as the entry file. Components use only
`{name, arguments}`; constructor arguments must be nested under `arguments`.
Initialization methods use mappings such as `{method: zeros}`.

The training command accepts `--model-name`, `--seed`, and `--skip-metadata` as
configuration overrides. Dataset paths, columns, metadata paths, and device
selection must use their canonical YAML locations.

## Artifacts

ONNX is exported by default; PT inference bundles are opt-in. ONNX favors a
portable deployment runtime, while PT embeds its execution contract and retains
PyTorch behavior. See the "ONNX or PT?" section of the inference guide for the
trade-offs.

Single-dataset filenames use `<model>`, while multi-dataset logs, metrics, and
ONNX files use `<model>-<dataset>`. Part names are metric-row fields, not
filename components. PT inference bundles and exact-resume checkpoints remain
run-wide. Generated filenames do not use a `sequifier-` prefix.

The PT inference bundle contains `artifact_type`, `format_version`,
`model_state_dict`, an execution-only `model_config`, and export metadata
(trace-site names and provenance). Optimizers, paths, parts, training plans,
evaluation policy, and dataset bindings remain outside that bundle.
`export_with_dropout` affects ONNX export only: enabling it exports
the ONNX graph in training mode and disables constant folding so dropout remains
active.

## Image ingestion

Image views are declared during preprocessing and selected by name in an
ingestion branch. They can be the only temporal inputs (`input_columns: []`) or
can be merged with tabular branches. Each configured view is required for every
sequence. An image encoder returns one embedding per source image; its frame
pool reduces the images assigned to one `itemPosition` to one embedding.

```yaml
model:
  interfaces:
    prediction:
      input_columns:
      - temperature
      target_columns:
      - nextAction
      preconditioning_image_encoder:
        type: frozen
        model: resnet18
        weights: IMAGENET1K_V1
      ingestion:
        type: composite
        branches:
          measurements:
            type: embedding
            columns:
            - temperature
            output_dim: 32
          cameras:
            type: images
            views:
            - front
            - side
            output_dim: 128
            encoder:
              type: resnet
              depth: 18
            frame_pooling: attention
            view_merge: concat
        merge:
          type: concat
      decoder:
        type: linear
        prediction_length: 1
```

Bind the selected views to their original folders in each dataset part. The
preprocessing metadata deliberately omits folder paths. `image_paths` is always
a view-to-folder mapping, including runs with a single selected view.
Preconditioning images use `preconditioning_image_paths` in the same way:

```yaml
dataset_training:
  camera_data:
    model_interface: prediction
    parts:
      main:
        metadata_config_path: configs/metadata_configs/camera_data.json
        image_paths:
          front: data/front_images
          side: data/side_videos
        preconditioning_image_paths:
          front: data/sequence_front_images
          side: data/sequence_side_images
    criterion:
      nextAction: CrossEntropyLoss
```

Image view names and structured image cell names cannot be target columns;
the available decoder heads produce tabular targets. For BERT training, temporal
image frames at selected positions receive the configured masked, random, or
unchanged replacement. Masked frames become black and random frames receive
random pixel values; frame validity stays the same. Sequence preconditioning
images remain available as static context.

The trainable `resnet` supports depths 18, 34, 50, 101, and 152, or a custom
`block` (`basic` or `bottleneck`) with four `layers` depths. Bottleneck models
also support `groups`, `width_per_group`, and `replace_stride_with_dilation`.
Trainable ResNets use per-image group normalization so later frames cannot
affect earlier image embeddings during causal training.
The `frozen` encoder supports TorchVision `resnet18` and `resnet50` with
an explicit weight version. Its backbone stays frozen and in evaluation mode;
its output projection trains. The default image size and normalization match
the V1 ImageNet weight transforms. ResNet-50 V2 requires `resize_size: 232`
on its source; model resolution checks this. `frame_pooling` is `mean` (default) or
`attention`; `view_merge` is `concat` (default) or `mean`.

Image embeddings may also occupy cells in an existing structured layout. The
cell names are part of `feature_layout`; `image_cells` maps those names to image
view names. The usual structured processing blocks then combine image and
tabular cells:

```yaml
feature_layout:
  sensors:
    axes:
      source:
      - camera
      - measurement
    columns:
      camera_cell:
        source: camera
      temperature:
        source: measurement
ingestion:
  type: structured
  layout: sensors
  output_dim: 128
  image_cells:
    camera_cell: front
  image_encoder:
    type: frozen
    model: resnet50
    weights: IMAGENET1K_V1
  processing_blocks:
  - type: axis_pool
    axes:
    - source
    pooling: mean
```

The preconditioning source is independent of temporal views. One or more
preconditioning views are encoded and merged into a single sequence prefix
embedding. When scalar static conditioning is also set, both are fused into
that one prefix position.

## Depth encoders and nested composites

Use `read_format: pt` on every prepared part for preprocessed depth inputs. Layout
capacities come from dataset metadata. The following named model fragment
combines one shallow branch with a depth encoder; its metadata comes from the
preprocessing example in [preprocess.md](preprocess.md#named-depth-layouts).

At every outer time step, a `depth_transformer` branch embeds the valid child
slots from its named layout, applies a masked transformer across the child
dimension, and uses the CLS output as that collection's single temporal token.
The shared backbone then models relationships across outer time steps.

```yaml
model:
  backbone:
    initialization_seed: 20260910
    architecture:
      model_dim: 128
      max_context_length: 128
      num_layers: 4
      attention:
        type: mha
        num_heads: 8
      feed_forward:
        activation: swiglu
        hidden_dim: 512
      position_encoding:
        type: rope
  interfaces:
    default:
      input_columns:
      - accountType
      - subitemType
      - subitemAmount
      target_columns:
      - nextAction
      ingestion:
        type: composite
        branches:
          item:
            type: embedding
            columns:
            - accountType
            output_dim: 32
          subitems:
            type: depth_transformer
            layout: subitems
            columns:
            - subitemType
            - subitemAmount
            feature_embedding_dims:
              subitemType: 48
              subitemAmount: 16
            output_dim: 96
            dropout: 0.1
            initialization_seed: 12345
            initialization:
              attention.qkv:
                weight:
                  method: xavier_uniform
                  gain: 1.0
              free_parameter:
                weight:
                  method: normal
                  mean: 0.0
                  std: 0.02
            architecture:
              model_dim: 64
              num_layers: 2
              attention:
                type: gqa
                num_heads: 4
                num_kv_heads: 2
              feed_forward:
                activation: swiglu
                hidden_dim: 256
              normalization:
                type: rmsnorm
                norm_first: true
              position_encoding:
                type: learned
              dropout: 0.1
        merge:
          type: concat
      decoder:
        type: linear
        prediction_length: 1
```

For a `gated` or `attention` merge, `merge.softmax` accepts `standard`
(the default) or `max_excluding`. The latter uses the [max-excluding softmax](https://arxiv.org/abs/2610.31642)
to retain small gradients when one branch score dominates, at additional
compute cost. Set it independently for each composite, including nested ones.
`concat` and `sum` merges require `standard`.

Depth encoders support learned, sinusoidal, or rotary positions; MHA, MQA, or
GQA; LayerNorm or RMSNorm; the shared feed-forward activations and layer-sharing
configuration. `architecture.dropout` controls depth position and transformer
sites. The ingestion-level `dropout` controls the pooled output. Mixed
categorical/real features require explicit feature widths; homogeneous features
can divide `architecture.model_dim` using the ordinary ingestion width rules.

For categorical-only depth features, automatic allocation gives each column at
least two dimensions, so `architecture.model_dim` must be at least twice the
number of columns. To use smaller shares, set positive widths for every column
in `feature_embedding_dims`. For example, with two categorical columns,
`feature_embedding_dims: {first: 1, second: 1}` permits one dimension each.
Explicit widths may sum to a value different from `architecture.model_dim`;
the depth input projection handles the difference. The pooled projection handles
differences between `architecture.model_dim` and `output_dim`.

CLS occupies position zero, and physical slot `s` occupies position `s+1`.
Empty collections have a learnable CLS-only representation. Depth columns
cannot be targets because there is no depth decoder head. BERT training masks
valid child slots at selected outer time steps, using the configured masked,
random, or unchanged replacement; padded child slots stay untouched.
Autoregressive depth inference is excluded.

A composite branch may itself be a composite. Every nested composite requires
`output_dim`; an omitted root composite width retains the existing backbone
input width. Child widths resolve recursively before merge projections are
built. `allow_shared_columns` applies to the immediate children of the node
where it appears, including overlaps between their descendant features.
`allow_unused_input_columns` and `auxiliary_input_columns` belong at the root;
non-default child policies are rejected. Temporal per-feature positions occur
at leaf outputs only, and global temporal positions occur in the backbone.
Composite merges do not add another temporal position stage.

## Ordered autoregressive target decoding

Use `type: autoregressive_transformer` when outputs at the same temporal
position depend on one another. `target_columns` is an ordered contract: each
target is conditioned on the backbone state and all targets before it. Training
and validation use parallel teacher forcing; inference generates that prefix
greedily inside the decoder. Categorical and real targets can be mixed.

```yaml
decoder:
  type: autoregressive_transformer
  prediction_length: 1
  target_columns:
  - patch_length
  - byte_0
  - byte_1
  - magnitude
  architecture:
    model_dim: 256
    num_layers: 2
    attention:
      type: mha
      num_heads: 4
    feed_forward:
      activation: swiglu
      hidden_dim: 768
    normalization:
      type: rmsnorm
      norm_first: true
    position_encoding:
      type: rope
      theta: 10000
    dropout: 0.0
  shared_categorical_target_groups:
  - - byte_0
    - byte_1
  tie_input_output_embeddings: true
  support_length: 1
```

Shared groups must contain categorical targets with identical decoder-ID
mappings. Weight tying aliases each categorical input table with its output
projection. A composite decoder may mix this branch with ordinary linear or
MLP branches. For variable-length byte patches, predict length first and encode
unused slots with an explicit padding category.

For BERT objectives, autoregressive transformer targets must not include `mask`
in `categorical_decoder_special_tokens`. Inference excludes mask predictions,
which would invalidate the prefix used to generate later targets. The default
decoder vocabulary includes `other` and excludes `mask`. Configure an explicit
empty token list for a categorical target to exclude `other` as well.

Initialization overrides inherit per semantic group and per weight/bias target.
A child overrides only the targets it specifies; `preserve` keeps the constructed
value. Parameters are initialized once per identity. The ingestion adapter
inherits the root ingestion policy. Branch overrides that older versions ignored
now take effect; exact resume of those older runs is rejected.

`initialization_seed` accepts integers from zero through `2**63-1`, excluding
booleans. An explicit seed isolates constructor and custom-initializer randomness
from the run stream. Descendants derive seeds using SHA-256 over version 1,
namespace seed, relative branch path, and phase; explicit child seeds start a new
namespace. Omitted seeds preserve the existing RNG stream and initialization
traversal. Initialization seeds do not control runtime dropout. Dropout continues
using the run RNG and checkpointed per-rank state.

Trainability belongs to the phase for every algorithm. Both supervised and policy
roles use catalog selectors; supervised source-dependent updates use
`trainability.by_dataset.<dataset>.roles.model` with the same shape.

```yaml
training_plan:
  phases:
  - name: adapt
    trainability:
      roles:
        model:
          freeze:
          - {component: backbone, semantic_groups: [attention.qkv]}
          - {module: interfaces.default.ingestion.branches.subitems,
             exclude_semantic_groups: [free_parameter, ingestion.output_projection]}
```

Role overrides replace the phase policy for that dataset; absence inherits it.
`freeze: []` masks nothing and `freeze_except: []` masks everything. Selectors
are strict and shared with optimizer routing. Fixed model capabilities cannot be
unfrozen by a phase mask. Gradients still traverse masked modules; their gradients
are removed before AMP checks and stepping, preventing momentum/weight decay
updates. See [parameter selection](ownership.md#parameter-selection).


## Policy and mixed plans

All algorithms use `training_plan.phases`. Declare policy sequence/preference
parts in `dataset_training`, and attach actions with `policy_bindings` keyed by interface. Run-wide settings remain in
`run` and `model.execution`; no alternate run discriminator is needed.

See the complete [supervised template](../../src/sequifier/templates/train-supervised.yaml),
[policy template](../../src/sequifier/templates/train-policy.yaml), and
[supervised-to-policy template](../../src/sequifier/templates/train-supervised-to-policy.yaml).
The [policy guide](../policy-learning.md) describes algorithm budgets and
[the mixed-plan guide](../supervised-policy-learning.md) describes artifact handoffs.
