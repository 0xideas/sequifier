# Train Command Guide

`sequifier train` trains one shared transformer backbone through one or more
named model interfaces. An interface is an ingestion module, its generated
adapter, and a decoder. Datasets own data and training policy; `model_spec`
owns architecture.

```console
sequifier train --config-path configs/train.yaml
```

## Singleton configuration

When a run has one model interface, one dataset, one part, and one phase, those
values can be authored directly without routing names or references:

```yaml
project_root: .
model_name: event-model
device: cuda
seed: 1010

global_training_spec:
  read_format: parquet
  training_objective: causal
  context_length: 128
  inference_batch_size: 256
  batch_size: 64
  learning_rate: 0.0001

model_spec:
  backbone:
    architecture:
      dim_model: 128
      max_context_length: 512
      num_layers: 6
      attention: {type: mha, n_heads: 8, n_kv_heads: 8, output_projection: true}
      feed_forward: {dim: 512, activation: swiglu}
      normalization: {type: rmsnorm, norm_first: true}
      position_encoding: {type: rope, theta: 10000}
      dropout: 0.1
      shared_layer_groups: []
  interface:
    input_columns: [event]
    target_columns: [event]
    ingestion: {type: direct_embed, output_dim: 128}
    decoder: {type: linear, prediction_length: 1, support: 1}

dataset:
  part: {metadata_config_path: configs/metadata/events.json}
  criterion: {event: CrossEntropyLoss}

training_plan:
  epochs: 5

evaluation: true
```

The concise form expands before validation:

| Authored field | Canonical form |
| --- | --- |
| `model_spec.interface` | `model_spec.interfaces.default` |
| `dataset.part` | `dataset_training_spec.default.parts.default` |
| `training_plan.epochs` | One sequential phase named `train` |
| `evaluation: true` | Evaluate the inferred single source |

These shortcuts can be used independently. A unique interface and training
source are inferred; multiple interfaces or datasets require explicit names and
references. Singular and plural spellings cannot be combined. Because errors
are reported after expansion, they may use the canonical paths above.

## Canonical configuration

```yaml
project_root: .
model_name: event-model
device: cuda
seed: 1010

global_training_spec:
  read_format: parquet
  training_objective: causal
  context_length: 128
  target_offset: 1
  model_window_stride: 1
  inference_batch_size: 256
  batch_size: 64
  accumulation_steps: 4
  learning_rate: 0.0001
  optimizer: {name: AdamW, weight_decay: 0.01}
  scheduler: {name: StepLR, step_size: 1, gamma: 0.99}
  scheduler_step_on: epoch
  gradient_clip: 1.0
  save_interval_epochs: 1

model_spec:
  backbone:
    architecture:
      dim_model: 128
      max_context_length: 512
      num_layers: 6
      attention: {type: mha, n_heads: 8, n_kv_heads: 8, output_projection: true}
      feed_forward: {dim: 512, activation: swiglu}
      normalization: {type: rmsnorm, norm_first: true}
      position_encoding: {type: rope, theta: 10000}
      dropout: 0.1
      shared_layer_groups: []
  interfaces:
    event_prediction:
      input_columns: [event]
      target_columns: [event]
      categorical_decoder_special_tokens: {event: [other]}
      ingestion: {type: direct_embed, output_dim: 128}
      decoder: {type: linear, prediction_length: 1, support: 1}

dataset_training_spec:
  events:
    model_interface: event_prediction
    parts:
      original: {metadata_config_path: configs/metadata/events.json}
      increment: {metadata_config_path: configs/metadata/events-increment.json}
    criterion: {event: CrossEntropyLoss}
    loss_weights: {event: 1.0}
    freezing:
      backbone: {freezing: [attention.qkv]}

training_plan:
  phases:
    - name: incremental_finetuning
      epochs: 2
      mode: sequential
      sources: [{ref: events.increment}]
    - name: complete_retraining
      epochs: 5
      mode: interleaved
      selection: round_robin
      sources: [{ref: events, batches_per_selection: 4}]

evaluation:
  sources: [{ref: events}]

export_generative_model: true
export_embedding_model: false
export_onnx: true
export_pt: false
export_with_dropout: false
```

The historical flat training schema is not accepted. In particular,
`training_spec`, top-level dataset paths/columns, `model_spec.ingestion`,
`model_spec.decoder`, and architecture-owned freezing are not canonical fields.

## Ownership and resolution

- `global_training_spec` owns objective, window, optimizer, precision,
  distribution, compilation, checkpoint, and data-loader behavior. Phase
  entries own `epochs`.
- `model_spec` contains exactly one backbone and one or more named interfaces.
  Different interface names create distinct ingestion and decoder weights;
  repeated references to one name share those weights.
- `dataset_training_spec` owns parts, criterion/weights, class-share logging,
  freezing, and the interface reference.
- Preprocessing metadata owns split paths, data types, class counts, ID maps,
  special-token IDs, normalization facts, and stored-window layout.

Every part of a dataset must resolve to the same schema, categorical semantics,
normalization contract, storage layout, and file/folder storage form. A source
named `events` iterates all parts in declaration order; `events.increment`
iterates only that part. Only parts selected by `evaluation.sources` require a
validation split.

## Training plans

A sequential phase exhausts each source in listed order. An interleaved phase
uses `round_robin` or `weighted_random`. Each selection consumes at most
`batches_per_selection`; every source is still exhausted once per phase epoch.
Weights affect order and burst frequency, not the amount of data consumed.

Gradient accumulation may cross part or source boundaries only while the
dataset stays the same. A dataset transition flushes a partial window using its
actual microbatch count. Dataset-specific frozen gradients are removed before
the optimizer step, preventing momentum and weight decay from changing frozen
parameters.

With several evaluation sources, configure an explicit monitor when
validation-based saving or early stopping is enabled:

```yaml
evaluation:
  sources: [{ref: events}, {ref: telemetry.main}]
  monitor: {source: events, metric: loss, mode: min}
```

Checkpoint interval fields retain distinct behavior. `save_interval_epochs`,
`save_interval_batches`, and `save_interval_minutes` create persistent
epoch/batch snapshots and refresh the rolling `latest` checkpoint;
`save_latest_interval_minutes` refreshes only `latest`.
`save_interval_val_loss` controls whether timed or batch snapshots also run
validation and record its monitored loss. Every checkpoint is written after
flushing any partial gradient-accumulation window.

## Composable YAML files

An entry file may declare complementary fragments with
`additional_config_paths`. Relative paths resolve against the entry file's
`project_root`. Fragments can contribute disjoint children under containers
such as `global_training_spec`, `model_spec.interfaces`, and
`dataset_training_spec`; duplicate fields are rejected. CLI overrides are
applied after composition and before metadata resolution.

Singleton fragments may likewise contribute disjoint fields under
`model_spec.interface` or `dataset`. Normalization occurs after fragment
composition, so all fragments in one training config must consistently use the
singular or named spelling at each level.

The training command accepts `--model-name`, `--seed`, and `--skip-metadata` as
configuration overrides. Dataset paths, columns, metadata paths, and device
selection must use their canonical YAML locations.

## Artifacts

ONNX is exported by default; PT inference bundles are opt-in. ONNX favors a
portable deployment runtime, while PT embeds its execution contract and retains
PyTorch behavior. See the [inference trade-offs](./infer.md#onnx-or-pt).

Single-dataset filenames use `<model>`, while multi-dataset logs, metrics, and
ONNX files use `<model>-<dataset>`. Part names are metric-row fields, not
filename components. PT inference bundles and exact-resume checkpoints remain
run-wide. Generated filenames do not use a `sequifier-` prefix.

The PT inference bundle contains exactly `artifact_type`, `format_version`,
`model_state_dict`, and an execution-only `model_config`. Optimizers, paths,
parts, training plans, evaluation policy, and dataset bindings remain outside
that bundle. `export_with_dropout` affects ONNX export only: enabling it exports
the ONNX graph in training mode and disables constant folding so dropout remains
active.
