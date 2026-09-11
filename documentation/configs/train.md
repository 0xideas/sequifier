# Train Command Guide

`sequifier train` trains one shared transformer backbone through one or more
named model interfaces. An interface is an ingestion module, its generated
adapter, and a decoder. Datasets own data and training policy; `model`
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

global_training:
  read_format: parquet
  training_objective: causal
  context_length: 128
  inference_batch_size: 256
  batch_size: 64
  learning_rate: 0.0001

model:
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
    ingestion: {type: embedding, output_dim: 128}
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
| `model.interface` | `model.interfaces.default` |
| `dataset.part` | `dataset_training.default.parts.default` |
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

global_training:
  read_format: parquet
  training_objective: causal
  context_length: 128
  target_offset: 1
  window_stride: 1
  inference_batch_size: 256
  batch_size: 64
  accumulation_steps: 4
  learning_rate: 0.0001
  optimizer: {name: AdamW, weight_decay: 0.01}
  scheduler: {name: StepLR, step_size: 1, gamma: 0.99}
  scheduler_step_on: epoch
  gradient_clip: 1.0
  save_interval_epochs: 1

model:
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
      ingestion: {type: embedding, output_dim: 128}
      decoder: {type: linear, prediction_length: 1, support: 1}

dataset_training:
  events:
    model_interface: event_prediction
    parts:
      original: {metadata_config_path: configs/metadata/events.json}
      increment: {metadata_config_path: configs/metadata/events-increment.json}
    criterion: {event: CrossEntropyLoss}
    loss_weights: {event: 1.0}
    freeze:
      backbone: {freeze: [attention.qkv]}

training_plan:
  phases:
    - name: incremental_finetuning
      epochs: 2
      mode: sequential
      sources: [{source: events.increment}]
    - name: complete_retraining
      epochs: 5
      mode: interleaved
      selection: round_robin
      sources: [{source: events, batches_per_selection: 4}]

evaluation:
  sources: [{source: events}]

export_generative_model: true
export_embedding_model: false
export_onnx: true
export_pt: false
export_with_dropout: false
```

The historical flat training schema is not accepted. In particular,
`training_spec`, top-level dataset paths/columns, `model.ingestion`,
`model.decoder`, and architecture-owned freezing are not canonical fields.

## Ownership and resolution

- `global_training` owns objective, window, optimizer, precision,
  distribution, compilation, checkpoint, and data-loader behavior. Phase
  entries own `epochs`.
- `model` contains exactly one backbone and one or more named interfaces.
  Different interface names create distinct ingestion and decoder weights;
  repeated references to one name share those weights.
- `dataset_training` owns parts, criterion/weights, class-share logging,
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
  sources: [{source: events}, {source: telemetry.main}]
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
such as `global_training`, `model.interfaces`, and
`dataset_training`; duplicate fields are rejected. CLI overrides are
applied after composition and before metadata resolution.

Singleton fragments may likewise contribute disjoint fields under
`model.interface` or `dataset`. Normalization occurs after fragment
composition, so all fragments in one training config must consistently use the
singular or named spelling at each level.

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

## Depth encoders and nested composites

Use `global_training.read_format: pt` for preprocessed depth inputs. Layout
capacities come from dataset metadata. The following singleton model fragment
combines one shallow branch with a depth encoder; its metadata comes from the
preprocessing example in [preprocess.md](preprocess.md#named-depth-layouts).

```yaml
model:
  backbone:
    initialization_seed: 20260910
    architecture:
      dim_model: 128
      max_context_length: 128
      num_layers: 4
      attention: {type: mha, n_heads: 8}
      feed_forward: {dim: 512, activation: swiglu}
      position_encoding: {type: rope}
  interface:
    input_columns: [accountType, subitemType, subitemAmount]
    target_columns: [nextAction]
    ingestion:
      type: composite
      branches:
        item:
          type: embedding
          columns: [accountType]
          output_dim: 32
        subitems:
          type: depth_transformer
          layout: subitems
          columns: [subitemType, subitemAmount]
          feature_embedding_dims: {subitemType: 48, subitemAmount: 16}
          output_dim: 96
          dropout: 0.1
          initialization_seed: 12345
          initialization:
            attention.qkv:
              weight: {method: xavier_uniform, gain: 1.0}
            free_parameter:
              weight: {method: normal, mean: 0.0, std: 0.02}
          architecture:
            dim_model: 64
            num_layers: 2
            attention: {type: gqa, n_heads: 4, n_kv_heads: 2}
            feed_forward: {dim: 256, activation: swiglu}
            normalization: {type: rmsnorm, norm_first: true}
            position_encoding: {type: learned}
            dropout: 0.1
      merge: {type: concat}
    decoder: {type: linear, prediction_length: 1}
```

Depth encoders support learned, sinusoidal, or rotary positions; MHA, MQA, or
GQA; LayerNorm or RMSNorm; the shared feed-forward activations and layer-sharing
configuration. `architecture.dropout` controls depth position and transformer
sites. The ingestion-level `dropout` controls the pooled output. Mixed
categorical/real features require explicit feature widths; homogeneous features
can divide `architecture.dim_model` using the ordinary ingestion width rules.
Input and pooled projections handle differing widths. CLS occupies position
zero, and physical slot `s` occupies position `s+1`. Empty collections have a
learnable CLS-only representation. Deep targets, deep BERT objectives, and deep
autoregressive inference are excluded.

A composite branch may itself be a composite. Every nested composite requires
`output_dim`; an omitted root composite width retains the existing backbone
input width. Child widths resolve recursively before merge projections are
built. `allow_shared_columns` applies to the immediate children of the node
where it appears, including overlaps between their descendant features.
`allow_unused_input_columns` and `auxiliary_input_columns` belong at the root;
non-default child policies are rejected. Temporal per-feature positions occur
at leaf outputs only, and global temporal positions occur in the backbone.
Composite merges do not add another temporal position stage.

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

Dataset freezing can follow the same tree:

```yaml
dataset:
  part: {metadata_config_path: configs/metadata_configs/raw-items.json}
  criterion: {nextAction: CrossEntropyLoss}
  freeze:
    backbone:
      freeze: [attention.qkv]
    ingestion:
      branches:
        subitems:
          freezing_except: [free_parameter, ingestion.output_projection]
        item:
          freeze: []
```

A local child selector replaces the inherited decision, including selective
unfreezing. `freeze: []` unfreezes the scope; `freezing_except: []` freezes it.
A node containing only `branches` delegates without changing inherited decisions.
Merge parameters retain their parent's policy; the adapter has its own dataset
selector. Unknown branches, policies below leaves, and contradictory alias
selections fail. Freezing leaves dropout active and permits gradients through
frozen modules. At optimizer boundaries, frozen gradients are removed **before**
AMP unscaling/overflow detection, so a frozen-only overflow cannot suppress a
healthy active update. Momentum and weight decay cannot update parameters whose
gradients are absent.
