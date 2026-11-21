# Infer Command Guide

For CPU numeric inference with externally calculated next-step inputs, see
[local forecast rollouts](../rollout.md). An explicit `rollout` configuration
selects a single seed window and saves a versioned trajectory through the
existing inference command.

Models with `static_conditioning` read the processed static feature table
named in the preprocessing metadata. Each inference window receives the
configured static values for its `sequenceId`.
Autoregressive generation retains these values while temporal inputs slide
forward.

Models with image ingestion need `input.image_paths` for the original image or video
folders selected by the model. Use a mapping from each temporal view name to its folder, even for one view. When the model has a preconditioning
image encoder, also set `input.preconditioning_image_paths` to a view-to-folder mapping, including a `default` key for an unnamed view. Preprocessing metadata records view
contracts and signatures, but omits folder paths. The folders must remain
available and unchanged after preprocessing. The inference loader applies the
same frame-to-`itemPosition` or time-column alignment and image transform used
during training. Timed sources also require the item-time index written by preprocessing.
PT and ONNX model execution accept image tensors and frame-valid masks; media
decoding stays outside the exported model.

```yaml
input:
  image_paths:
    front: data/front_images
    side: data/side_videos
  preconditioning_image_paths:
    front: data/sequence_front_images
    side: data/sequence_side_images
```

`sequifier infer` produces predictions, probabilities, or embeddings from a
PyTorch (`.pt`) or ONNX (`.onnx`) model.

```console
sequifier infer --config-path configs/infer.yaml
```

## Start here: ONNX

> **GPU ONNX inference requires `onnxruntime-gpu`.** If you set `run.execution.device: cuda`,
> install a build compatible with your CUDA and cuDNN versions. The CPU-only
> `onnxruntime` package does not provide `CUDAExecutionProvider`; setting
> `run.execution.device: cuda` alone does not enable GPU inference. See the
> [ONNX Runtime CUDA compatibility guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

PT is the default training export. Supervised runs may explicitly request
ONNX for deployment-oriented inference.
Exports embed their execution contract. Select a training route when providing
model configuration through a training config:

```yaml
task: forecast
run:
  project_root: .
  execution:
    device: cuda
artifact:
  paths:
  - models/event-model-best-5.onnx
input:
  path: data/events-test.parquet
  training_route:
    config_path: configs/train.yaml
    dataset: events
```

The route supplies columns, types, objective, window sizes, and preprocessing
metadata. Set `input.training_route.view` to read a named view, or supply `input.path` for fresh rows. Current ONNX exports
also embed this contract and metadata, so the route is optional. Older exports
need a training route or the full contract and metadata supplied explicitly.

All routes accept `output.path` for one selected model. It overrides the first
prediction or embedding file; later chunks append `-part-NNN` to its stem.
Prediction identity and probability-file layouts are described in
[Training artifacts](../training/artifacts.md#inference-and-collection-outputs).

## ONNX or PT?

| | ONNX | PT (default) |
| --- | --- | --- |
| Best fit | Portable, deployment-oriented inference. | Python/PyTorch workflows and easier configuration. |
| Runtime | ONNX Runtime on CPU or CUDA (with a CUDA-enabled ONNX Runtime installation). | PyTorch on CPU, CUDA, or MPS. |
| Configuration | Current exports embed their contract and metadata; older exports need a training route or explicit values. | Embeds its contract and metadata. |
| Behavior | Runs the exported graph; dropout requires a dropout-preserving export. | Retains PyTorch behavior and supports self-describing, multi-interface bundles. |

Benchmark the target workload rather than assuming either runtime is faster.
Choose PT when portability matters less than a compact, self-contained config:

```yaml
task: forecast
run:
  project_root: .
  execution:
    device: cuda
artifact:
  paths:
  - models/event-model-best-5.pt
input:
  path: data/events-test.parquet
```

For multi-interface PT, add `artifact.interface`. `task` is required for both formats: `forecast`, `policy`, or `embedding`.
`run.project_root` defaults to `.`, and `run.execution.batch_size` defaults
to `1`.

## Effective configuration and validation

Sequifier resolves inference configuration in this order:

1. Compose `additional_config_paths`, then apply CLI overrides.
2. Fill omitted model-contract fields from a selected training route and/or a
   self-describing PT artifact.
3. Load metadata from an explicit metadata path, selected dataset part, or PT
   artifact, in that order.
4. Apply defaults and validate the complete configuration.

Explicit values are assertions, not silent overrides. If an authored column,
type, objective, context, prediction length, interface, or metadata value
disagrees with its training config or portable artifact, inference stops and names the
conflicting field and source. Multiple model paths must share one contract.

Relative model, data, and metadata paths resolve under `project_root`.
`additional_config_paths` also resolve there; fragments cannot include further
fragments or define the same field twice.

## Fields

### Model, data, and routing

| Field | Default | Purpose |
| --- | --- | --- |
| `artifact.paths` | Required | Non-empty list of compatible model paths; policy requires exactly one. |
| `task` | Required | `forecast`, `policy`, or `embedding`. |
| `run.execution.device` | `cpu` | ONNX: `cpu` or `cuda`; PT also supports `mps`. ONNX CUDA requires an installed runtime exposing `CUDAExecutionProvider`. |
| `run.project_root` | `.` | Base for project paths; normally `.`. |
| `input.path` | Named test or sole split | Input file or folder. Usually required with artifact-only inference because artifacts do not store split paths. |
| `input.split` | Named test or sole split | Read a named split from `input.metadata_config_path`. |
| `input.metadata_config_path` | `null` | Explicit preprocessing metadata. |
| `input.image_paths` | `null` | View-to-folder mapping for every selected temporal view. Required when the model consumes temporal images. |
| `input.preconditioning_image_paths` | `null` | View-to-folder mapping for all preconditioning views. Required when the model uses preconditioning images. |
| `input.training_route.config_path` | `null` | Training config used to resolve a route. |
| `input.training_route.dataset` / `input.training_route.view` | `null` | Select a dataset and named view from the training config. |
| `artifact.interface` | Implicit when unique | Select a route from a training config or portable artifact. |
| `input.format` | `parquet` | `csv`, `parquet`, or folder-based `pt`. |
| `output.format` | `csv` | `csv` or `parquet`. |
| `run.execution.batch_size` | `1` | Sequences processed per batch. |

### Model contract

The following fields are optional when supplied by a PT artifact or training
route, and otherwise required as applicable: `assertions.input_columns`, `assertions.target_columns`,
`assertions.column_data_types`, `assertions.target_column_types`, `assertions.training_objective`,
`assertions.context_length`, `assertions.target_offset`, and `assertions.prediction_length`.

Preprocessing metadata may contain a Date/Datetime split column retained for
traceability. Inference ignores temporal metadata columns when `assertions.input_columns`
is omitted and rejects them if they are explicitly selected as model inputs or
targets. Other numeric columns from the same dataset remain usable normally.

`input.window_stride` optionally evaluates several model windows inside each
stored preprocessing row. `null` uses the legacy right-aligned view.
`assertions.prediction_alignment_steps` defaults to `1` and must match the value in
dataset metadata. For aligned splits, it sets the number of consecutive
prediction groups evaluated per stored window. Aligned inference requires
`input.window_stride: null` and a stored width of at least `context_length +
target_offset + (prediction_alignment_steps - 1) × prediction_length`.
With `task: embedding`, aligned splits label each embedding by its target
`itemPosition` and omit positions outside the split. For causal models, the
embedding represents the context used to predict that row. Other splits label
embeddings by the input activation position.

Folder inputs must carry `metadata.json` with matching storage and semantic
metadata, including category maps and normalization statistics. When using
metadata embedded in a portable artifact alone, inference can accept a wider
stored folder window if the remaining layout and semantics match.

### Output and runtime options

| Field | Default | Purpose |
| --- | --- | --- |
| `output.output_probabilities` | `false` | Write full distributions for categorical targets. Invalid for embeddings. |
| `output.decode_categories` | `true` | Decode categorical predictions to their original values. |
| `sample_from_distribution_columns` | `null` | Sample these categorical targets instead of using argmax. |
| `infer_with_dropout` | `false` | Enable dropout at inference; ONNX also requires dropout-preserving export. |
| `run.reproducibility.enforce_determinism` | `false` | Request deterministic PyTorch algorithms. |
| `run.reproducibility.seed` | `1010` | Random seed. |
| `generation_steps` | `null` | Positive step count. Setting it enables autoregressive forecasting. |

For an `autoregressive_transformer` decoder branch, same-position targets are
generated greedily in configured `assertions.target_columns` order. Returned categorical
probabilities are conditional on the preceding greedy predictions; they are not
independent marginals that can be resampled into a coherent joint result.
Consequently, `sample_from_distribution_columns` must not include a target from
an autoregressive transformer branch. External sampling remains available for
ordinary decoder branches.

Autoregressive inference requires a forward-looking generative model, prediction length
`1`, and identical input and target columns. For tabular CSV or Parquet input,
it begins at the first input window for each sequence and generates the same
number of steps for every sequence. Folder-based PT input is processed using
its stored windows rather than this first-window reduction.

## CLI overrides

`--data-path`, `--input-columns`, `--metadata-config-path`, `--model-paths`,
`--seed`, `--dataset`, `--view`, `--model-interface`, `--task`,
`--enforce-determinism`, and `--action-selection` override YAML values.
`--seed random` resolves a fresh integer seed once; the resolved configuration records it.
`--enforce-determinism true|false` explicitly overrides the YAML boolean. `--skip-metadata` skips external
metadata loading; the optional `input.inline_metadata` mapping, a selected training route, or a self-describing
PT artifact must then provide the required metadata. `--skip-metadata` is
available for forecast and embedding requests.

## Outputs

- Generative predictions: `outputs/predictions/<model>/part-NNN.<format>`.
- Categorical probabilities, when enabled:
  `outputs/probabilities/<model>/<target-column>/part-NNN.<format>`.
- Embeddings: `outputs/embeddings/<model>/part-NNN.<format>`.

`<model>` is the model artifact filename without its extension. Prediction and
embedding outputs contain sequence and window identifiers. Probability files
contain only the class-probability columns for their target. Categorical
predictions are decoded when `output.decode_categories` is enabled, and normalized
real predictions are restored to their original scale. Every input writes one
or more numbered parts.

## Portable depth models and dropout modes

New ONNX artifacts embed the selected interface, vocabulary/normalization
metadata, layouts, input name mapping, output descriptors, and fixed capacities.
They can resolve inference without the training YAML. Older flat ONNX files keep
the existing training-config/explicit-metadata fallback.

The `depth_transformer` ingestion branch is part of the saved model contract;
inference does not configure it again. Inference supplies the matching named
layout tensors and masks through the preprocessed PT input.

```yaml
infer_with_dropout: false
task: forecast
run:
  project_root: .
  execution:
    device: cpu
  reproducibility:
    seed: 1010
artifact:
  paths:
  - models/item-model-best-5.onnx
input:
  path: data/raw-items-split2
  format: pt
```

Use the corresponding embedding artifact with `task: embedding`, or change
`artifact.paths` to a portable PT model. `input.format: pt` describes preprocessed
input; it is independent of the model artifact extension. Deep inputs retain
`[B,T,D]` and explicit boolean depth masks through batching. New graphs accept
symbolic batch size with fixed temporal/depth capacities, including a partial
last batch. Legacy static graphs repeat features and masks together and trim
synthetic outputs. Empty input chunks do not invoke ONNX Runtime.

New ONNX graphs have a fixed dropout mode:

| `run.export.<artifact>.onnx.with_dropout` | `infer_with_dropout` | Behavior |
| --- | --- | --- |
| false | false | Evaluation graph; deterministic dropout behavior. |
| false | true | Error: export a stochastic graph. |
| true | true | Stochastic graph; runtime dropout remains enabled. |
| true | false | Error: export an evaluation graph. |

PT artifacts retain executable modules and can switch either way at inference.
Stochastic ONNX CLI inference seeds ORT once before session creation, then lets
successive calls advance the session RNG. Separate processes with the same seed,
provider, toolchain, graph, and call/batch ordering define the intended
repeatability boundary. Cross-provider identity and PT/ORT matching random draws
are not promised. The direct `Inferer` API never reseeds the process-global ORT
RNG. Legacy ONNX graphs lacking mode metadata cannot make this guarantee and
emit a warning when stochastic inference is requested.

Selected metadata must match layout names, selected deep membership, capacity,
position column/base, and gap policy. Input validation rejects missing masks,
incorrect ranks/capacities, forbidden gaps, illegal categories, and non-finite
values. Legal masked values are sanitized before narrowing and inside the graph.
The declared provider must be available; CUDA inference requires a CUDA-enabled
ORT installation. Outputs stay indexed by outer item coordinates.


## Policy model interfaces

`task` explicitly selects the operation. `task: policy` requires one
policy-bound portable PT artifact. `task: forecast` requests ordinary predictions,
including from a policy-bound interface with a supported forecast contract.
`task: embedding` extracts representations. Select `artifact.interface` when ambiguous.
The artifact validates task support; it never chooses the operation implicitly.
The CLI uses `--task forecast|policy|embedding`.

Policy input accepts raw observation rows or an indexed prediction view. History alignment, codecs and
normalization come from the artifact. Common fields cover model/interface/data
selection, read/write formats, device, seed, `enforce_determinism` and batch size.

| Policy field | Default / meaning |
| --- | --- |
| `output.path` | null; override the generated prediction destination |
| `action_selection` | `deterministic`; sequential mode/location choices, or `stochastic` sampling |
| `desired_return` | null; fixed budget for return-conditioned policies |
| `input.desired_return_column` | `desired_return`; per-row budget when no fixed budget is supplied |
| `input.previous_action_columns` | {}; recorded lagged-action column mappings for fragments |
| `input.sequence_id_column`, `input.item_position_column` | `sequenceId`, `itemPosition`; raw history ordering |
| `output.output_probabilities` | false; include joint log probability and write separate target-specific categorical probability files |
| `rollout` | null; execute in an online policy environment instead of writing prediction rows |

Optional assertions `input.metadata_config_path`, `assertions.input_columns`, `assertions.static_columns`,
`assertions.target_columns`, `assertions.column_data_types`, `assertions.target_column_types`, `assertions.training_objective`,
`assertions.context_length`, `assertions.target_offset`, `assertions.prediction_length` and `assertions.decoder_support_length` must
match the selected artifact. Unknown/cross-route options fail even at default values.
Policy rejects external per-column sampling, dropout, category decoding,
training-route selection, autoregression and `generation_steps`.
Forecast rejects policy-only output/return/action/history options; embeddings also
reject generative sampling, decoding, probabilities, autoregression and rollout.

Policy rollout declares `environment: {factory: module:factory, config: {},
units: {}}`, required `max_steps`, `seed_row_index` and
`trajectory_path`, plus optional `max_episode_steps`. It rejects forecast `bindings`.
Forecast rollout uses the forecast rollout adapter protocol, requires bindings and
selects `seed_window_index`. Neither accepts `rollout.type` or `seed_index`.
Both indices are zero-based and validated before environment creation. Trajectory
metadata uses the same route-specific index name. See the [rollout guide](../rollout.md) and
[policy template](../../src/sequifier/templates/infer-policy.yaml).

Execution determinism, action selection and seed are independent. A seeded
stochastic policy can reproduce its actions under the same supported runtime,
device and input ordering. Online evaluation uses the same action-selection
meaning and the training run's execution-determinism setting.

## Indexed splits and fresh raw data

Point `input.path` at an indexed split manifest to run inference for that split.
For example, point it at `data/events-indexed/2.split.json` for a third split. The
manifest resolves its sequence shards and preprocessing metadata.

To infer directly from fresh data in the original preprocessing input layout,
use the training metadata config and select the raw input layout.

```yaml
input:
  path: data/new-events.parquet
  layout: raw
  metadata_config_path: data/events-indexed/metadata.json
```

Alternatively run `sequifier preprocess` with `operation: prepare_inference` and
the same original metadata, then use the prediction manifest.

```yaml
input: {path: data/new-events-indexed/predict.split.json}
```

Both paths apply the same frozen maps, casting rules, and statistics. The
prepared manifest carries a snapshot of the original metadata.

For policy prediction, the direct and prepared paths use the same categorical
lookup and numeric normalization. An unseen category in an ordinary vocabulary
maps to `[other]` (ID 1). When preparing a policy view explicitly, list recorded
action columns used for lagged feedback in `passthrough_columns` to preserve
their original values. Raw policy inference keeps these columns automatically
when they are present. Older hashed actors can load the original categorical
settings through `load_policy(..., metadata_config_path=...)`.

For a folder of raw files, set `input.format` to their source format.
Fresh forward forecasts default to the right edge of each sequence. Set
`input.fresh_prediction_positions: observed` to select predictions within the supplied
data. Supply prior items when a new batch contains only a sequence fragment.
