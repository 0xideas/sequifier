# Infer Command Guide

`sequifier infer` produces predictions, probabilities, or embeddings from a
PyTorch (`.pt`) or ONNX (`.onnx`) model.

```console
sequifier infer --config-path configs/infer.yaml
```

## Start here: ONNX

ONNX is the default training export and the deployment-oriented inference path.
Select its training route so Sequifier can recover the contract that the ONNX
file does not contain:

```yaml
model_path: models/event-model-best-5.onnx
training_config_path: configs/train.yaml
dataset: events
data_path: data/events-test.parquet
model_type: generative
device: cuda
```

The route supplies columns, types, objective, window sizes, and preprocessing
metadata. Add `part` when the dataset has several parts. You may instead provide
the full model contract and metadata explicitly.

## ONNX or PT?

| | ONNX (default) | PT |
| --- | --- | --- |
| Best fit | Portable, deployment-oriented inference. | Python/PyTorch workflows and easier configuration. |
| Runtime | ONNX Runtime on CPU or CUDA. | PyTorch on CPU, CUDA, or MPS. |
| Configuration | Needs a training route or explicit contract and metadata. | Embeds its contract and metadata. |
| Behavior | Runs the exported graph; dropout requires a dropout-preserving export. | Retains PyTorch behavior and supports self-describing, multi-interface bundles. |

Benchmark the target workload rather than assuming either runtime is faster.
Choose PT when portability matters less than a compact, self-contained config:

```yaml
model_path: models/event-model-best-5.pt
data_path: data/events-test.parquet
model_type: generative
device: cuda
```

For multi-interface PT, add `model_interface`. `model_type` stays explicit for
both formats because inference may generate outputs or extract embeddings.
`project_root` defaults to `.`, and `inference_batch_size` defaults to `1`.

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
disagrees with its training config or PT artifact, inference stops and names the
conflicting field and source. Multiple model paths must share one contract.

Relative model, data, and metadata paths resolve under `project_root`.
`additional_config_paths` also resolve there; fragments cannot include further
fragments or define the same field twice.

## Fields

### Model, data, and routing

| Field | Default | Purpose |
| --- | --- | --- |
| `model_path` | Required | Model path, or a list of compatible model paths. |
| `model_type` | Required | `generative` or `embedding`. |
| `device` | Required | ONNX: `cpu` or `cuda`; PT also supports `mps`. |
| `project_root` | `.` | Base for project paths. |
| `data_path` | Metadata test/last split | Input file or folder. Usually required with artifact-only inference because artifacts do not store split paths. |
| `preprocessing_data_path` | `null` | Derives the generated metadata path. |
| `metadata_config_path` | `null` | Explicit preprocessing metadata. |
| `training_config_path` | `null` | Training config used to resolve a route. |
| `dataset` / `part` | `null` | Select a dataset and optional part from the training config. |
| `model_interface` | Implicit when unique | Select a route from a training config or PT artifact. |
| `read_format` | `parquet` | `csv`, `parquet`, or folder-based `pt`. |
| `write_format` | `csv` | `csv` or `parquet`. |
| `inference_batch_size` | `1` | Sequences processed per batch. |

### Model contract

The following fields are optional when supplied by a PT artifact or training
route, and otherwise required as applicable: `input_columns`, `target_columns`,
`column_data_types`, `target_column_types`, `training_objective`,
`context_length`, `target_offset`, and `prediction_length`.

`window_stride` optionally evaluates several model windows inside each
stored preprocessing row. `null` uses the legacy right-aligned view.

### Output and runtime options

| Field | Default | Purpose |
| --- | --- | --- |
| `output_probabilities` | `false` | Write full distributions for categorical targets. Invalid for embeddings. |
| `decode_categories` | `true` | Decode categorical predictions to their original values. |
| `sample_from_distribution_columns` | `null` | Sample these categorical targets instead of using argmax. |
| `infer_with_dropout` | `false` | Enable dropout at inference; ONNX also requires dropout-preserving export. |
| `deterministic` | `false` | Request deterministic PyTorch algorithms. |
| `seed` | `1010` | Random seed. |
| `autoregressive` | `false` | Feed predictions back for multi-step generation. |
| `generation_steps` | `null` | Required positive step count when autoregressive is enabled. |

Autoregressive inference requires a forward-looking generative model, prediction length
`1`, and identical input and target columns. It begins at the first input window
for each sequence and generates the same number of steps for every sequence.

## CLI overrides

`--data-path`, `--input-columns`, `--metadata-config-path`, `--model-path`,
`--seed`, `--dataset`, `--part`, and `--model-interface` override YAML values.
`--randomize` takes precedence over `--seed`. `--skip-metadata` skips external
metadata loading; inline metadata, a selected training route, or a self-describing
PT artifact must then provide the required metadata.

## Outputs

- Generative predictions: `outputs/predictions/`.
- Categorical probabilities, when enabled: `outputs/probabilities/`.
- Embeddings: `outputs/embeddings/`.

Outputs contain sequence and window identifiers. Categorical values are decoded
when `decode_categories` is enabled, and normalized real outputs are restored to their
original scale. Folder inputs produce sharded output directories in the same
locations.
