<img src="./design/sequifier.png">


## What is sequifier?

Sequifier makes training and inference of powerful transformer sequence models fast and trustworthy.

It can be used to train causal and masked reconstuction tranformer models, and causal variants 'next occurrence' and 'final value', which do not use the next token value but the next *relevant* token value as target during training.

The process looks like this:

<img src="./design/sequifier-illustration.png">



### Value Proposition

Implementing a model from scratch takes time, and there are a surprising number of aspects to consider. The idea is: why not do it once, make it configurable, and then use the same implementation across domains and datasets.

This gives us a number of benefits:

- rapid prototyping
- configurable architecture
- trusted implementation (you can't create bugs inadvertedly)
- standardized logging
- native multi-gpu support (DDP and FSDP)
- native multi-core preprocessing
- scales to datasets larger than RAM
- hyperparameter optimization using Optuna (Bayesian, Random, or Grid search)
- can be used for prediction, generation, and embedding of arbitrary sequences

The only requirement is having sequifier installed, and having input data in the right format.


### The Six Commands

There are six standalone commands within sequifier: `make`, `preprocess`, `train`, `infer`, `hyperparameter-search`, and `visualize-training`.

`make` sets up a new sequifier project in a new folder, `preprocess` preprocesses the data from the input format into subsequences of a fixed length, `train` trains a model on the preprocessed data, `infer` generates predictions, probabilities, or embeddings from data in the preprocessed format, `hyperparameter-search` executes multiple training runs using Optuna to find optimal configurations, and `visualize-training` reads structured training metrics to generate interactive HTML plots of your loss curves.

There are documentation pages for each command, except make:

 - [preprocess documentation](./documentation/configs/preprocess.md)
 - [train documentation](./documentation/configs/train.md)
 - [infer documentation](./documentation/configs/infer.md)
 - [hyperparameter-search documentation](./documentation/configs/hyperparameter-search.md)
 - [visualize-training documentation](./documentation/commands/visualize-training.md)



### Other Materials

To get the full auto-generated documentation, visit [sequifier.com](https://sequifier.com)

If you want to first get a more specific understanding of the transformer architecture, have a look at
the [Wikipedia article.](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

If you want to see an end-to-end example on very simple synthetic data, check out this [this notebook.](./documentation/demos/self-contained-example.ipynb)



## Structure of a Sequifier Project

Sequifier is designed with a specific folder structure in mind:

```text
YOUR_PROJECT_NAME/
├── configs/
│   ├── preprocess.yaml
│   ├── train.yaml
│   └── infer.yaml
├── data/
│   └── (Place your CSV/Parquet files here)
├── models/
├── checkpoints/
├── outputs/
│   ├── embeddings(?)
│   ├── predictions(?)
│   ├── probabilities(?)
│   └── visualization/
├── logs/
├── state/
└── scripts/

```

The `sequifier` commands should typically be run in the project root.

Within YOUR_PROJECT_NAME, you can also add other folders for additional steps, such as `notebooks` or `scripts` for pre- or postprocessing, and `analysis`, `visualizations` or `evals` for files you generate in other, manual steps.

### Data Transformations in Sequifier

Let's start with the data format expected by sequifier. The basic data format that is used as input to the library takes the following form:

|sequenceId|itemPosition|column1|column2|...|
|----------|------------|-------|-------|---|
|0|0|"high"|12.3|...|
|0|1|"high"|10.2|...|
|...|...|...|...|...|
|1|0|"medium"|20.6|...|
|...|...|...|...|...|

The two columns "sequenceId" and "itemPosition" have to be present, and then there must be at least one feature column. There can also be many feature columns, and these can be categorical or real valued.

Data of this input format can be transformed into the format that is used for model training and inference using `sequifier preprocess`. Preprocessing defines the physical `window_length` and `max_target_offset`; training and inference choose the model-facing `context_length` from that stored capacity:

|sequenceId|subsequenceId|startItemPosition|leftPadLength|inputCol|[Context Length - 1]|[Context Length - 2]|...|0|
|----------|-------------|-----------------|-------------|--------|-------------------|-------------------| - |-|
|0|0|0|0|column1|"high"|"high"|...|"low"|
|0|0|0|0|column2|12.3|10.2|...|14.9|
|...|...|...|...|...|...|...|...|...|
|1|0|15|0|column1|"medium"|"high"|...|"medium"|
|1|0|15|0|column2|20.6|18.5|...|21.6|
|...|...|...|...|...|...|...|...|...|

Generative inference returns a row-oriented table with the predicted target
columns plus identifiers for the source sequence and model window:

|sequenceId|subsequenceId|windowStartOffset|itemPosition|column1|column2|...|
|----------|-------------|-----------------|------------|-------|-------|---|
|0|0|0|963|"medium"|8.9|...|
|0|0|0|964|"low"|6.3|...|
|...|...|...|...|...|...|...|
|1|4|0|732|"medium"|14.4|...|
|...|...|...|...|...|...|...|



### Complete Example of Training and Inferring a Transformer Model

Once you have your data in the input format described above, you can train a transformer model in a couple of steps on them.

1.  Create and activate an environment with Python \>=3.10, then run

```console
pip install sequifier
```

2.  To create the project folder with the config templates in the configs subfolder, run

```console
sequifier make YOUR_PROJECT_NAME
```

3.  cd into the `YOUR_PROJECT_NAME` folder, create a `data` folder and add your data and adapt `preprocessing_data_path` in `preprocess.yaml` to point to the data
4.  run

```console
sequifier preprocess
```

5.  the preprocessing step outputs metadata at `configs/metadata_configs/[INPUT BASENAME].json`. For a single dataset and part, reference that file from `dataset.part.metadata_config_path` in `train.yaml`; named configurations use `dataset_training.<dataset>.parts.<part>.metadata_config_path`. Inference may still use `preprocessing_data_path` or `metadata_config_path`
6.  Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run


```console
sequifier train
```

7.  point `model_path` in `infer.yaml` at the default ONNX export. Keep the
    scaffold's explicit contract, or replace it with `training_config_path` and
    `dataset`; see the [ONNX/PT trade-offs](./documentation/configs/infer.md#onnx-or-pt)
8.  run


```console
sequifier infer
```

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/[EXPORTED_MODEL_BASENAME]/part-000.[FORMAT]`, for example `outputs/predictions/your-model-best-3/part-000.csv`


## Other Features

### Causal Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:

- Training: Set export_embedding_model: true in the training config.
- Inference: Set model_type: embedding in the inference config.

Technical Details: Selected activations are restricted to the configured final
`prediction_length` positions and concatenated in configuration order along the
feature dimension. Backbone selectors contribute `dim_model` values. Decoder MLP
hidden-block selectors contribute their configured hidden width and receive the
same flattened `decoding_support * dim_model` windows used during training. The
default, `embedding_layer_names: [backbone.final_norm]`, preserves the final
normalized backbone representation.

If you are interested in activations *other* than the last backbone layer, you can configure the exact layers you want to contribute to the export using `embedding_layer_names`. You can pass an ordered list, such as
- Activation sources: Set `embedding_layer_names` to an ordered list such as `[backbone.layers.1, decoder.branches.default.hidden_blocks.0]`, and the activations of these layers will be concatenated and output.

Layer names follow the network hierarchy using zero-based indices: `backbone.layers.<index>` selects a transformer block output, `backbone.final_norm` the normalized backbone output, and `decoder.branches.<branch>.hidden_blocks.<index>` an MLP decoder hidden-block output; the same scheme applies to BERT embedding models.

### BERT Model

Sequifier also supports training and inference of BERT-style masked reconstruction models.

Configuration:

- Preprocessing: Set `max_target_offset: 0` for equal-width input and target windows.
- Training: Set `training_objective: bert`, configure `bert_spec`, and set decoder `prediction_length` equal to `context_length`. Enable generative and/or embedding export according to the desired inference.
- Inference: Set `model_type: generative` to reconstruct explicitly masked input, or `model_type: embedding` to output contextual representations.

Technical Details: BERT-style models use bidirectional attention and learn by reconstructing positions sampled according to `bert_spec`. Inference does not apply random masking; inputs that should be reconstructed must be masked explicitly, for example using `mask_column` during preprocessing. Embedding inference returns one contextual representation for every valid position in the input window.

### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel` and `FullyShardedDataParallel`. To make use of multi gpu support, the preprocessing step must write sharded output with `merge_output: false`. `write_format: pt` is the recommended production format; sharded `parquet` is also supported but currently considered beta for distributed training.

For the full guide on how to configure a distributed run, check the [multi-GPU training guide](./documentation/training/multi-gpu-training.md).

### System Requirements

Tiny transformer models on little data can be trained on CPU. Bigger ones require an Nvidia GPU with a compatible CUDA version installed.

Sequifier currently runs on MacOS and Ubuntu.

## Citation

Please cite with:

```bibtex
@software{sequifier_2025,
  author = {Luithlen, Leon},
  title = {sequifier - transformers for multivariate sequence generation and representation learning},
  year = {2025},
  publisher = {GitHub},
  version = {v2.0.0.0},
  url = {https://github.com/0xideas/sequifier}
}

```


# Preprocess Command Guide

The `sequifier preprocess` command transforms raw tabular data (CSV or Parquet) into the specific sequence format required for training transformer sequence models. It handles windowing, data splitting (train/validation/test), categorical encoding, and optional numerical standardization.

## Usage

```console
sequifier preprocess --config-path configs/preprocess.yaml
```

## CLI Overrides

Values passed on the command line override the YAML before validation.

| Flag | Overrides / Action |
| :--- | :--- |
| `-r`, `--randomize` | Generates a random `seed`. The seed affects `between_sequence` split assignment. |
| `-dp`, `--data-path` | Overrides `preprocessing_data_path`. |
| `-sc`, `--selected-columns` | Overrides `selected_columns` with a space-separated list. Use `None` to process all columns. |

## Composable Configuration Files

A preprocessing entry config may set `additional_config_paths` to one
non-empty string, a list of non-empty strings, or `null`. Relative paths
resolve against the entry config's `project_root`; absolute paths are used
directly. Fragments are direct only and cannot include further fragments. They
may share nested containers when their child fields are disjoint, but duplicate
fields are errors. CLI values override the completed file composition.

## Configuration Fields

The configuration is defined in a YAML file (e.g., `preprocess.yaml`). Below are the available fields, their requirements, and their functions.

### 1\. File System & Input/Output

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `additional_config_paths` | `str`, `list[str]`, or `null` | No | `null` | Direct complementary YAML fragments. Relative paths resolve against `project_root`; recursive composition and duplicate fields are rejected. |
| `preprocessing_data_path` | `str` | **Yes** | - | Path to the raw input file or folder. |
| `read_format` | `str` | No | `csv` | Format of input data (`csv`, `parquet`). |
| `write_format` | `str` | No | `parquet` | Format of output data (`csv`, `parquet`, `pt`). |
| `merge_output` | `bool` | No | `true` | Whether to merge split files into single files or keep them sharded. |
| `continue_preprocessing`| `bool` | No | `false` | If `true`, resumes from an existing preprocessing temp folder created by an interrupted run. |


> **Important Constraint on `write_format`:**
>
>   * If `write_format` is **`pt`** (PyTorch tensors), `merge_output` must be **`false`**.
>   * If `write_format` is **`parquet`**, `merge_output` can be **`false`** or **`true`**.
>   * If `write_format` is **`csv`**, `merge_output` must be **`true`**.
> For distributed training, `merge_output` must be set to **`false`**.

### 2\. Column Selection & Filtering

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `selected_columns` | `list[str]` | No | `null` | A specific list of columns to process. If `null`, all columns (except metadata) are processed. |
| `column_data_types` | `dict[str, str]` | No | `null` | Optional output dtype map for processed columns, such as `Float32`, `Float64`, `Int32`, or `Int64`. If set, every processed column must be included. Parquet uses one unified sequence dtype; `pt` writes each variable to its configured tensor dtype. |
| `normalize_real_columns` | `bool` | No | `true` | If `true`, Z-score normalizes real-valued columns. Set to `false` to preserve their original values. Statistics are still recorded in metadata. |
| `max_rows` | `int` | No | `null` | Limits processing to the first N rows. Useful for rapid debugging. |
| `metadata_config_path` | `Optional[str]` | No | `null` | Use a preexisting metadata config for tokenizing discrete columns and, when enabled, standardizing real-valued columns. |
| `mask_column` | `Optional[str]` | No | `null` | Optional input column used as a row-level mask. If set, `metadata_config_path` must also be set. |
| `use_precomputed_maps`| `list[str]` | No | `null` | If not `null`, enforces the use of precomputed maps for the variables in the list. |

### 3\. Sequence Logic & Splitting

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `window_length` | `int` | **Yes** | - | The physical serialized window width written to preprocessed data. |
| `max_target_offset` | `int` | No | `1` | Number of future items retained after the model input window. Use `0` for BERT-style same-width inputs and targets; use `1` for causal next-item training. |
| `split_ratios` | `list[float]`| **Yes** | - | Ordered train/validation/test proportions. Must sum to 1.0. |
| `split_method` | `str` | No | `within_sequence` | How rows are assigned to splits (`within_sequence` or `between_sequence`). |
| `window_strides` | `list[int]` | No | `[window_length]*N` | Window stride for each split; entry `i` corresponds to `split_ratios[i]`. |
| `window_placement`| `str` | No | `distribute` | Strategy for selecting start indices (`distribute` or `exact`). |
| `allow_sequence_splitting` | `bool` | No | `false` | If `false`, a single sequence is kept within one preprocessing batch. |

### 4\. Performance & System

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `seed` | `int` | No | `1010` | Random seed for reproducibility. |
| `n_cores` | `int` | No | Max Cores | Number of CPU cores to use for parallel processing. |
| `batches_per_file` | `int` | No | `1024` | Only used when `write_format: pt`. Controls how many sequences are packed into one `.pt` file. |
| `process_by_file` | `bool` | No | `true` | Memory optimization. If `true`, processes one input file at a time. |

-----

## Key Trade-offs and Decisions

### 1\. `write_format`: `parquet` vs. `pt`

  * **Choose `parquet` (default):** Unless you have a specific reason, use `parquet`. *Note: If you are doing distributed training, Parquet support is currently in **Beta**.*
  * **Choose `pt`:** Use `pt` data loading if speed and CPU overhead are your primary bottlenecks, **or if you are running multi-GPU distributed training.** This format is the most stable choice for high-throughput scaling.

### 2\. `window_strides` configuration

- `window_length`: non-overlapping windows and less data.
- `1`: maximum overlap, coverage, storage, and training time.
- A common compromise is a larger train/validation stride and test stride `1`,
  for example `window_strides: [24, 24, 1]`.

### 3\. `window_placement`: `distribute` vs `exact`

  * **`distribute` (Default):** The algorithm adjusts the start indices slightly to minimize the overlap of the final subsequence with the previous one, ensuring the data covers the full sequence length as evenly as possible. Recommended for most use cases.
  * **`exact`:** Strictly enforces the stride. If the sequence length minus the window size isn't perfectly divisible by the stride, this will raise an error. Use this only if mathematical precision of the sliding window is strictly required by your downstream application or evaluation code.

### 4. Advanced: Static Vocabularies (Custom ID Maps)

By default, Sequifier dynamically builds ID maps from the data found in the input file. However, in production systems, you often need a **fixed vocabulary** to ensure that ID "105" always maps to "Item_X", regardless of the daily training batch.

To use a static vocabulary:
1. Create a folder `configs/id_maps/` in your project root.
2. Add JSON files named `{COLUMN_NAME}.json`.
3. The format must be a dictionary mapping ordinary data values to integers **starting at 3**. Reserved labels may be included only with their fixed IDs.

> **Reserved Indices:**
> * **0**: Reserved for `[unknown]` (padding/missing).
> * **1**: Reserved for `[other]` (unseen values not in your map).
> * **2**: Reserved for `[mask]`.
> * **3+**: Your data.

**Example `configs/id_maps/itemId.json`:**
```json
{
    "apple": 3,
    "banana": 4,
    "cherry": 5
}
```
-----

## Outputs

After running `preprocess`, the following are generated:

1.  **Data Files:** Located in `data/`. Depending on your configuration, these will be merged files such as `[NAME]-split0.parquet` (Training), `[NAME]-split1.parquet` (Validation), etc., or split folders such as `[NAME]-split0/` containing `.pt` or `.parquet` shards.
2.  **Metadata Config:** Located in `configs/metadata_configs/[NAME].json`.
      * **Crucial:** This file contains the integer mappings for categorical variables (`id_maps`), statistics for real variables (`selected_columns_statistics`), and whether those variables were normalized (`normalize_real_columns`).
      * **Next Step:** Reference this file from `dataset.part.metadata_config_path` in a singleton training config, or from `dataset_training.<dataset>.parts.<part>.metadata_config_path` in a named training config. In inference, either `preprocessing_data_path` or `metadata_config_path` can locate the metadata and its split paths.


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
project_root: .
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
| Runtime | ONNX Runtime on CPU or CUDA (with a CUDA-enabled ONNX Runtime installation). | PyTorch on CPU, CUDA, or MPS. |
| Configuration | Needs a training route or explicit contract and metadata. | Embeds its contract and metadata. |
| Behavior | Runs the exported graph; dropout requires a dropout-preserving export. | Retains PyTorch behavior and supports self-describing, multi-interface bundles. |

Benchmark the target workload rather than assuming either runtime is faster.
Choose PT when portability matters less than a compact, self-contained config:

```yaml
project_root: .
model_path: models/event-model-best-5.pt
data_path: data/events-test.parquet
model_type: generative
device: cuda
```

For multi-interface PT, add `model_interface`. `model_type` stays explicit for
both formats because inference may generate outputs or extract embeddings.
`project_root` is required (normally `.`), and `inference_batch_size` defaults
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
| `device` | Required | ONNX: `cpu` or `cuda`; PT also supports `mps`. ONNX CUDA requires an installed runtime exposing `CUDAExecutionProvider`. |
| `project_root` | Required | Base for project paths; normally `.`. |
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
`1`, and identical input and target columns. For tabular CSV or Parquet input,
it begins at the first input window for each sequence and generates the same
number of steps for every sequence. Folder-based PT input is processed using
its stored windows rather than this first-window reduction.

## CLI overrides

`--data-path`, `--input-columns`, `--metadata-config-path`, `--model-path`,
`--seed`, `--dataset`, `--part`, and `--model-interface` override YAML values.
`--randomize` takes precedence over `--seed`. `--skip-metadata` skips external
metadata loading; inline metadata, a selected training route, or a self-describing
PT artifact must then provide the required metadata.

## Outputs

- Generative predictions: `outputs/predictions/<model>/part-NNN.<format>`.
- Categorical probabilities, when enabled:
  `outputs/probabilities/<model>/<target-column>/part-NNN.<format>`.
- Embeddings: `outputs/embeddings/<model>/part-NNN.<format>`.

`<model>` is the model artifact filename without its extension. Prediction and
embedding outputs contain sequence and window identifiers. Probability files
contain only the class-probability columns for their target. Categorical
predictions are decoded when `decode_categories` is enabled, and normalized
real predictions are restored to their original scale. Every input writes one
or more numbered parts.


# Visualize Training Command Guide

The `sequifier visualize-training` command reads the structured metric files generated during training and hyperparameter search to create interactive Plotly HTML visualizations of the training and validation losses. It supports viewing a single model's progress or comparing multiple models side-by-side.

## Usage

```console
# Visualize a single model
sequifier visualize-training my-model-name

# Visualize multiple models side-by-side
sequifier visualize-training model-A,model-B,model-C

# Visualize every run from a hyperparameter search
sequifier visualize-training my-hyperparameter-search

# Visualize models listed in a text file
sequifier visualize-training path/to/models.txt --log-scale

```

## Arguments

Unlike other commands that rely on a YAML config, `visualize-training` is configured directly via command-line arguments.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `models` | `str` | **Required** | A model name, hyperparameter-search name, comma-separated list of model names, or path to a `.txt` file containing model names (one per line). A search name includes all models named `[SEARCH]-run-[NUMBER]`. |
| `--log-scale` | `flag` | `False` | Use a logarithmic scale on the y-axis for the loss curves. |
| `--bucket-training-batches` | `int` | `null` | Smooths the training loss curve by averaging the loss over a specified number of batches. **Must be a multiple of the logged batch interval** used during training. |
| `--project-root` | `str` | `.` | The root directory of your Sequifier project. |

For a single-dataset model, the command reads `logs/[MODEL_NAME]/[MODEL_NAME]-training-full.csv` and `logs/[MODEL_NAME]/[MODEL_NAME]-validation-full.csv`. The corresponding files without `-full` contain only condensed global-loss records.

## Outputs

The interactive HTML reports are saved in the `outputs/visualization/` directory.

* **Single Model:** `outputs/visualization/[MODEL_NAME]-training-visualization.html` (Includes global losses and normalized variable validation losses if applicable).
* **Multiple Models:** `outputs/visualization/multi-model-training-visualization.html` (Side-by-side comparison of validation and training losses across all specified models).
* **Hyperparameter Search:** `outputs/visualization/[SEARCH_NAME].html` (Includes all valid runs and lists skipped invalid runs and their reasons).

If every run in a hyperparameter search is invalid, Sequifier still creates the report with an empty plot and the invalid-run list.

When comparing multiple models, their initial baseline validation loss must match unless `SKIP_BASELINE_CHECK` or `SEQUIFIER_SKIP_BASELINE_CHECK` is set.


# Hyperparameter Search Command Guide

`sequifier hyperparameter-search` searches over complete canonical training
configurations with Optuna. It supports Bayesian optimization, random sampling,
finite grid search, cooperative pruning, and custom single- or multi-objective
evaluation.

```console
sequifier hyperparameter-search --config-path configs/hyperparameter-search.yaml
```

## Minimal search configuration

Every search starts from a canonical training config named by
`base_config_path`. The `parameters` tree describes fixed replacements and
search spaces using the same paths as the training schema.

```yaml
base_config_path: train.yaml
name: transformer-width-search
model_config_write_path: configs/hp-search
method: bayesian
trials: 40

parameters:
  global_training:
    context_length: [64, 128]
    batch_size: [16, 32]
    learning_rate:
      low: 0.0001
      high: 0.001
      log: true
  model:
    backbone:
      architecture:
        num_layers: {low: 4, high: 8, step: 2}
  training_plan:
    epochs: [2, 4]
```

`base_config_path` resolves relative to the hyperparameter-search entry file.
The base may itself be a composed training config. If the search config supplies
`project_root`, that value is used in every generated trial; otherwise the base
training config's root is inherited.

For a singleton base, use `model.interface`, `dataset.part`, and
`training_plan.epochs`. Named paths remain available for multi-value bases.

The historical self-contained search schema and historical flat training base
configs are not accepted. Every generated trial is validated as an authored
canonical `SequifierConfig` before training begins, so unknown paths, invalid
references, incompatible component types, and cross-field violations fail with
their canonical validation paths.

`model_name` and `project_root` cannot appear in `parameters`. Generated model
names use `[name]-run-[index]`, and `project_root` is controlled by the
top-level search field.

## Parameter expressions

Fields omitted from `parameters` retain their base values. Parameter expressions
have these forms:

| Form | Meaning |
| --- | --- |
| Scalar | Fixed replacement. |
| List on a scalar field | Categorical choices. |
| `{low, high, step?, log?, type?}` | Integer or float distribution. `type` may be `int` or `float`; otherwise the base value and bounds determine it. |
| `{choices: [...]}` or `{$choices: [...]}` | Categorical choices for an entire value, including mappings and lists. |
| `{fixed: value}` or `{$fixed: value}` | Unambiguous fixed replacement for a mapping or list. |
| Numeric keys under a base list | Recursive replacement of zero-based list entries. |
| `{variants: [...]}` or `{$variants: [...]}` | Paired partial mapping variants, optionally followed by independently sampled sibling fields. |

Integer ranges default to `step: 1`. Grid search requires a `step` for float
ranges because an unstepped float interval is infinite. Logarithmic integer
ranges require `step: 1`; logarithmic float ranges cannot use `step`.

A direct list of lists samples a complete list-valued field. Prefer the explicit
`choices` wrapper for mapping or list candidates:

```yaml
parameters:
  model:
    backbone:
      architecture:
        choices:
          - dim_model: 128
            max_context_length: 512
            num_layers: 4
            attention: {type: mha, n_heads: 8}
            feed_forward: {dim: 512, activation: swiglu}
          - dim_model: 256
            max_context_length: 512
            num_layers: 6
            attention: {type: gqa, n_heads: 16, n_kv_heads: 4}
            feed_forward: {dim: 1024, activation: swiglu}
```

Each choice replaces the complete overridden subtree, keeping coupled values
valid. The same mechanism can select complete ingestion or decoder components,
training phases, or other canonical subtrees.

Use `variants` when candidates should patch the base mapping and other sibling
fields should remain independent. A variant that changes a component's `type`
starts from that new component shape rather than retaining fields belonging to
the old type.

```yaml
parameters:
  model:
    interfaces:
      event_prediction:
        ingestion:
          variants:
            - {type: embedding, output_dim: 128}
            - {type: passthrough, output_dim: 128}
          dropout: [0.0, 0.1]
```

## Search controls

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `base_config_path` | `str` | Yes | - | Canonical training config used as the base. |
| `parameters` | `mapping` | Yes | - | Recursive fixed values and search spaces. May be empty. |
| `project_root` | `str` | No | Base value | Root written to every generated training config. |
| `additional_config_paths` | `str`, `list[str]`, or `null` | No | `null` | Complementary direct fragments of the search config. Relative paths resolve against the entry config's `project_root`; recursive composition and duplicate fields are rejected. |
| `name` | `str` | Yes | - | Study name and generated-run prefix. |
| `model_config_write_path` | `str` | Yes | - | Directory, under `project_root`, for generated trial configs. |
| `method` | `bayesian`, `sample`, or `grid` | No | `bayesian` | Optuna TPE, random, or exhaustive finite-grid sampling. |
| `trials` | `int` | Except grid | - | Target total number of completed or pruned runs in the persisted study. |
| `global_seed` | `int` or `null` | No | `null` | Sampler seed. Training seeds belong in canonical `parameters`. |
| `prune_trials` | `bool` | No | `true` | Enables cooperative pruning. Distributed pruning remains experimental. |
| `pruning_warmup_epochs` | `int` or `null` | No | `null` | Complete epochs before pruning may begin. Mutually exclusive with batch warmup. |
| `pruning_warmup_batches` | `int` or `null` | No | `null` | Training batches before pruning may begin. Mutually exclusive with epoch warmup. |
| `evaluation_metrics` | `list[str]` or `null` | No | `null` | Metric names expected from a custom evaluation script. |
| `evaluation_metric_directions` | `list[minimize\|maximize]` or `null` | With metrics | `null` | One optimization direction per metric. |
| `evaluation_script` | `str` or `null` | With metrics | `null` | Script invoked with the best exported model's evaluation ID. |
| `evaluation_inference_config` | `str` or `null` | No | `null` | Inference config run before the custom evaluation script. |

For grid search, omitting `trials` runs the complete finite grid. If it is
provided, it must exactly equal the grid size. For Bayesian and random search,
`trials` is a target total across invocations of the persisted study. If an
identical completed or pruned parameter set is proposed again, it is recorded
as a failed duplicate and does not consume a generated run number or count
toward the target.

## Custom evaluation

Without custom metrics, Sequifier minimizes the best validation loss. With one
metric it performs single-objective optimization; with several it records the
Pareto front.

```yaml
evaluation_metrics: [accuracy, latency_ms]
evaluation_metric_directions: [maximize, minimize]
evaluation_inference_config: configs/infer-validation.yaml
evaluation_script: scripts/evaluate.py
```

The evaluation script receives the exported model's evaluation ID, formatted
as `<run-name>-best-<epoch>`, as its only
argument. It must write
`outputs/evaluations/[evaluation-id].json` under `project_root`, containing all
configured metric names. Extra metrics are allowed but produce a warning and
are ignored by the optimization.

## CLI and outputs

The search definition comes from YAML. `--skip-metadata` is the only
configuration-related command flag: it validates the canonical base without
loading metadata-derived values, so all required authored fields must already
be present.

Generated canonical training configs are written below
`model_config_write_path`. The Optuna SQLite study is persisted at
`state/optuna/[name].db` under `project_root`, allowing later
invocations to continue toward the configured total.


# Distributed and Multi-Node Training in Sequifier

Sequifier natively supports multi-GPU and multi-node training using PyTorch's `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP).

## 1. Prerequisites: Preprocessing for Distributed Training

To use distributed training, your data must be sharded into multiple files so that different GPUs can read different chunks simultaneously without memory bottlenecks.

In your `preprocess.yaml`, you **must** write sharded output:

```yaml
merge_output: false
```

For production multi-GPU training, use PyTorch tensor shards:

```yaml
write_format: pt
```

*Note: Distributed training is not supported if your data is kept as a single `csv` or `parquet` file. You must use `merge_output: false` to generate a folder of sharded files.*

> **Beta Notice for Parquet in Distributed Training:**
> While `write_format: parquet` is supported for distributed training, it is currently considered **Beta**. Because Parquet chunk reading relies on Polars' multi-threading, using it alongside PyTorch's multiprocess `DataLoader` in heavy multi-GPU environments can lead to CPU thread contention, high RAM usage, or NCCL timeouts.
> **Recommendation:** For production multi-GPU runs, use `write_format: pt`. It relies on native PyTorch serialization and is significantly more stable under heavy hardware loads.


## 2. Configuration: `train.yaml`

Once your data is preprocessed into `.pt` shards, or beta `.parquet` shards, you need to tell the Sequifier training engine to expect a distributed environment.

In your `train.yaml`, configure the canonical `global_training` block:

```yaml
global_training:
  read_format: pt             # or parquet for beta sharded Parquet loading
  distributed: true
  data_parallelism: fsdp # or ddp
  fsdp_cpu_offload: false   # omit if using ddp; set true to offload FSDP parameters to CPU RAM
  layer_type_dtypes: null    # required for FSDP; use layer_autocast for mixed precision
  torch_compile: inner       # use inner or none for FSDP; use outer or none for DDP
  world_size: 32       # The TOTAL number of GPUs across all nodes (e.g., 8 nodes * 4 GPUs = 32)
  backend: nccl        # 'nccl' is the standard and most efficient backend for NVIDIA GPUs

```

When shards do not divide evenly across ranks, Sequifier automatically pads shorter ranks with repeated samples for step alignment. Those repeats are masked out of loss calculation, so each real sample contributes once.

## 3. Launching the Training Job

How you launch the training depends on whether you are using a single machine with multiple GPUs, or multiple machines (nodes) connected over a network.

### Scenario A: Single-Node, Multi-GPU

If you are running on a single machine that has multiple GPUs (e.g., an AWS EC2 instance with 4x A100s), Sequifier can handle process generation internally using `torch.multiprocessing.spawn`.

You simply run the standard command:

```bash
sequifier train --config-path configs/train.yaml

```

Sequifier will read the `world_size` config parameter and automatically spawn that exact number of worker processes.

### Scenario B: Multi-Node, Multi-GPU (HPC / Slurm)

Sequifier cannot automatically spawn Python processes across physical network boundaries. For multi-node training, you must use an external cluster manager (like Slurm) combined with PyTorch's `torchrun` utility.

When `sequifier` detects `torchrun` environment variables (like `RANK` and `WORLD_SIZE`), it bypasses its internal spawner and attaches to the distributed network established by the cluster. In that mode, the environment `WORLD_SIZE` is used.

Here is a standard `sbatch` script template for launching Sequifier across multiple nodes:

```bash
#!/bin/bash
#[SBATCH COMMANDS]

MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun torchrun \
    [-- torchrun args]...
    $(which sequifier) train --config-path=configs/train.yaml
```

### Important Considerations for Multi-Node

* **Batch Size:** The `batch_size` in your `train.yaml` is the **per-process** batch size. If `batch_size` is 100 and `world_size` is 32, each synchronized backward pass covers 3,200 samples. With gradient accumulation, the samples per optimizer update are `batch_size * world_size * accumulation_steps` (apart from a final partial accumulation window).
* **Learning Rate:** You may need to scale your `learning_rate` up if you drastically increase your global batch size via distributed training.
* **Data Access:** All nodes must have access to the same shared filesystem (e.g., NFS, GPFS) where the `project_root` and the sharded preprocessing output are stored.
