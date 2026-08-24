<img src="./design/sequifier.png">


## What is sequifier?

Sequifier makes training and inference of powerful transformer sequence models fast and trustworthy.

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
- can be used for prediction, generation and embedding on/of arbitrary sequences

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
├── outputs/
│   ├── embeddings(?)
│   ├── predictions(?)
│   ├── probabilities(?)
│   └── visualization/
└── logs/

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

Data of this input format can be transformed into the format that is used for model training and inference using `sequifier preprocess`. Preprocessing defines the physical `stored_context_width` and `max_target_offset`; training and inference choose the model-facing `context_length` from that stored capacity:

|sequenceId|subsequenceId|startItemPosition|leftPadLength|inputCol|[Window Length - 1]|[Window Length - 2]|...|0|
|----------|-------------|-----------------|-------------|--------|-------------------|-------------------| - |-|
|0|0|0|0|column1|"high"|"high"|...|"low"|
|0|0|0|0|column2|12.3|10.2|...|14.9|
|...|...|...|...|...|...|...|...|...|
|1|0|15|0|column1|"medium"|"high"|...|"medium"|
|1|0|15|0|column2|20.6|18.5|...|21.6|
|...|...|...|...|...|...|...|...|...|

On inference, the output is returned in the library input format, introduced first.

|sequenceId|itemPosition|column1|column2|...|
|----------|------------|-------|-------|---|
|0|963|"medium"|8.9|...|
|0|964|"low"|6.3|...|
|...|...|...|...|...|
|1|732|"medium"|14.4|...|
|...|...|...|...|...|



### Complete Example of Training and Inferring a Transformer Model

Once you have your data in the input format described above, you can train a transformer model in a couple of steps on them.

1.  create a conda environment with python \>=3.10 and \<=3.13 activate and run

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

5.  the preprocessing step outputs metadata at `configs/metadata_configs/[FILE NAME]`. Reference that file from `dataset_training_spec.<dataset>.parts.<part>.metadata_config_path` in `train.yaml`; inference may still use `preprocessing_data_path` or `metadata_config_path`
6.  Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run


```console
sequifier train
```

7.  optionally override `data_path` in `infer.yaml`; otherwise it defaults to the inference/test split from preprocessing metadata
8.  run


```console
sequifier infer
```

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/[EXPORTED_MODEL_BASENAME]-predictions.[FORMAT]`, for example `outputs/predictions/your-model-best-predictions.csv`


## Other Features

### Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:

- Training: Set export_embedding_model: true in the training config.
- Activation sources: Set `embedding_layer_names` to an ordered list such as
  `[backbone.layers.1, decoder.branches.default.hidden_blocks.0]`.
- Inference: Set model_type: embedding in the inference config.

Technical Details: Selected activations are restricted to the configured final
`prediction_length` positions and concatenated in configuration order along the
feature dimension. Backbone selectors contribute `dim_model` values. Decoder MLP
hidden-block selectors contribute their configured hidden width and receive the
same flattened `decoding_support * dim_model` windows used during training. The
default, `embedding_layer_names: [backbone.final_norm]`, preserves the final
normalized backbone representation. Because a causal model is trained to predict
future state, its embedding is forward-looking.

### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel` and `FullyShardedDataParallel`. To make use of multi gpu support, the preprocessing step must write sharded output with `merge_output: false`. `write_format: pt` is the recommended production format; sharded `parquet` is also supported but currently considered beta for distributed training.

For the full guide on how to configure a distributed run, check the [multi-GPU training guide](./documentation/training/multi-gpu-training.md).

### System Requirements

Tiny transformer models on little data can be trained on CPU. Bigger ones require an Nvidia GPU with a compatible cuda version installed.

Sequifier currently runs on MacOS and Ubuntu.

## Citation

Please cite with:

```bibtex
@software{sequifier_2025,
  author = {Luithlen, Leon},
  title = {sequifier - causal transformer models for multivariate sequence modelling},
  year = {2025},
  publisher = {GitHub},
  version = {v2.0.0.0},
  url = {[https://github.com/0xideas/sequifier](https://github.com/0xideas/sequifier)}
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
| `stored_context_width` | `int` | **Yes** | - | The physical serialized window width written to preprocessed data. |
| `max_target_offset` | `int` | No | `1` | Number of future items retained after the model input window. Use `0` for BERT-style same-width inputs and targets; use `1` for causal next-item training. |
| `split_ratios` | `list[float]`| **Yes** | - | Proportions for data splits (e.g., `[0.8, 0.1, 0.1]` for train/val/test). Must sum to 1.0. |
| `split_method` | `str` | No | `within_sequence` | How rows are assigned to splits (`within_sequence` or `between_sequence`). |
| `stride_by_split` | `list[int]` | No | `[stored_context_width]*N` | The step size used to slide the window for each split. Corresponds to `split_ratios`. |
| `subsequence_start_mode`| `str` | No | `distribute` | Strategy for selecting start indices (`distribute` or `exact`). |
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

### 2\. `stride_by_split` configuration

This controls data augmentation and redundancy.

  * **Stride = `stored_context_width` (Non-overlapping):** The model sees every stored window once as a target. Training is faster, but the model might miss patterns that cross the window boundary.
  * **Stride = 1 (Maximum Overlap):** Maximizes data volume. The model sees every possible sequence. This yields the highest accuracy but significantly increases the size of the preprocessed data and training time.
  * **Hybrid Approach:** It is common practice to set a large stride for the training and validation splits (indices 0 and 1) to reduce the size on disk of the dataset, and a stride=1 for the test split to evaluate the model on each point in the test set. This supposes that the test split value is low.
      * *Example:* `stride_by_split: [24, 24, 1]` (assuming `stored_context_width: 49`).

### 3\. `subsequence_start_mode`: `distribute` vs `exact`

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
      * **Next Step:** Set `preprocessing_data_path` in `train.yaml` and `infer.yaml` to derive this metadata path and the appropriate split paths automatically. You can still set `metadata_config_path` explicitly.


# Train Command Guide

`sequifier train` trains one shared transformer backbone through one or more
named model interfaces. An interface is an ingestion module, its generated
adapter, and a decoder. Datasets own data and training policy; `model_spec`
owns architecture.

```console
sequifier train --config-path configs/train.yaml
```

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
export_pt: true
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

For exactly one dataset, `--data-path`, `--validation-data-path`,
`--metadata-config-path`, `--input-columns`, and `--skip-metadata` map to the
unambiguous dataset part/interface. They fail when several datasets or parts
make the target ambiguous.

## Artifacts

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


# Infer Command Guide

The `sequifier infer` command uses a trained Sequifier model (PyTorch `.pt` or ONNX `.onnx`) to generate predictions, probabilities, or vector embeddings on new data. It handles batching, data normalization (and denormalization), and supports complex inference modes like **autoregression**.

## Usage

```console
sequifier infer --config-path configs/infer.yaml
```

## CLI Overrides

Values passed on the command line are deep-merged into the authored YAML before
validation. Nested mappings merge recursively, while lists, scalars, `null`,
and typed components replace the YAML value.

| Flag | Overrides / Action |
| :--- | :--- |
| `-r`, `--randomize` | Generates a random `seed`, taking precedence over `--seed`. |
| `-dp`, `--data-path` | Overrides `data_path`. |
| `-ic`, `--input-columns` | Overrides `input_columns` with a space-separated list. Use `None` to derive all columns from metadata. |
| `-mc`, `--metadata-config-path` | Overrides `metadata_config_path`. |
| `-sm`, `--skip-metadata` | Skips loading metadata-derived config values. All required schema fields must then be supplied directly. |
| `-mp`, `--model-path` | Overrides `model_path`. |
| `-s`, `--seed` | Overrides `seed`, unless `--randomize` is also set. |
| `--dataset` | Selects a configured dataset and therefore its mapped model interface. |
| `--part` | Selects a part within `--dataset`; it does not change the interface. |
| `--model-interface` | Selects a named interface directly from a multi-interface PT bundle. |

Inference follows the same authored/resolved boundary as training. The YAML is
validated as `InferenceConfig`, then preprocessing metadata is resolved into an
internal `ResolvedInferenceConfig`. Storage layout, column groups, ID maps, and
normalization statistics are runtime values and do not need to be copied into
authored inference YAML.

## Composable Configuration Files

An inference entry config may set `additional_config_paths` to one non-empty
string, a list of non-empty strings, or `null`. Relative paths resolve against
the entry config's `project_root`; absolute paths are used directly. Fragments
are direct only and cannot include further fragments. They may share nested
containers when their child fields are disjoint, but duplicate fields are
errors. CLI values override the completed file composition.

## Configuration Fields

The configuration is defined in a YAML file (e.g., `infer.yaml`).

### 1\. File System & Model Loading

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `additional_config_paths` | `str`, `list[str]`, or `null` | No | `null` | Direct complementary YAML fragments. Relative paths resolve against `project_root`; recursive composition and duplicate fields are rejected. |
| `preprocessing_data_path` | `str` | Conditional | `null` | Raw preprocessing input path. When set, Sequifier derives `metadata_config_path` and defaults `data_path` to the inference/test preprocessing split. |
| `data_path` | `str` | No | Metadata split 2 | Path to the input data file (`csv` or `parquet`) or folder (`pt` or `parquet`). Defaults to split 2 from metadata, or the last available split if fewer than three splits exist. |
| `model_path` | `str` or `list[str]` | **Yes** | - | Path to a specific model file, or a list of paths to process sequentially, for example `models/my-model-best.pt`. |
| `training_config_path`| `str` | No | `null` | Optional training config used to resolve a dataset selection to its interface. Lean PT bundles reconstruct themselves from `model_config`. |
| `dataset` | `str` | Conditional | `null` | Dataset to resolve through `training_config_path`. Required when that config has several datasets unless `model_interface` is supplied. |
| `part` | `str` | No | `null` | Part within `dataset`; changes the data selection, not model weights. |
| `model_interface` | `str` | Conditional | Implicit for one interface | Named PT route. Required for a multi-interface PT bundle unless `dataset` resolves it. |
| `metadata_config_path`| `str` | Conditional | Derived from `preprocessing_data_path` | Path to the JSON metadata file generated during preprocessing. Required when `preprocessing_data_path` is omitted. |
| `read_format` | `str` | No | `parquet` | Format of input data. Single-file inference supports `csv` and `parquet`; folder inference supports `parquet` and `pt`. |
| `write_format` | `str` | No | `csv` | Format for output predictions (`csv`, `parquet`). |

### 2\. Schema & Columns

These fields tell the inference engine which columns to extract from the new data and how to interpret them.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_columns` | `list[str]` or `null`| **Yes** | `null` | List of feature columns. Must match the columns the model was trained on. Set to `null` to use all metadata columns. |
| `target_columns` | `list[str]`| **Yes** | - | The column(s) to predict. |
| `column_data_types` | `dict` | No | Metadata column types | Map of all columns to their type (e.g., `Int64`, `Float64`). Usually copied from metadata. |
| `target_column_types`| `dict` | Conditional | Derived from `column_data_types` | Map of target columns to `categorical` or `real`. Integer dtypes derive as categorical and floating dtypes derive as real. |

### 3\. Inference Logic & Modes

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model_type` | `str` | **Yes** | - | `generative` (predict next value) or `embedding` (extract vector representation). |
| `training_objective` | `str` | **Yes** | - | Objective used during training: `causal`, `bert`, `final_value`, or `next_occurrence`. |
| `context_length` | `int` | **Yes** | - | The model context window size. It must match the trained model view and fit inside the stored metadata capacity. |
| `model_window_stride` | `int` or `null` | No | `null` | Distance between model-window starts inferred from each stored preprocessing row. `null` uses the legacy right-aligned view; a positive integer infers every contained view on a right-anchored grid. |
| `target_offset` | `int` | No | `1` | Future offset used for forward-looking objectives. BERT-style inference forces this to `0`. |
| `prediction_length` | `int` | No | `1` for forward objectives; `context_length` for BERT | Number of steps to predict *simultaneously*. **Must be 1** if `autoregression: true`. |
| `inference_batch_size`| `int` | **Yes** | - | Number of sequences to process at once. |
| `autoregression` | `bool` | No | `false` | If `true`, feeds predictions back into the model to predict further into the future. |
| `autoregression_total_steps`| `int` | No | `null` | If `autoregression: true`, how many total steps to predict, starting from the *first* subsequence in the inference data. |
| `output_probabilities`| `bool` | No | `false` | If `true`, outputs the full probability distribution for categorical targets. Real-valued targets do not produce probability files. |
| `sample_from_distribution_columns`| `Optional[list[str]]`| No | `null` | If set, the model **samples** from the predicted distribution for these columns instead of taking the top-1 (argmax). Essential for diversity in generation. |
| `map_to_id` | `bool` | No | `true` | If `true`, converts integer class predictions back to original string IDs (e.g., 0 -\> "cat"). Must be `false` when all targets are real-valued. |
| `infer_with_dropout` | `bool` | No | `false` | For PyTorch, explicitly re-enables dropout after model loading. For ONNX, preserves dropout nodes at runtime and is effective only when the model was exported with `export_with_dropout: true`. |
| `seed` | `int` | No | `1010` | Random seed for reproducibility. |

Prediction and embedding outputs include `subsequenceId` and
`windowStartOffset`. `itemPosition` is calculated from the physical stored-row
start plus this model-window offset.

### 4\. System

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `enforce_deterministic_inference` | `bool` | No | `false` | Forces PyTorch inference to use deterministic algorithms. |
-----

## Key Trade-offs and Decisions

### 1\. Input Format (`read_format`)

  * **`csv`:** Best for standard inference on small data. The inferer will filter the data to `input_columns` automatically.
  * **`parquet`** Best for most use cases. Can be used with lazy loading, will use less disk space but more CPU than `pt`
  * **`pt`** Folder-only format optimized for lazy loading. Uses more disk space but less CPU than `parquet`.

### 2\. `model_type`: `generative` vs. `embedding`

  * **`generative`:** Use this when you want to predict the next value in a sequence (forecasting, classification, next-token prediction).
      * *Output:* A file in `outputs/predictions/` containing the predicted values for specific item positions.
  * **`embedding`:** Use this when you want to represent the sequence as a fixed-size vector. The training config's `embedding_layer_names` selects ordered backbone and decoder-MLP activations, which are concatenated into one vector. It defaults to the final normalized backbone output.
      * *Output:* A file in `outputs/embeddings/` containing vectors (e.g., 128 floats) for each sequence. Useful for clustering, similarity search, or downstream ML tasks.

### 3\. Sampling vs. Argmax

  * **Default (Argmax):** The model selects the class with the highest probability. Best for accuracy metrics and "most likely" forecasts.
  * **Sampling (`sample_from_distribution_columns`):** The model picks the next token randomly based on the probability distribution.
      * *Use Case:* Creative generation or simulation where you want diversity. If `Probability(A)=0.6` and `Probability(B)=0.4`, Argmax always picks A. Sampling picks B 40% of the time.


### Autoregressive Inference

When performing multi-step forecasting (`autoregression: true`), the model feeds its own predictions back into itself to generate future time steps. If you are configuring this feature, note the following strict behavioral rules for how generation is handled:

* **Uniform Step Count:** The model will generate the exact same number of predictions (defined by `autoregression_total_steps`) for **all** `sequenceId`s in your dataset.
* **Independent of Ground Truth:** The length of the generated forecast is completely independent of how many actual ground truth values or historical rows exist for a given sequence.
* **Fixed Starting Point:** Generation strictly begins from the **first** subsequence encountered in the inference data for each sequence. The model will anchor to that initial starting point and forecast forward sequentially, meaning any subsequent historical data provided for that specific `sequenceId` will not alter the trajectory of that specific autoregressive loop.
* **Matching Inputs and Targets:** Autoregression requires `input_columns` and `target_columns` to contain the same columns, and it is not available for embedding or BERT-style models.

-----

## Outputs

Results are saved in the `outputs/` folder within your project root.

1.  **Predictions:** `outputs/predictions/[MODEL_NAME]-predictions.[format]`

      * Standard tabular data containing `sequenceId`, `itemPosition`, and columns for your predicted targets.
      * If `map_to_id` is true, categorical predictions will be the original strings (e.g., "Product\_A"). If false, they will be integers (e.g., 42).
      * Real-valued predictions are denormalized back to their original scale when preprocessing used `normalize_real_columns: true`; otherwise they are returned unchanged.

2.  **Probabilities:** `outputs/probabilities/[MODEL_NAME]-[TARGET_COLUMN]-probabilities.[format]`

      * Generated only for categorical targets if `output_probabilities: true`.
      * Contains one column per class.

3.  **Embeddings:** `outputs/embeddings/[MODEL_NAME]-embeddings.[format]`

      * Generated only if `model_type: embedding`.
      * Contains `sequenceId`, `subsequenceId`, `itemPosition`, and columns `0`, `1`, `2`... representing the vector dimensions.

### Directory Output Mode (Sharded Inference)

When using a folder of files as input, sequifier creates a directory containing multiple sharded outputs.

**File Structure**
* **folder inputs:** `outputs/predictions/[MODEL_NAME]-predictions/[MODEL_NAME]-[CHUNK_ID]-predictions.[format]` *(Directory of files)*
* **folder inputs:** `outputs/probabilities/[MODEL_NAME]-[TARGET_COLUMN]-probabilities/[MODEL_NAME]-[CHUNK_ID]-probabilities.[format]` *(Directory of files)*
* **folder inputs:** `outputs/embeddings/[MODEL_NAME]-embeddings/[MODEL_NAME]-[CHUNK_ID]-embeddings.[format]` *(Directory of files)*


**Pipeline Note:** If you switch to `.pt` inputs, ensure your downstream scripts are configured to read from a directory of files rather than a single file. This behavior applies to predictions, probabilities, and embeddings.


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

For a single-dataset model, the command reads `logs/[MODEL_NAME]/[MODEL_NAME]-training.csv` and `logs/[MODEL_NAME]/[MODEL_NAME]-validation.csv`.

## Outputs

The interactive HTML reports are saved in the `outputs/visualization/` directory.

* **Single Model:** `outputs/visualization/[MODEL_NAME]-training-visualization.html` (Includes global losses and normalized variable validation losses if applicable).
* **Multiple Models:** `outputs/visualization/multi-model-training-visualization.html` (Side-by-side comparison of validation and training losses across all specified models).
* **Hyperparameter Search:** `outputs/visualization/[SEARCH_NAME].html` (Includes all valid runs and lists skipped invalid runs and their reasons).

If every run in a hyperparameter search is invalid, Sequifier still creates the report with an empty plot and the invalid-run list.

When comparing multiple models, their initial baseline validation loss must match unless `SKIP_BASELINE_CHECK` or `SEQUIFIER_SKIP_BASELINE_CHECK` is set.


# Hyperparameter Search Command Guide

The `sequifier hyperparameter-search` command automates the process of finding the optimal model architecture and training configuration. Powered by **Optuna**, it supports **Bayesian Optimization** (TPE), **Grid Search** (exhaustive), and **Random Sampling**. The engine manages trial execution, cooperatively prunes unpromising training runs, and supports multi-objective optimization using custom evaluation scripts.

## Usage

```console
sequifier hyperparameter-search --config-path configs/hyperparameter_search.yaml

```

## Composable Configuration Files

A hyperparameter-search entry config may set `additional_config_paths` to one
non-empty string, a list of non-empty strings, or `null`. Relative paths
resolve against the entry config's `project_root`; absolute paths are used
directly. Fragments are direct only and cannot include further fragments. They
may share nested containers when their child fields are disjoint, but duplicate
fields are errors. Composition occurs before Sequifier detects the legacy or
`base_config_path`/`overrides` search format.

## Base Training Config with Partial Overrides

Hyperparameter search can be configured as a normal training config plus a
typed partial search config. The presence of the top-level `overrides` field
selects this format. `base_config_path` is then required; a missing or empty
path is reported as an override-format configuration error rather than falling
back to the legacy format.

```yaml
base_config_path: configs/train.yaml
hp_search_name: transformer-width-search
model_config_write_path: configs/hp-search
search_strategy: bayesian
n_samples: 40

overrides:
  context_length: [64, 128]
  model_spec:
    dim_model: [128, 256]
    # n_head is inherited from train.yaml and repeated for both widths.
    dim_feedforward:
      low: 256
      high: 1024
      step: 256
  training_spec:
    batch_size: [16, 32]
    learning_rate: [0.001, 0.0005]
    # epochs and scheduler are inherited and repeated to match both rates.
```

Fields omitted from `overrides` inherit their training-config values. Inherited
sampleable fields become singleton search spaces. For index-coupled fields, an
inherited value is repeated to match the configured candidate count. If two or
more explicitly overridden members of a coupled group have different lengths,
configuration loading fails instead of repeating an explicitly configured
value.

Configured override fields replace their corresponding base fields. Lists are
replaced, never merged by index. Explicit `null` clears nullable fields. Typed
structures such as ingestion and decoding specifications are replaced as
complete values, so changing a discriminator such as `type: mlp` to
`type: linear` cannot retain fields that are invalid for the new type.

The partial `model_spec`, `training_spec`, and BERT override models use the same
list and distribution grammar documented below. Unknown fields are rejected
with their complete override path. Relative project paths continue to use the
compiled config's `project_root`; `base_config_path` itself follows normal
config-path resolution and may also be written relative to the partial config.

The original self-contained hyperparameter-search format remains supported
when `overrides` is absent.

Each sampled trial is emitted as an authored `SequifierConfig`. Generated YAML
therefore contains `context_length` and `target_offset`, but not metadata-derived
`storage_layout`, `window_view`, column groups, class counts, or ID maps. The
normal training loader resolves those values from metadata when the trial
starts.

## CLI Overrides

The search runner reads most configuration from YAML. The config-related CLI flag currently used by this command is:

| Flag | Action |
| --- | --- |
| `-sm`, `--skip-metadata` | Skips loading metadata-derived config values. All required schema fields must then be supplied directly. |

Although the parser accepts `--input-columns` and `--metadata-config-path`, the current `hyperparameter-search` command does not apply them as config overrides.

## Configuration Fields

The configuration is defined in a YAML file. To define the search space, fields accept either **lists** of categorical choices or **distribution dictionaries** defining numerical ranges.

### 1. File System & Strategy

| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. |
| `additional_config_paths` | `str`, `list[str]`, or `null` | No | `null` | Direct complementary YAML fragments. Relative paths resolve against `project_root`; recursive composition and duplicate fields are rejected. |
| `metadata_config_path` | `str` | **Yes** | - | Path to the JSON metadata file generated by `preprocess`. |
| `hp_search_name` | `str` | **Yes** | - | A prefix for the generated runs and the Optuna database (e.g., `my-search`). |
| `model_config_write_path` | `str` | **Yes** | - | Directory to save the generated config files for each run (e.g., `configs/hp_search/`). |
| `search_strategy` | `str` | No | `bayesian` | `bayesian` (TPE sampler), `sample` (Random Search), or `grid` (Brute Force Grid Search). |
| `n_samples` | `int` | *Conditional* | - | Target total number of trained runs in the persisted study. Required unless `search_strategy: grid`. |
| `seed` | `list[int]` | No | `null` | Training seeds to search. Random and Bayesian search sample from the list; grid search iterates through every value. When `null`, every run uses seed `101`. |
| `target_offset` | `int` | No | `1` | Fixed target offset for forward-looking objectives. In the partial format it inherits the authored training value unless explicitly overridden; it is not sampled. |
| `prune_trials` | `bool` | No | `true` | Enables cooperative early stopping of unpromising trials via Optuna. *Beta notice: Pruning with distributed training is currently experimental.* |
| `pruning_warmup_epochs` | `int` | No | `null` | Number of complete training epochs required before Optuna may prune a trial. Mutually exclusive with `pruning_warmup_batches`. |
| `pruning_warmup_batches` | `int` | No | `null` | Number of training batches required before Optuna may prune a trial. Mutually exclusive with `pruning_warmup_epochs`. |
| `override_input` | `bool` | No | `false` | Parsed for compatibility; the current search runner does not use this field. |
| `data_path` | `str` | No | Metadata split 0 | Path to training data. |
| `validation_data_path` | `str` | No | Metadata split 1 | Path to validation data. |
| `read_format` | `str` | No | `parquet` | Format of preprocessed training data (`parquet`, `csv`, or `pt`). |

`n_samples` is a target total across invocations of the same persisted study,
like `epochs` when resuming training. Sequifier only launches enough new runs to
reach that total and exits without training when the study already contains the
requested number of completed or pruned runs.

For `bayesian` and `sample` searches, if Optuna proposes the exact parameters of
a completed or pruned trial in the same study, Sequifier records the proposal as
a failed duplicate and immediately asks for another one without writing a run
config or starting training. Duplicate proposals do not count toward
`n_samples`, and generated training run numbers remain contiguous even though
Optuna's internal trial numbers include the rejected proposals. After 1,000
consecutive duplicate proposals, Sequifier reports that the search space may be
exhausted instead of retrying indefinitely.

### 2. Custom Evaluation & Multi-Objective Search

By default, Sequifier optimizes for the best validation loss. However, you can configure it to optimize for custom downstream metrics (like accuracy, precision, or custom business logic) by providing an evaluation script. If multiple metrics are provided, Optuna will execute a **multi-objective search** to find the Pareto front.

| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `evaluation_metrics` | `list[str]` | No | `null` | A list of metric names output by your script (e.g., `['accuracy', 'f1']`). |
| `evaluation_metric_directions` | `list[str]` | *Conditional* | `null` | Required if metrics are defined. List of `minimize` or `maximize` for each metric. |
| `evaluation_script` | `str` | *Conditional* | `null` | Required if metrics are defined. Path to a Python script that takes `[RUN_NAME]-best` as an argument and outputs a JSON file to `outputs/evaluations/` containing the metrics. |
| `evaluation_inference_config` | `str` | No | `null` | Path to an inference config. If provided, Sequifier runs inference on the newly trained model *before* calling your evaluation script. |

### 3. System & Export (Fixed Values)

These fields are constant across all search runs.

| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `export_generative_model` | `bool` | **Yes** | - | Export the standard next-token prediction model for every run. |
| `export_embedding_model` | `bool` | **Yes** | - | Export the vector embedding model for every run. |
| `embedding_layer_names` | `list[str]` | No | `[backbone.final_norm]` | Fixed ordered activation sources concatenated into every run's embedding output. Each sampled model must contain the named layers. |
| `inference_batch_size` | `int` | **Yes** | - | Batch size hardcoded into exported ONNX models. |
| `export_onnx` | `bool` | No | `true` | Export to ONNX format. |
| `export_pt` | `bool` | No | `false` | Export a self-contained PyTorch bundle (`.pt`). |
| `export_with_dropout` | `bool` | No | `false` | Export models with dropout enabled. |

### 4. Schema & Feature Selection
Sequifier allows you to search not just for model parameters, but for the best **subset of input features**.

| Field | Type | Mandatory | Description |
| --- | --- | --- | --- |
| `input_columns` | `list[list[str]]` or `null` | **Yes** | A list of input sets. E.g., `[['col1'], ['col1', 'col2']]`. Set to `null` to derive one input set from `column_data_types`. |
| `target_columns` | `list[str]` | **Yes** | The target column(s) to predict. Fixed across all runs. |
| `context_length` | `list[int]` | **Yes** | List of sequence lengths to test (e.g., `[24, 48]`). |
| `model_window_stride` | `int` or `null` | No | `null` | Fixed model-window stride used by every trial. `null` preserves one right-aligned sample per stored row. |
| `target_column_types` | `dict` | **Yes** | Map of target columns to `categorical` or `real`. |
| `categorical_decoder_special_tokens` | `dict[str, list[str]]` | No | Fixed per-target overrides selecting which of `unknown`, `other`, and `mask` occupy categorical decoder classes. |
| `special_token_ids` | `dict[str, int]` | No | Fixed special-token IDs passed to every generated training config. In the partial format these inherit from the resolved base training config and metadata. |
| `column_data_types` | `list[dict]` | *Conditional* | Required if `input_columns` varies. List of type maps corresponding to the input sets. |
| `feature_layout` | `dict` or `null` | No | Optional cartesian layout registry passed through to every sampled train config. Required when `ingestion_spec` references a structured layout. |

---

## Defining the Search Space: Lists vs. Distributions

In the architecture and training specifications below, Sequifier supports Optuna's native numerical distributions. You can define a hyperparameter as either a traditional discrete list, or as a distribution dictionary for continuous sampling.

**Format 1: Discrete List (Categorical)**

```yaml
batch_size: [16, 32, 64]

```

**Format 2: Numerical Distribution (Optuna)**
Requires a dictionary containing `low` and `high`. For floats, `step` and `log` scaling are supported. For integers, `step` and `log` are supported (but cannot be combined).

```yaml
# Float Distribution
dropout:
  low: 0.1
  high: 0.5
  step: 0.1

# Integer Distribution with Log Sampling
dim_feedforward:
  low: 64
  high: 512
  log: true

```

### 5. Model Architecture Sampling (`model_hyperparameter_sampling`)

| Field | Type | Mandatory | Description |
| --- | --- | --- | --- |
| `dim_model` | `list[int]` | **Yes** | Internal dimension of the Transformer. |
| `max_context_length` | `int` | No | Maximum context supported by every sampled backbone. Defaults to `2048`. |
| `backbone_id` | `str` | No | Prefix for generated repository `backbone_id` values. The exact architecture fingerprint is appended automatically. |
| `num_layers` | `list` or `Distribution` | **Yes** | Number of layers. |
| `n_head` | `list[int]` | **Yes** | Number of attention heads. |
| `dim_feedforward` | `list` or `Distribution` | **Yes** | Feedforward network dimension. |
| `ingestion_spec` | `dict`, `list[dict]`, or `null` | No | Fixed or dim-model-paired ingestion config. A dict may be one ingestion definition or a mapping of named branches. If a list is provided, it must have the same length as `dim_model` and is paired by index. Defaults to `{type: direct_embed, output_dim: dim_model}`. Any required projection is owned by the sampled ingestion. |
| `ingestion_merge` | `dict`, `list[dict]`, or `null` | No | Fixed or dim-model-paired merge config for named multi-ingestion configs. Supports `concat`, `sum`, `gated`, or `attention`. If omitted for multiple ingestions, defaults to `{type: concat}` and produces `dim_model`. |
| `initialization` | `dict` | No | Per-layer-group initialization configuration. Each `weight` or `bias` entry may be one fixed method or a `candidates` list sampled independently. Uses the same direct group mapping as `sequifier train`. |
| `ingestion_freezing`, `backbone_freezing`, `decoder_freezing` | `list[str]` or `null` | No | Fixed semantic groups to freeze in the corresponding component. |
| `ingestion_freezing_except`, `backbone_freezing_except`, `decoder_freezing_except` | `list[str]` or `null` | No | Fixed semantic groups to keep trainable while freezing the rest of the corresponding component. Mutually exclusive with that component's `*_freezing` field. |
| `allow_shared_ingestion_columns` | `bool` | No | Allows named ingestion streams to share flat input columns. Defaults to `false`. |
| `auxiliary_input_columns` | `list[str]` | No | Input columns that are intentionally kept in `batch.inputs` but must not be consumed by sampled ingestion configs. Defaults to `[]`. |
| `allow_unused_input_columns` | `bool` | No | Allows sampled train configs to leave input columns unused and log the unused names. Defaults to `false`; prefer `auxiliary_input_columns` for intentional auxiliary inputs. |
| `shared_layer_groups` | `list[list[int]]` | No | Fixed transformer layer-sharing groups applied to every sampled model, e.g. `[[0, 1], [6, 7]]`. Defaults to `[]`. Each group must contain at least two unique, in-range indices, and groups cannot overlap. |
| `prediction_length` | `int` | **Yes** | Number of steps to predict simultaneously. BERT trials override this to the sampled `context_length`. |
| `decoding_support` | `int`, `list[int]`, `Distribution` | No | Fixed or sampled number of consecutive transformer output positions flattened into each decoded target position. Defaults to `1`. |
| `decoding_spec` | `dict`, `list[dict]`, or `null` | No | Fixed target decoder config or a list of decoder configs sampled by index. Defaults to `{type: linear}`. Use `{type: mlp, hidden_dims: [...]}` for a shared MLP target head. |
| `activation_fn` | `list[str]` | **Yes** | E.g., `['swiglu', 'gelu']`. |
| `attention_type` | `list[str]` | **Yes** | One or more of `mha`, `mqa`, or `gqa`. |
| `attention_output_projection` | `list[bool]` | No | Whether sampled attention blocks apply a bias-free output projection. Defaults to `[true]`; use `[true, false]` to sample both variants. |
| `n_kv_heads` | `list[int or null]` | **Yes** | Number of KV heads. Use `1` for MQA, a divisor of `n_head` for GQA, and `null` only with MHA. Invalid values are filtered for each sampled `n_head`. |
| `normalization` | `list[str]` | **Yes** | E.g., `['rmsnorm']`. |
| `norm_first` | `list[bool]` | **Yes** | Pre-LN vs Post-LN. |
| `positional_encoding` | `list[str]` | **Yes** | One or more of `learned`, `rope`, `range`, or `sinusoidal`. Temporal position handling is part of the sampled backbone. |
| `rope_theta` | `list` or `Distribution` | **Yes** | Base frequency for RoPE. |

`ingestion_spec` accepts the same ingestion definitions as `sequifier train`.
For `temporal_conv`, this includes `base_ingestion` (`direct_embed` or
`pass_through`), `post_conv_norm` (`layer_norm`, `rmsnorm`, or `none`), and
`orientation` (`within_item_position` or `within_column`). The default
`within_item_position` normalizes feature channels independently at each time
step; `within_column` normalizes each feature channel across context positions.
`base_ingestion: pass_through` is useful for raw real-valued temporal features:
the first Conv1D maps from the raw feature width to the branch `output_dim`.

`decoding_spec` accepts the same target decoder definitions as `sequifier train`,
including optional decoder-hidden-kernel regularization via `hidden_weight_l2`
on `mlp` branches. A list samples one complete decoder definition per trial.
`decoding_support` accepts a fixed integer, categorical list, or integer
distribution. When it is larger than `1`, sampled configs must still satisfy
`prediction_length <= context_length - decoding_support + 1`.

Initialization methods can be held fixed or sampled as complete method
configurations. Candidate lists are configured at the individual `weight` or
`bias` entry; there is no additional `overrides` level:

```yaml
model_hyperparameter_sampling:
  initialization:
    decoder.output:
      weight:
        candidates:
          - {method: normal, mean: 0.0, std: 0.02}
          - xavier_uniform
      bias: zeros
    attention.qkv:
      weight: preserve
```

Here Optuna samples the decoder output weight method while the decoder bias and
attention weights remain fixed. Candidate lists on different groups and
parameter kinds are sampled independently and contribute their cartesian
product to grid-search sizing. Each sampled training config contains only the
selected concrete methods, in the regular expanded and canonical
`model_spec.initialization` format. String shorthand is accepted whenever a
method's required arguments are satisfied by defaults.

Layer freezing is fixed for all trials rather than sampled. Use the component
fields at the same level as `initialization`, for example:

```yaml
model_hyperparameter_sampling:
  backbone_freezing_except:
    - attention.output
    - normalization
  decoder_freezing: []
```

These fields use the same semantics, validation, and layer-group names as
`sequifier train`. Only one of `*_freezing` and `*_freezing_except` may be
non-null for a given component.

### 6. Training Hyperparameters (`training_hyperparameter_sampling`)
Most fields here are lists for sampling, but some are scalar values fixed for all runs.
| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `device` | `str` | **Yes** | - | The device to train on (e.g., `cuda`). |
| `learning_rate` | `list[float]` | **Yes** | - | List of learning rates. Linked to `epochs` and `scheduler`. |
| `epochs` | `list[int]` | **Yes** | - | Epochs to train. Paired with `learning_rate`. |
| `scheduler` | `list[dict]` | **Yes** | - | List of scheduler configs. |
| `training_objective` | `list[str]` or `str` | No | `['causal']` | Objectives to sample from: `causal`, `bert`, `final_value`, or `next_occurrence`. |
| `batch_size` | `list` or `Distribution` | **Yes** | - | Batch sizes to test. |
| `accumulation_steps` | `list` or `Distribution` | **Yes** | - | Gradient accumulation steps. |
| `gradient_clip` | `float`, `null`, `list`, or `Distribution` | No | `null` | Fixed or sampled maximum gradient norm. A list may include `null` to sample disabled clipping; float distributions sample enabled clipping thresholds. |
| `dropout` | `list` or `Distribution` | No | `[0.0]` | Backbone dropout probabilities. Each value contributes to the exact architecture fingerprint. |
| `criterion` | `dict` | **Yes** | - | Map of target columns to loss functions. |
| `bert_spec` | `dict` | Conditional | `null` | Required if `training_objective` includes `bert`; samples BERT masking settings. |
| `next_occurrence_config` | `dict` | Conditional | `null` | Required if `training_objective` includes `next_occurrence`; configures the categorical target column and target values. |
| `optimizer` | `list[dict]` | **Yes** | - | List of optimizer configs. |
| `resume` | `dict` or `null` | No | `null` (`policy: never`) | Run-checkpoint resume policy and optional explicit checkpoint path. Omitting it or setting it to `null` disables resume. Sampled backbones are not published. |
| `save_interval_epochs` | `int` | **Yes** | - | Checkpoint save frequency. |
| `scheduler_step_on` | `str` | No | `epoch` | When to step the scheduler: `epoch` or `batch`. |
| `save_latest_interval_minutes`| `float`| No | `null` | Time interval to overwrite a "latest" checkpoint. |
| `save_interval_minutes` | `float` | No | `null` | Time interval to save a unique, batch-specific checkpoint. |
| `save_interval_batches` | `int` | No | `null` | Batch interval to save a unique, batch-specific checkpoint. |
| `save_interval_val_loss` | `bool` | No | `true` | Whether to calculate validation loss at the moment of the batch interval save. |
| `calculate_validation_loss_on_initialization` | `bool` | No | `false` | Determines if a validation pass runs before epoch 1 begins. Standard `train` defaults this field to `true`. |
| `log_interval` | `int` | No | `10` | Structured training metric frequency (batches). |
| `class_share_log_columns`| `list[str]`| No | `[]` | Columns whose predicted validation distributions are recorded in the class-share CSV. |
| `early_stopping_epochs`| `int` | No | `null` | Stop if validation metric doesn't improve. |
| `num_workers` | `int` | No | `0` | Data loading subprocesses. |
| `loss_weights` | `dict` | No | `null` | Weights for multi-objective loss. |
| `class_weights` | `dict` | No | `null` | Weights for imbalanced classes. |
| `world_size` | `int` | No | `1` | Number of processes for distributed training. |
| `backend` | `str` | No | `nccl` | The distributed training backend to use (e.g., `nccl` for GPUs). Only relevant if `distributed: true`. |
| `device_max_concat_length` | `int` | No | `12` | Controls recursive tensor concatenation to prevent CUDA kernel limits. |
| `max_ram_gb` | `int` or `float`| No | `16` | RAM limit (GB) for the cache when using lazy loading. |
| `load_full_data_to_ram` | `bool` | No | `true` | If `false`, uses lazy loading (requires `read_format: pt` or `read_format: parquet`). |
| `distributed` | `bool` | No | `false`| Enable multi-GPU training (DDP or FSDP). Requires `read_format: pt` or `read_format: parquet` and folder-style sharded data. |
| `layer_type_dtypes` | `dict` | No | `null` | Map of layer types (`linear`, `embedding`, `conv`, `norm`, `decoder`) to dtypes (`float32`, `float16`, `bfloat16`, `float64`, `float8_e4m3fn`, `float8_e5m2`). Must be `null` with FSDP. |
| `layer_autocast` | `bool` | No | `false` | Enable `torch.autocast`. |
| `data_parallelism` | `Optional[str]` | No | `null` | Set data parallelism approach, one of `DDP` and `FSDP`. Required when `distributed: true`. |
| `fsdp_cpu_offload` | `Optional[bool]` | No | `null` | Must be explicitly `true` or `false` if data\_parallelism is 'FSDP'. |
| `torch_compile` | `str` | No | `outer` | Controls torch.compile. Options are "outer", "inner", or "none". |
| `float32_matmul_precision` | `str` | No | `highest` | Sets the internal PyTorch matmul precision. Options are "highest", "high", or "medium". |

-----

## Parameter Linkage vs. Independence

To prevent mathematical incompatibilities (e.g., dimension mismatches) and illogical training schedules, the hyperparameter search does **not** perform a simple Cartesian product of every field. Instead, specific parameters are **linked by index**, while others remain **independent**.

### 1\. Linked Parameters (Coupled by List Index)

If you provide a list of $N$ values for an anchor parameter, you **must** provide a list of $N$ values for its linked parameters. The search will strictly pair index $i$ of the anchor with index $i$ of the linked field.

| Group | Anchor Field | Linked Fields (Must match index) | Reason for Linkage |
| :--- | :--- | :--- | :--- |
| **Model Backbone** | `dim_model` | `n_head`<br>`ingestion_spec` when provided as a list<br>`ingestion_merge` when provided as a list | $d_{model}$ determines transformer width and must be divisible by the number of heads. Ingestion and merge lists intentionally select different complete frontends by width; fixed frontends are reused and projected automatically. |
| **Training Schedule** | `learning_rate` | `epochs`<br>`scheduler` | The magnitude of the learning rate often dictates how many epochs are needed. Schedulers often require `T_max` to match `epochs`. |
| **Data Schema** | `input_columns` | `column_data_types` | Different subsets of columns require specific data type definitions. |

> **Example:**
> If `dim_model: [64, 128]` and `n_head: [4, 8]`:
>
>   * **Run A** uses `dim_model=64` AND `n_head=4`.
>   * **Run B** uses `dim_model=128` AND `n_head=8`.
>   * *It will NOT attempt `dim_model=64` with `n_head=8`.*

### 2\. Independent Parameters (Cartesian Product)

All other parameters are considered **Independent**. Sequifier will test every value in these lists against every combination of the linked groups above.

  * **Model:** `num_layers`, `dim_feedforward`, `activation_fn`, `normalization`, `norm_first`, `positional_encoding`, `attention_type`, `attention_output_projection`, `rope_theta`.
  * **Training:** `training_objective`, `batch_size`, `dropout`, `gradient_clip`, `accumulation_steps`, `optimizer`.
  * **Data:** `context_length`.

`shared_layer_groups` is fixed for every sampled model rather than sampled as a
list of alternatives. If `num_layers` is sampled, make sure every configured
sharing group is valid for every possible sampled layer count.

### 3\. Special Case: `n_kv_heads`

`n_kv_heads` is sampled independently after filtering out values that do not divide the selected `n_head`. Ensure the remaining values are compatible with `attention_type`: `mqa` requires `n_kv_heads: 1`, `gqa` requires a non-null divisor of `n_head`, and `mha` accepts `null` or `n_head`.

-----

## Key Trade-offs and Decisions

### 1. `search_strategy`: `bayesian` vs. `grid` vs. `sample`

  * **`bayesian` (Default - TPE Sampler):**
      * *How it works:* Tree-structured Parzen Estimator (TPE). Learns from past trials to guess which hyperparameter regions are most promising.
      * *Pros:* Vastly more efficient than grid or random search, making it the industry standard for neural network tuning.
  * **`grid` (Brute Force):**
      * *How it works:* Generates every possible combination of all provided lists.
      * *Pros:* Exhaustive.
      * *Cons:* Exponential explosion. Does not support Distribution dictionaries (cannot discretize continuous boundaries automatically).
  * **`sample` (Random Search):**
      * *How it works:* Randomly draws from the provided ranges.


### 2\. Feature Selection (`input_columns`)

Sequifier uniquely allows you to treat "data" as a hyperparameter.

  * **Usage:** Provide a list of lists.
      * Run 1 might use `['sales', 'day_of_week']`
      * Run 2 might use `['sales', 'day_of_week', 'promotion_flag']`
  * **Benefit:** Helps identify if adding extra features (which increases model size and training time) actually yields better performance or simply adds noise.


### 3. Cooperative Trial Pruning (`prune_trials: true`)

Optuna monitors intermediate validation loss at validation loss calculation, which is every epoch and optionally every configured number of minutes. If the trajectory of the current run is definitively worse than previously completed trials, the searcher will issue a `SIGTERM` signal to the subprocess, aborting the run early.

Set either `pruning_warmup_epochs` or `pruning_warmup_batches` to defer pruning. They are mutually exclusive, and configuration validation rejects setting both. `pruning_warmup_epochs: 10` allows the first pruning decision after the epoch-10 validation result. Batch warm-up uses the training `global_step`, so `pruning_warmup_batches: 1000` allows pruning at the first validation report at or after batch 1000. Intermediate validation losses are still reported to Optuna during either warm-up. When both settings are omitted, immediate pruning remains enabled, including pruning from the initial epoch-0 validation result.

* *Pros:* Saves massive amounts of compute time.
* *Cons:* Can occasionally prune a "late bloomer" model.

### 4. Multi-Objective Search (Pareto Front)

If you define multiple metrics in `evaluation_metrics` (e.g., you want to maximize `accuracy` but also minimize `latency`), Sequifier creates a multi-objective Optuna study with the configured sampler and reports the **Pareto Front**: a set of best models where no metric can be improved without degrading another.

## Outputs

1. **Optuna Database:** Located at `state/optuna/[hp_search_name].db`.
      * A portable SQLite database containing the entire history of the study, enabling you to pause and resume the search at any time, or hook it into Optuna Dashboard (`optuna-dashboard sqlite:///state/optuna/...`).
2. **Generated Configs:** Located in `model_config_write_path` (e.g., `configs/hp_search/`).
      * Valid, standalone `train.yaml` files generated for each trial.
3. **Logs:** Located in `logs/`.
      * Includes operational logs and rank-0 structured training, validation, and class-share CSV files under `logs/[RUN]/`. Optuna tails `[RUN]-validation.csv` for intermediate validation loss.
4.  **Models & Checkpoints:**
      * Saved in `models/` and `checkpoints/` with filenames including the run number (for example, `models/my-search-run-5-best.onnx` and `checkpoints/runs/my-search-run-5/my-search-run-5-latest.pt`).
5. **Evaluations (Optional):**
      * Saved in `outputs/evaluations/[RUN_NAME]-best.json` if an evaluation script was utilized.


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

In your `train.yaml`, set the top-level `read_format` to match the preprocessing output and update the `training_spec` block:

```yaml
read_format: pt # or parquet for beta sharded Parquet loading

training_spec:
  distributed: true
  data_parallelism: 'FSDP' # or 'DDP'
  fsdp_cpu_offload: false   # omit if using 'DDP'; set true to offload FSDP parameters to CPU RAM
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

* **Batch Size:** The `batch_size` in your `train.yaml` is the **per-GPU** batch size. If your `batch_size` is 100, and your `world_size` is 32, your effective global batch size is 3,200.
* **Learning Rate:** You may need to scale your `learning_rate` up if you drastically increase your global batch size via distributed training.
* **Data Access:** All nodes must have access to the same shared filesystem (e.g., NFS, GPFS) where the `project_root` and the sharded preprocessing output are stored.
