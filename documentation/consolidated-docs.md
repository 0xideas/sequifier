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

`make` sets up a new sequifier project in a new folder, `preprocess` preprocesses the data from the input format into subsequences of a fixed length, `train` trains a model on the preprocessed data, `infer` generates predictions, probabilities, or embeddings from data in the preprocessed format, `hyperparameter-search` executes multiple training runs using Optuna to find optimal configurations, and `visualize-training` parses training logs to generate interactive HTML plots of your loss curves.

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

3.  cd into the `YOUR_PROJECT_NAME` folder, create a `data` folder and add your data and adapt the config file `preprocess.yaml` in the configs folder to take the path to the data
4.  run

```console
sequifier preprocess
```

5.  the preprocessing step outputs a metadata config at `configs/metadata_configs/[FILE NAME]`. Adapt the `metadata_config_path` parameter in `train.yaml` and `infer.yaml` to the path `configs/metadata_configs/[FILE NAME]`
6.  Adapt the config file `train.yaml` to specify the transformer hyperparameters you want and run


```console
sequifier train
```

7.  adapt `data_path` in `infer.yaml` to one of the files output in the preprocessing step
8.  run


```console
sequifier infer
```

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/[EXPORTED_MODEL_BASENAME]-predictions.[FORMAT]`, for example `outputs/predictions/sequifier-your-model-best-10-predictions.csv`


## Other Features

### Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:

- Training: Set export_embedding_model: true in the training config.
- Inference: Set model_type: embedding in the inference config.

Technical Details: The generated embedding has dimensionality `dim_model` and consists of the final hidden state (activations) of the transformer's last layer corresponding to the last token in the sequence. Because the model is trained on a causal objective, this is a "forward-looking" embedding: it is optimized to compress the sequence history into a representation that maximizes information about the future state of the data.

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

The `sequifier preprocess` command transforms raw tabular data (CSV or Parquet) into the specific sequence format required for training transformer sequence models. It handles windowing, data splitting (train/validation/test), categorical encoding, and numerical standardization.

## Usage

```console
sequifier preprocess --config-path configs/preprocess.yaml
```

## CLI Overrides

Values passed on the command line override the YAML before validation.

| Flag | Overrides / Action |
| :--- | :--- |
| `-r`, `--randomize` | Generates a random `seed`. Only affects between_sequence split_method |
| `-dp`, `--data-path` | Overrides `data_path`. |
| `-sc`, `--selected-columns` | Overrides `selected_columns` with a space-separated list. Use `None` to process all columns. |

## Configuration Fields

The configuration is defined in a YAML file (e.g., `preprocess.yaml`). Below are the available fields, their requirements, and their functions.

### 1\. File System & Input/Output

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `data_path` | `str` | **Yes** | - | Path to the raw input file or folder. |
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
| `column_types` | `dict[str, str]` | No | `null` | Optional output dtype map for processed columns, such as `Float32`, `Float64`, `Int32`, or `Int64`. If set, every processed column must be included. Parquet uses one unified sequence dtype; `pt` writes each variable to its configured tensor dtype. |
| `max_rows` | `int` | No | `null` | Limits processing to the first N rows. Useful for rapid debugging. |
| `metadata_config_path` | `Optional[str]` | No | `null` | use a preexisting metadata config path for tokenizing discrete columns and standardising real-valued columns |
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
  * **Hybrid Approach:** It is common practice to set a large stride for the training and validation splits (index 0) to reduce the size on disk of the dataset, and a stride=1 for the test split to evaluate the model on each point in the test set. This supposes that the test split value is low.
      * *Example:* `stride_by_split: [24, 24, 1]` (assuming `stored_context_width: 49`).

### 3\. `subsequence_start_mode`: `distribute` vs `exact`

  * **`distribute` (Default):** The algorithm adjusts the start indices slightly to minimize the overlap of the final subsequence with the previous one, ensuring the data covers the full sequence length as evenly as possible. Recommended for most use cases.
  * **`exact`:** Strictly enforces the stride. If the sequence length minus the window size isn't perfectly divisible by the stride, this will raise an error. Use this only if mathematical precision of the sliding window is strictly required by your downstream application or evaluation code.

### 4. Advanced: Static Vocabularies (Custom ID Maps)

By default, Sequifier dynamically builds ID maps from the data found in the input file. However, in production systems, you often need a **fixed vocabulary** to ensure that ID "105" always maps to "Item_X", regardless of the daily training batch.

To use a static vocabulary:
1. Create a folder `configs/id_maps/` in your project root.
2. Add JSON files named `{COLUMN_NAME}.json`.
3. The format must be a dictionary mapping ordinary data values to integers **starting at 3**.

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
      * **Crucial:** This file contains the integer mappings for categorical variables (`id_maps`) and normalization stats for real variables (`selected_columns_statistics`).
      * **Next Step:** You must link this file path in your `train.yaml` and `infer.yaml` under `metadata_config_path`.


## 5\. Advanced: Custom ID Mapping

By default, Sequifier automatically generates integer IDs for categorical columns starting from index 3 (indices 0, 1, and 2 are reserved for `[unknown]`, `[other]`, and `[mask]`).

If you need to enforce specific integer mappings (e.g., to maintain consistency across different training runs or datasets), you can provide **precomputed ID maps**.

1.  Create a folder named `id_maps` inside your configs directory: `configs/id_maps/`.
2.  Create a JSON file named exactly after the column you want to map (e.g., `my_column_name.json`).
3.  The JSON file must contain a key-value dictionary where keys are the raw values and values are the integer IDs.

**Constraints:**
* Ordinary data IDs must start at **3**.
* IDs **0**, **1**, and **2** are reserved.

**Example `configs/id_maps/category_col.json`:**
```json
{
  "cat": 3,
  "dog": 4,
  "mouse": 5
}
```


# Train Command Guide

The `sequifier train` command initializes and trains a transformer sequence model based on the sequence data generated during the preprocessing step. It supports custom architectures (e.g., varying layers, heads, embedding sizes), several training objectives, various optimizers (including AdEMAMix), and distributed training strategies.

## Usage

```console
sequifier train --config-path configs/train.yaml
```

## Configuration Fields

The configuration is defined in a YAML file (e.g., `train.yaml`). The file is structured into root-level fields (mostly data/paths), an optional `feature_layout` annotation section, and two subsections: `model_spec` (architecture) and `training_spec` (hyperparameters).

### 1\. File System & Inputs

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `metadata_config_path`| `str` | **Yes** | - | Path to the JSON file generated by `preprocess`. E.g., `configs/metadata_configs/data.json`. |
| `model_name` | `str` | **Yes** | - | A unique identifier for this training run. Used for naming logs and output files. Must not contain the substring `embedding`. |
| `training_data_path` | `str` | No | `data/*split0*`| Path to training data. Defaults to split 0 from metadata. |
| `validation_data_path`| `str` | No | `data/*split1*`| Path to validation data. Defaults to split 1 from metadata. |
| `read_format` | `str` | No | `parquet` | Format of input data (`parquet`, `csv`, `pt`). Must match `preprocess` output. |

### 2\. Schema & Columns

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `target_columns` | `list[str]`| **Yes** | - | The specific column(s) the model should learn to predict. |
| `target_column_types`| `dict` | **Yes** | - | Map of target columns to their type: `'categorical'` or `'real'`. The key order in target_column_types must exactly match the list order in target_columns |
| `input_columns` | `list[str]` or `null`| **Yes** | `null` | Subset of columns to use as input features. Set to `null` to use all columns available in metadata. |
| `feature_layout` | `dict` or `null` | No | `null` | Optional annotation registry for structured flat input columns. It does not change preprocessing output or stored files. |
| `context_length` | `int` | **Yes** | - | Model input context length. It must fit inside the metadata `stored_context_width` with the stored `max_target_offset`. |
| `target_offset` | `int` | No | `1` | Future offset used for forward-looking objectives. BERT-style training forces this to `0`. |

### 3\. Model Architecture (`model_spec`)

These fields determine the size and complexity of the Transformer.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `dim_model` | `int` | **Yes** | - | The internal dimension ($d_{model}$) of the Transformer. |
| `n_head` | `int` | **Yes** | - | Number of attention heads. `dim_model` must be divisible by `n_head`. |
| `num_layers` | `int` | **Yes** | - | Number of transformer encoder layers. |
| `dim_feedforward` | `int` | **Yes** | - | Dimension of the feedforward network model ($d_{ff}$). |
| `initial_embedding_dim`| `int` | **Yes** | - | Size of initial feature embeddings. Must equal`dim_model` unless a `joint_embedding_dim` is configured. |
| `joint_embedding_dim` | `int` | No | `null` | If set, projects concatenated inputs to this dim before the transformer. If set, must equal `dim_model`. |
| `prediction_length` | `int` | **Yes** | - | Number of steps to predict simultaneously. For BERT-style training, this must equal `context_length`. |
| `feature_embedding_dims`| `dict` | No | `null` | Manual map of column names to embedding sizes. If `null`, sizes are auto-calculated. This works only if there are *only* real or *only* categorical variables, and `initial_embedding_dim` is divisible by the number of variables |
| `frontend` | `dict` | No | `{type: flat}` | Feature frontend specification. `flat` reproduces the classic per-column embedding path. `composite` can merge branches such as flat, feature-token, grouped, siamese, structured, conv, or patch frontends. |
| `activation_fn` | `str` | No | `swiglu` | Activation function: `swiglu`, `gelu`, or `relu`. |
| `attention_type` | `str` | No | `mha` | `mha` (Multi-Head), `mqa` (Multi-Query), or `gqa` (Grouped-Query). |
| `n_kv_heads` | `int` | No | `null` | Number of Key/Value heads for GQA/MQA. If `null`, defaults to `n_head` (standard MHA). |
| `positional_encoding` | `str` | No | `learned`| `learned` (Standard absolute) or `rope` (Rotary Positional Embedding). |
| `rope_theta` | `float` | No | `10000.0` | The base frequency for RoPE. Increase for long-context extrapolation. |
| `normalization` | `str` | No | `rmsnorm`| `rmsnorm` or `layer_norm`. |
| `norm_first` | `bool` | No | `true` | If `true` (Pre-LN), applies normalization before attention/FFN. More stable. |

#### Feature Layout And Frontends

`feature_layout` describes reusable structure for existing flat columns. `model_spec.frontend` chooses how the model consumes those columns. Preprocessing, datasets, and exported ONNX inputs remain flat-column based.

```yaml
feature_layout:
  version: 1
  layouts:
    order_book:
      type: dense_axes
      axis_order: [side, level, field]
      axes:
        side: [a, b]
        level: [1]
        field: [price, size]
      columns:
        a_1_price: {side: a, level: 1, field: price}
        a_1_size:  {side: a, level: 1, field: size}
        b_1_price: {side: b, level: 1, field: price}
        b_1_size:  {side: b, level: 1, field: size}

model_spec:
  frontend:
    type: composite
    branches:
      book:
        frontend:
          type: structured
          layout: order_book
          output_dim: 128
      context:
        columns: [spread, volatility]
        frontend:
          type: feature_token
          output_dim: 64
    merge:
      type: concat
      output_dim: 256
```

### 4\. Training Hyperparameters (`training_spec`)

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `training_objective` | `str` | **Yes** | - | Objective to train: `causal`, `bert`, `final_value`, or `next_occurrence`. |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `epochs` | `int` | **Yes** | - | Maximum number of training epochs. |
| `batch_size` | `int` | **Yes** | - | Samples per batch. |
| `learning_rate` | `float` | **Yes** | - | Initial learning rate. |
| `accumulation_steps` | `Optional[int]` | No | `null` | Accumulation steps between weight updates, to increase effective batch size. |
| `dropout` | `float` | No | `0.0` | Dropout probability. |
| `optimizer` | `dict` | **Yes** | - | Optimizer config. Supports `Adam`, `AdamW`, `AdEMAMix`, etc. |
| `scheduler` | `dict` | **Yes** | - | LR Scheduler config (e.g., `StepLR` or `CosineAnnealingLR`). `scheduler.step()` is only called if < total_steps, so correct configuration is essential. |
| `scheduler_step_on` | `str` | No | `epoch` | When to step the scheduler: `epoch` or `batch`. |
| `criterion` | `dict` | **Yes** | - | Map of target columns to loss functions (e.g., `CrossEntropyLoss`, `MSELoss`). |
| `bert_spec` | `dict` | Conditional | `null` | Required when `training_objective: bert`; configures masking probability, replacement distribution, and span masking. |
| `next_occurrence_config` | `dict` | Conditional | `null` | Required when `training_objective: next_occurrence`; configures the categorical target column and target values. |
| `loss_weights` | `dict` | No | `null` | Weights for combining losses if predicting multiple targets. |
| `class_weights` | `dict` | No | `null` | Weights for specific classes (useful for imbalanced datasets). |
| `save_interval_epochs` | `int` | **Yes** | - | Save a checkpoint every N epochs. |
| `save_latest_interval_minutes`| `float`| No | `null` | Time interval to overwrite a "latest" checkpoint. |
| `save_interval_minutes` | `float` | No | `null` | Time interval to save a unique, batch-specific checkpoint. |
| `save_interval_batches` | `int` | No | `null` | Batch interval to save a unique, batch-specific checkpoint. |
| `save_interval_val_loss` | `bool` | No | `true` | Whether to calculate validation loss at the moment of the batch interval save. |
| `calculate_validation_loss_on_initialization` | `bool` | No | `true` | Determines if a validation pass runs before epoch 1 begins. |
| `early_stopping_epochs`| `int` | No | `null` | Stop training if validation loss doesn't improve for N epochs. |
| `log_interval` | `int` | No | `10` | Print training logs every N batches. |
| `class_share_log_columns`| `list[str]`| No | `[]` | Columns for which to log the predicted class distribution in validation. |
| `enforce_determinism` | `bool` | No | `false` | Force deterministic algorithms (slower, but reproducible). |
| `num_workers` | `int` | No | `0` | Number of subprocesses for data loading. |
| `max_ram_gb` | `float` | No | `16` | RAM limit (GB) for the cache when using lazy loading. |
| `world_size` | `int` | No | `1` | Number of distributed processes/GPUs. |
| `backend` | `str` | No | `nccl` | The distributed training backend to use (e.g., `nccl` for GPUs, `gloo` for CPUs). Only relevant if `distributed: true`. |
| `device_max_concat_length`| `int` | No | `12` | Controls recursive tensor concatenation to prevent CUDA kernel limits on specific hardware. Lower this if you encounter "CUDA error: too many resources requested for launch". |
| `continue_training` | `bool` | No | `true` | Load model weights and optimizer state from the latest checkpoint and continue training. |
| `distributed` | `bool` | No | `false`| Enable multi-GPU training (DDP or FSDP). Requires `read_format: pt` or `read_format: parquet` and folder-style sharded data. |
| `load_full_data_to_ram`| `bool` | No | `true` | If `false`, uses lazy loading (requires `read_format: pt` or `read_format: parquet`). |
| `layer_type_dtypes` | `dict` | No | `null` | Map of layer types (`linear`, `embedding`, `norm`, `decoder`) to dtypes (`float32`, `float16`, `bfloat16`, `float64`, `float8_e4m3fn`, `float8_e5m2`). Used for mixed-precision/quantization. |
| `layer_autocast` | `bool` | No | `true` | If `true`, enables `torch.autocast` for automatic mixed precision training. |
| `data_parallelism` | `Optional[str]` | No | `null` | Set data parallelism approach, one of `DDP` and `FSDP`. |
| `fsdp_cpu_offload` | `Optional[bool]` | No | `null` | Must be explicitly true or false if `data_parallelism` is `FSDP`. Must be `null` otherwise. |
| `torch_compile` | `str` | No | `outer` | Controls torch.compile. Options are `outer` (compiles the whole model), `inner` (compiles individual transformer layers, for FSDP), or `none` (no compilation). |
| `float32_matmul_precision` | `str` | No | `highest` | Sets the internal PyTorch matmul precision. Options are `highest`, `high`, or `medium`. |


### 5\. System & Export

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `export_generative_model`| `bool` | **Yes** | - | Export the standard model for next-token prediction. |
| `export_embedding_model` | `bool` | **Yes** | - | Export a model that outputs the vector embedding of the sequence. |
| `inference_batch_size` | `int` | **Yes** | - | Batch size hardcoded into the exported ONNX model. |
| `seed` | `int` | No | `1010` | Root-level random seed for reproducible training. |
| `export_onnx` | `bool` | No | `true` | Export model as `.onnx` for high-performance inference. |
| `export_pt` | `bool` | No | `false`| Export model as `.pt` (PyTorch state dict). |
| `export_with_dropout` | `bool` | No | `false`| Export model with dropout enabled (useful for Monte Carlo Dropout inference). |

-----

## CLI Overrides

Values passed on the command line override the YAML before validation.

| Flag | Overrides / Action |
| :--- | :--- |
| `-r`, `--randomize` | Generates a random `seed`, taking precedence over `--seed`. |
| `-ic`, `--input-columns` | Overrides `input_columns` with a space-separated list. Use `None` to derive all columns from metadata. |
| `-mc`, `--metadata-config-path` | Overrides `metadata_config_path`. |
| `-sm`, `--skip-metadata` | Skips loading metadata-derived config values. All required schema fields must then be supplied directly. |
| `-mn`, `--model-name` | Overrides `model_name`. |
| `-s`, `--seed` | Overrides the root-level `seed`, unless `--randomize` is also set. |

## Key Trade-offs and Decisions

### 1\. Data Loading Strategy (`load_full_data_to_ram`)

* **`true` (Default):** Loads the entire dataset into system RAM.
      * *Mechanism:* Uses a native PyTorch IterableDataset that handles global shuffling and pre-collates batches directly in memory.
      * *Pros*: Fastest training speed.
      * *Cons*: Limited by physical RAM. If the dataset is 64GB and you have 32GB RAM, this will crash.
  * **`false` (Lazy Loading):** Loads individual files on-demand during training.
      * *Requirements:* `read_format` must be `parquet` or `pt`.
      * *Mechanism:* Uses an `IterableDataset` with cross-file buffering to stream pre-processed chunked files sequentially, automatically calculating exact sample boundaries across GPU ranks and workers.
      * *Pros:* Can train on datasets much larger than RAM, safely supporting DDP/FSDP synchronization.
      * *Cons:* Slight I/O overhead depending on disk speed. Increase `num_workers` to mitigate this. **Note for Parquet users:** Lazy loading distributed Parquet files is currently in **Beta** and may cause high CPU overhead or deadlocks on large multi-GPU nodes. For distributed lazy loading, `read_format: pt` is strongly recommended.

### 2\. Attention Mechanism (`attention_type` & `n_kv_heads`)

  * **`mha` (Multi-Head Attention - Default):** Standard Transformer attention. Best for general accuracy but memory intensive for the KV cache during inference.
  * **`mqa` (Multi-Query Attention):** Shares a single Key/Value head across all Query heads (`n_kv_heads: 1`). Significantly reduces memory usage during inference and speeds up generation.
  * **`gqa` (Grouped-Query Attention):** A middle ground. Set `n_kv_heads` to a value that divides `n_head` (e.g., 8 heads, 2 KV heads).

### 3\. Activation Function (`activation_fn`)

  * **`swiglu` (Default):** Generally offers better convergence and performance than ReLU or GeLU in modern LLMs (e.g., Llama 2/3).
  * **`gelu` / `relu`:** Standard older activations. Use these if you need strictly smaller models or compatibility with older inference runtimes.

### 4\. BERT Masking Objective (`bert_spec`)

Use `training_objective: bert` when you want the model to reconstruct masked positions from both left and right context instead of predicting future positions. This is useful for denoising, representation learning, or embedding models where the full observed sequence should inform each token.

When `training_objective: bert` is set:

* `bert_spec` is required.
* `target_offset` is forced to `0`.
* `model_spec.prediction_length` must equal `context_length`.
* Preprocessing should normally use `max_target_offset: 0`, because BERT uses same-width inputs and targets rather than future target capacity.
* Loss is calculated only on valid positions selected by the generated BERT mask.

`bert_spec` has three required fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `masking_probability` | `float` | Fraction of valid positions selected for the BERT prediction task. Must be `> 0.0` and `<= 1.0`. |
| `replacement_distribution` | `dict` | Probabilities for how selected positions are corrupted before being passed to the model. `masked + random + identical` must sum to `1.0`. |
| `span_masking` | `dict` | Distribution used to sample non-overlapping mask span lengths. |

`replacement_distribution` supports:

* `masked`: replace selected categorical values with `[mask]` and selected real values with `0.0`.
* `random`: replace selected categorical values with a random ordinary class ID and selected real values with standard normal noise.
* `identical`: leave selected input values unchanged while still asking the model to predict them.

`span_masking` must include a `type` discriminator. Supported distributions are:

| Type | Parameters | Notes |
| :--- | :--- | :--- |
| `GeometricDistribution` | `p` | Samples span lengths from a geometric distribution. Higher `p` means shorter spans. |
| `NormalDistributionDiscretizedFloor` | `mean`, `standard_deviation` | Samples rounded normal span lengths, clamped to at least 1. |
| `LogNormalDistributionDiscretizedFloor` | `mean`, `standard_deviation` | Samples rounded log-normal span lengths, clamped to at least 1. |
| `PoissonDistributionFloor` | `rate` | Samples Poisson span lengths plus 1. |

Example:

```yaml
context_length: 48

model_spec:
  prediction_length: 48

training_spec:
  training_objective: bert
  bert_spec:
    masking_probability: 0.15
    replacement_distribution:
      masked: 0.8
      random: 0.1
      identical: 0.1
    span_masking:
      type: GeometricDistribution
      p: 0.2
```

### 5\. Distributed Training (`distributed`)

If you have multiple GPUs:

1.  Set `distributed: true` in `training_spec`.
2.  **Crucial:** You must have run `preprocess` with `merge_output: false`.
3.  Set `world_size` to the number of GPUs.
4.  Set `data_parallelism` to `DDP` for `DistributedDataParallel` training or `FSDP` for `FullyShardedDataParallel` training.
5.  Set `torch_compile` to `inner` or `none` when training with `FSDP`, and to `outer` or `none` when training with `DDP`.

### 6\. Export Formats (`export_generative_model` vs `export_embedding_model`)

  * **Generative:** Exports the full model head. Use this if you want to predict the next token/value (forecasting, generation).
  * **Embedding:** Exports a model that outputs the vector representation of the final token *before* the decoding layer. Use this for clustering, similarity search, or feeding dense features into downstream models (e.g., XGBoost).

### 7\. Mixed Precision & Weight Types (`layer_type_dtypes`)

Sequifier v1.0.0.4 introduces advanced controls for numerical precision. This allows you to trade off numerical accuracy for significantly reduced memory usage (VRAM) and faster training speeds.

  * **`layer_autocast`**: Enables PyTorch's Automatic Mixed Precision (AMP).
  * **`layer_type_dtypes`**: Manually casts specific model components to lower precision formats (e.g., `bfloat16`, `float8`).

#### A. Automatic vs. Manual Precision

  * **Autocast (`layer_autocast: true`):**
      * *How it works:* PyTorch automatically determines which operations are safe to run in half-precision (`float16` or `bfloat16`) and which require full precision (`float32`).
      * *Pros:* Safest approach. Maintains model stability while speeding up math operations.
      * *Cons:* Weights often remain stored in `float32` in memory, only downcasting for the calculation. Less VRAM savings than manual casting.
  * **Manual Casting (`layer_type_dtypes`):**
      * *How it works:* You explicitly force specific layers (e.g., `linear`) to store their weights in lower precision.
      * *Pros:* Maximizes VRAM savings. Allows training much larger models on the same hardware.
      * *Cons:* Higher risk of numerical instability (NaNs) or divergence if the precision is too low for the task.

#### B. Data Type Selection (`float16` vs. `bfloat16` vs. `float64` vs. `float8`)

When defining `layer_type_dtypes`, you must choose the right format for your hardware:

| Type | Description | Hardware Support | Trade-off |
| :--- | :--- | :--- | :--- |
| **`float32`** | Standard single precision. | All GPUs. | High precision, high memory usage. The safe default. |
| **`float16`** | Half precision. | Volta (V100) & newer. | Fast, but has a small dynamic range. Prone to "overflow" (Infinity values). Sequifier automatically enables a `GradScaler` to mitigate this. |
| **`bfloat16`** | Brain Floating Point. | Ampere (A100/3090) & newer. | **Recommended for modern GPUs.** Has the same dynamic range as `float32` but lower precision. Rarely overflows, requires no scaler, and is very stable. |
| **`float64`** | Double precision. | CPU and supported GPUs. | Highest precision and memory use. Useful for high-precision experiments, but usually slower. |
| **`float8_e4m3fn`** | 8-bit floating point. | Hopper (H100) & newer. | **Experimental.** Extreme speed and memory efficiency. Only useful on cutting-edge hardware; may degrade model accuracy. |
| **`float8_e5m2`** | 8-bit floating point with wider range. | Hopper (H100) & newer. | **Experimental.** Wider dynamic range than `float8_e4m3fn`, with lower mantissa precision. |

#### C. Layer Granularity

You can mix and match precision for different parts of the model using the dictionary keys:

  * **`linear`:** The bulk of the transformer parameters. Casting this to `bfloat16` or `float16` yields the biggest performance gains.
  * **`embedding`:** Stores the vector representations of inputs. Often kept in `float32` for stability, as aggressive quantization here can hurt representation quality.
  * **`norm`:** Normalization layers (RMSNorm/LayerNorm). **Strongly recommended** to keep these in `float32` to avoid exploding gradients.
  * **`decoder`:** The final prediction head. Can usually match the `linear` type.

> **Possible Configuration for A100/H100/3090/4090:**
>
> ```yaml
> layer_autocast: true
> layer_type_dtypes:
>   linear: bfloat16
>   decoder: bfloat16
>   embedding: float32
>   norm: float32
> ```
-----

## Outputs

After running `train`, the following are generated:

1.  **Models:** Located in `models/`.
      * `sequifier-[NAME]-best-[EPOCH].onnx`: The model with the lowest validation loss.
      * `sequifier-[NAME]-last-[EPOCH].onnx`: The model state at the final epoch.
      * *Note:* If `export_embedding_model: true`, you will also see files such as `sequifier-[NAME]-best-embedding-[EPOCH].onnx` or `.pt`, depending on export settings.
2.  **Checkpoints:** Located in `checkpoints/`.
      * `.pt` files containing optimizer states, allowing you to resume training later by setting `continue_training: true`.
3.  **Logs:** Located in `logs/`.
      * Detailed logs of training loss, validation loss, and learning rate per epoch/batch.


# Infer Command Guide

The `sequifier infer` command uses a trained Sequifier model (PyTorch `.pt` or ONNX `.onnx`) to generate predictions, probabilities, or vector embeddings on new data. It handles batching, data normalization (and denormalization), and supports complex inference modes like **autoregression**.

## Usage

```console
sequifier infer --config-path configs/infer.yaml
```

## CLI Overrides

Values passed on the command line override the YAML before validation.

| Flag | Overrides / Action |
| :--- | :--- |
| `-r`, `--randomize` | Generates a random `seed`, taking precedence over `--seed`. |
| `-dp`, `--data-path` | Overrides `data_path`. |
| `-ic`, `--input-columns` | Overrides `input_columns` with a space-separated list. Use `None` to derive all columns from metadata. |
| `-mc`, `--metadata-config-path` | Overrides `metadata_config_path`. |
| `-sm`, `--skip-metadata` | Skips loading metadata-derived config values. All required schema fields must then be supplied directly. |
| `-mp`, `--model-path` | Overrides `model_path`. |
| `-s`, `--seed` | Overrides `seed`, unless `--randomize` is also set. |

## Configuration Fields

The configuration is defined in a YAML file (e.g., `infer.yaml`).

### 1\. File System & Model Loading

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `data_path` | `str` | No | Metadata split 2 | Path to the input data file (`csv` or `parquet`) or folder (`pt` or `parquet`). Defaults to split 2 from metadata, or the last available split if fewer than three splits exist. |
| `model_path` | `str` or `list[str]` | **Yes** | - | Path to a specific model file, or a list of paths to process sequentially. (e.g., `models/sequifier-[NAME]-best-[EPOCH].pt`). |
| `training_config_path`| `str` | No | `configs/train.yaml`| Path to the config used to train the model. Required to reconstruct PyTorch `.pt` exports. |
| `metadata_config_path`| `str` | **Yes** | - | Path to the JSON metadata file generated during preprocessing. Used for ID mapping and normalization. |
| `read_format` | `str` | No | `parquet` | Format of input data (`csv`, `parquet`, `pt`). |
| `write_format` | `str` | No | `csv` | Format for output predictions (`csv`, `parquet`). |

### 2\. Schema & Columns

These fields tell the inference engine which columns to extract from the new data and how to interpret them.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_columns` | `list[str]` or `null`| **Yes** | - | List of feature columns. Must match the columns the model was trained on. Set to `null` to use all metadata columns. |
| `target_columns` | `list[str]`| **Yes** | - | The column(s) to predict. |
| `column_types` | `dict` | No | Metadata column types | Map of all columns to their type (e.g., `Int64`, `Float64`). Usually copied from metadata. |
| `target_column_types`| `dict` | **Yes** | - | Map of target columns to `categorical` or `real`. |

### 3\. Inference Logic & Modes

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model_type` | `str` | **Yes** | - | `generative` (predict next value) or `embedding` (extract vector representation). |
| `training_objective` | `str` | **Yes** | - | Objective used during training: `causal`, `bert`, `final_value`, or `next_occurrence`. |
| `context_length` | `int` | **Yes** | - | The model context window size. It must match the trained model view and fit inside the stored metadata capacity. |
| `target_offset` | `int` | No | `1` | Future offset used for forward-looking objectives. BERT-style inference forces this to `0`. |
| `prediction_length` | `int` | No | `1` for forward objectives; `context_length` for BERT | Number of steps to predict *simultaneously*. **Must be 1** if `autoregression: true`. |
| `inference_batch_size`| `int` | **Yes** | - | Number of sequences to process at once. |
| `autoregression` | `bool` | No | `false` | If `true`, feeds predictions back into the model to predict further into the future. |
| `autoregression_total_steps`| `int` | No | `null` | If `autoregression: true`, how many total steps to predict, starting from the *first* subsequence in the inference data. |
| `output_probabilities`| `bool` | No | `false` | If `true`, outputs the full probability distribution for categorical targets. |
| `sample_from_distribution_columns`| `Optional[list[str]]`| No | `null` | If set, the model **samples** from the predicted distribution for these columns instead of taking the top-1 (argmax). Essential for diversity in generation. |
| `map_to_id` | `bool` | No | `true` | If `true`, converts integer class predictions back to original string IDs (e.g., 0 -\> "cat"). Set to `false` for real-only targets. |
| `infer_with_dropout` | `bool` | No | `false` | If `true`, keeps dropout active during inference (useful for uncertainty estimation/Monte Carlo Dropout). For ONNX models, this is only effective if the model was exported with `export_with_dropout: true` during training. |
| `seed` | `int` | No | `1010` | Random seed for reproducibility. |

### 4\. System

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `enforce_determinism` | `bool` | No | `false` | Forces PyTorch to use deterministic algorithms. |
-----

## Key Trade-offs and Decisions

### 1\. Input Format (`read_format`)

  * **`csv`:** Best for standard inference on small data. The inferer will filter the data to `input_columns` automatically.
  * **`parquet`** Best for most use cases. Can be used with lazy loading, will use less disk space but more CPU than `pt`
  * **`pt`** Optimized for lazy loading, uses more disk space but less CPU than `parquet`

### 2\. `model_type`: `generative` vs. `embedding`

  * **`generative`:** Use this when you want to predict the next value in a sequence (forecasting, classification, next-token prediction).
      * *Output:* A file in `outputs/predictions/` containing the predicted values for specific item positions.
  * **`embedding`:** Use this when you want to represent the sequence as a fixed-size vector. This uses the output of the Transformer's last layer *before* the decoding head.
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
      * Real-valued predictions are automatically denormalized back to their original scale.

2.  **Probabilities:** `outputs/probabilities/[MODEL_NAME]-[TARGET_COLUMN]-probabilities.[format]`

      * Generated only if `output_probabilities: true`.
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

The `sequifier visualize-training` command parses the log files generated during training and hyperparameter search to create interactive Plotly HTML visualizations of the training and validation losses. It supports viewing a single model's progress or comparing multiple models side-by-side.

## Usage

```console
# Visualize a single model
sequifier visualize-training my-model-name

# Visualize multiple models side-by-side
sequifier visualize-training model-A,model-B,model-C

# Visualize models listed in a text file
sequifier visualize-training path/to/models.txt --log-scale

```

## Arguments

Unlike other commands that rely on a YAML config, `visualize-training` is configured directly via command-line arguments.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `models` | `str` | **Required** | A single model name, a comma-separated list of model names, or the path to a `.txt` file containing model names (one per line). |
| `--log-scale` | `flag` | `False` | Use a logarithmic scale on the y-axis for the loss curves. |
| `--bucket-training-batches` | `int` | `null` | Smooths the training loss curve by averaging the loss over a specified number of batches. **Must be a multiple of the logged batch interval** used during training. |
| `--project-root` | `str` | `.` | The root directory of your Sequifier project. |

## Outputs

The interactive HTML reports are saved in the `outputs/visualization/` directory.

* **Single Model:** `outputs/visualization/[MODEL_NAME]-training-visualization.html` (Includes global losses and normalized variable validation losses if applicable).
* **Multiple Models:** `outputs/visualization/multi-model-training-visualization.html` (Side-by-side comparison of validation and training losses across all specified models).


# Hyperparameter Search Command Guide

The `sequifier hyperparameter-search` command automates the process of finding the optimal model architecture and training configuration. Powered by **Optuna**, it supports **Bayesian Optimization** (TPE), **Grid Search** (exhaustive), and **Random Sampling**. The engine manages trial execution, cooperatively prunes unpromising training runs, and supports multi-objective optimization using custom evaluation scripts.

## Usage

```console
sequifier hyperparameter-search --config-path configs/hyperparameter_search.yaml

```

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
| `metadata_config_path` | `str` | **Yes** | - | Path to the JSON metadata file generated by `preprocess`. |
| `hp_search_name` | `str` | **Yes** | - | A prefix for the generated runs and the Optuna database (e.g., `my-search`). |
| `model_config_write_path` | `str` | **Yes** | - | Directory to save the generated config files for each run (e.g., `configs/hp_search/`). |
| `search_strategy` | `str` | No | `bayesian` | `bayesian` (TPE sampler), `sample` (Random Search), or `grid` (Brute Force Grid Search). |
| `n_samples` | `int` | *Conditional* | - | Number of distinct runs to execute. Required unless `search_strategy: grid`. |
| `seed` | `int` | No | `null` | Seed passed to the Optuna sampler. |
| `prune_trials` | `bool` | No | `true` | Enables cooperative early stopping of unpromising trials via Optuna. *Beta notice: Pruning with distributed training is currently experimental.* |
| `override_input` | `bool` | No | `false` | Parsed for compatibility; the current search runner does not use this field. |
| `training_data_path` | `str` | No | Metadata split 0 | Path to training data. |
| `validation_data_path` | `str` | No | Metadata split 1 | Path to validation data. |
| `read_format` | `str` | No | `parquet` | Format of preprocessed training data (`parquet`, `csv`, or `pt`). |

### 2. Custom Evaluation & Multi-Objective Search

By default, Sequifier optimizes for the best validation loss. However, you can configure it to optimize for custom downstream metrics (like accuracy, precision, or custom business logic) by providing an evaluation script. If multiple metrics are provided, Optuna will execute a **multi-objective search** to find the Pareto front.

| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `evaluation_metrics` | `list[str]` | No | `null` | A list of metric names output by your script (e.g., `['accuracy', 'f1']`). |
| `evaluation_metric_directions` | `list[str]` | *Conditional* | `null` | Required if metrics are defined. List of `minimize` or `maximize` for each metric. |
| `evaluation_script` | `str` | *Conditional* | `null` | Required if metrics are defined. Path to a Python script that takes `[RUN_NAME]-best-[EPOCH]` as an argument and outputs a JSON file to `outputs/evaluations/` containing the metrics. |
| `evaluation_inference_config` | `str` | No | `null` | Path to an inference config. If provided, Sequifier runs inference on the newly trained model *before* calling your evaluation script. |

### 3. System & Export (Fixed Values)

These fields are constant across all search runs.

| Field | Type | Mandatory | Default | Description |
| --- | --- | --- | --- | --- |
| `export_generative_model` | `bool` | **Yes** | - | Export the standard next-token prediction model for every run. |
| `export_embedding_model` | `bool` | **Yes** | - | Export the vector embedding model for every run. |
| `inference_batch_size` | `int` | **Yes** | - | Batch size hardcoded into exported ONNX models. |
| `export_onnx` | `bool` | No | `true` | Export to ONNX format. |
| `export_pt` | `bool` | No | `false` | Export to PyTorch state dict (`.pt`). |
| `export_with_dropout` | `bool` | No | `false` | Export models with dropout enabled. |

### 4. Schema & Feature Selection
Sequifier allows you to search not just for model parameters, but for the best **subset of input features**.

| Field | Type | Mandatory | Description |
| --- | --- | --- | --- |
| `input_columns` | `list[list[str]]` or `null` | **Yes** | A list of input sets. E.g., `[['col1'], ['col1', 'col2']]`. Set to `null` to derive one input set from `column_types`. |
| `target_columns` | `list[str]` | **Yes** | The target column(s) to predict. Fixed across all runs. |
| `context_length` | `list[int]` | **Yes** | List of sequence lengths to test (e.g., `[24, 48]`). |
| `target_column_types` | `dict` | **Yes** | Map of target columns to `categorical` or `real`. |
| `column_types` | `list[dict]` | *Conditional* | Required if `input_columns` varies. List of type maps corresponding to the input sets. |

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
| `num_layers` | `list` or `Distribution` | **Yes** | Number of layers. |
| `n_head` | `list[int]` | **Yes** | Number of attention heads. |
| `dim_feedforward` | `list` or `Distribution` | **Yes** | Feedforward network dimension. |
| `initial_embedding_dim` | `list[int]` | **Yes** | Feature embedding size. Usually matches `dim_model`. |
| `feature_embedding_dims` | `list[dict]` or `null` | **Yes** | List of maps for feature embedding dimensions. Use `null` only when auto-calculation is valid. |
| `joint_embedding_dim` | `list[int or null]` | **Yes** | Joint embedding size. If not null, must match `dim_model`. |
| `prediction_length` | `int` | **Yes** | Number of steps to predict simultaneously. BERT trials override this to the sampled `context_length`. |
| `activation_fn` | `list[str]` | **Yes** | E.g., `['swiglu', 'gelu']`. |
| `attention_type` | `list[str]` | **Yes** | E.g., `['mha', 'mqa']`. |
| `n_kv_heads` | `list[int or null]` | **Yes** | Number of KV heads. Use `1` for MQA, a divisor of `n_head` for GQA, and `null` only with MHA. |
| `normalization` | `list[str]` | **Yes** | E.g., `['rmsnorm']`. |
| `norm_first` | `list[bool]` | **Yes** | Pre-LN vs Post-LN. |
| `positional_encoding` | `list[str]` | **Yes** | `['learned', 'rope']`. |
| `rope_theta` | `list` or `Distribution` | **Yes** | Base frequency for RoPE. |

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
| `dropout` | `list` or `Distribution` | No | `[0.0]` | Dropout probabilities. |
| `criterion` | `dict` | **Yes** | - | Map of target columns to loss functions. |
| `bert_spec` | `dict` | Conditional | `null` | Required if `training_objective` includes `bert`; samples BERT masking settings. |
| `next_occurrence_config` | `dict` | Conditional | `null` | Required if `training_objective` includes `next_occurrence`; configures the categorical target column and target values. |
| `optimizer` | `list[dict]` | **Yes** | - | List of optimizer configs. |
| `continue_training` | `bool` | **Yes** | - | Load model weights from the latest checkpoint to resume. |
| `save_interval_epochs` | `int` | **Yes** | - | Checkpoint save frequency. |
| `scheduler_step_on` | `str` | No | `epoch` | When to step the scheduler: `epoch` or `batch`. |
| `save_latest_interval_minutes`| `float`| No | `null` | Time interval to overwrite a "latest" checkpoint. |
| `save_interval_minutes` | `float` | No | `null` | Time interval to save a unique, batch-specific checkpoint. |
| `save_interval_batches` | `int` | No | `null` | Batch interval to save a unique, batch-specific checkpoint. |
| `save_interval_val_loss` | `bool` | No | `true` | Whether to calculate validation loss at the moment of the batch interval save. |
| `calculate_validation_loss_on_initialization` | `bool` | No | `false` | Determines if a validation pass runs before epoch 1 begins. Standard `train` defaults this field to `true`. |
| `log_interval` | `int` | No | `10` | Logging frequency (batches). |
| `class_share_log_columns`| `list[str]`| No | `[]` | Columns for which to log the predicted class distribution in validation. |
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
| `layer_type_dtypes` | `dict` | No | `null` | Map of layer types (`linear`, `embedding`, `norm`, `decoder`) to dtypes (`float32`, `float16`, `bfloat16`, `float64`, `float8_e4m3fn`, `float8_e5m2`). |
| `layer_autocast` | `bool` | No | `true` | Enable `torch.autocast`. |
| `data_parallelism` | `Optional[str]` | No | `null` | Set data parallelism approach, one of `DDP` and `FSDP`. |
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
| **Model Backbone** | `dim_model` | `n_head`<br>`initial_embedding_dim`<br>`joint_embedding_dim`<br>`feature_embedding_dims` | $d_{model}$ determines embedding sizes and must be divisible by the number of heads. |
| **Training Schedule** | `learning_rate` | `epochs`<br>`scheduler` | The magnitude of the learning rate often dictates how many epochs are needed. Schedulers often require `T_max` to match `epochs`. |
| **Data Schema** | `input_columns` | `column_types` | Different subsets of columns require specific data type definitions. |

> **Example:**
> If `dim_model: [64, 128]` and `n_head: [4, 8]`:
>
>   * **Run A** uses `dim_model=64` AND `n_head=4`.
>   * **Run B** uses `dim_model=128` AND `n_head=8`.
>   * *It will NOT attempt `dim_model=64` with `n_head=8`.*

### 2\. Independent Parameters (Cartesian Product)

All other parameters are considered **Independent**. Sequifier will test every value in these lists against every combination of the linked groups above.

  * **Model:** `num_layers`, `dim_feedforward`, `activation_fn`, `normalization`, `norm_first`, `positional_encoding`, `attention_type`, `rope_theta`.
  * **Training:** `training_objective`, `batch_size`, `dropout`, `accumulation_steps`, `optimizer`.
  * **Data:** `context_length`.

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
      * Includes individual training logs and JSONL files (`sequifier-[RUN]-metrics.jsonl`) tracking the validation curve.
4.  **Models & Checkpoints:**
      * Saved in `models/` and `checkpoints/` with filenames including the run number (e.g., `models/sequifier-my-search-run-5-best-10.onnx`).
5. **Evaluations (Optional):**
      * Saved in `outputs/evaluations/[RUN_NAME]-best-[EPOCH].json` if an evaluation script was utilized.


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

When `sequifier` detects `torchrun` environment variables (like `RANK` and `WORLD_SIZE`), it bypasses its internal spawner and attaches to the distributed network established by the cluster.

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
