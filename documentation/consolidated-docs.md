<img src="./design/sequifier.png">


## What is sequifier?

Sequifier makes training and inference of powerful causal transformer models fast and trustworthy.

The process looks like this:

<img src="./design/sequifier-illustration.png">



### Value Proposition

Implementing a model from scratch takes time, and there are a surprising number of aspects to consider. The idea is: why not do it once, make it configurable, and then use the same implementation across domains and datasets.

This gives us a number of benefits:

- rapid prototyping
- configurable architecture
- trusted implementation (you can't create bugs inadvertedly)
- standardized logging
- native multi-gpu support
- native multi-core preprocessing
- scales to datasets larger than RAM
- hyperparameter search
- can be used for prediction, generation and embeddding on/of arbitrary sequences

The only requirement is having sequifier installed, and having input data in the right format.



### The Five Commands

There are five standalone commands within sequifier: `make`, `preprocess`, `train`, `infer` and `hyperparameter-search`. `make` sets up a new sequifier project in a new folder, `preprocess` preprocesses the data from the input format into subsequences of a fixed length, `train` trains a model on the preprocessed data, `infer` generates outputs from data in the preprocessed format and outputs it in the initial input format, and `hyperparameter-search` executes multiple training runs to find optimal configurations.

There are documentation pages for each command, except make:

 - [preprocess documentation](./documentation/configs/preprocess.md)
 - [train documentation](./documentation/configs/train.md)
 - [infer documentation](./documentation/configs/infer.md)
 - [hyperparameter-search documentation](./documentation/configs/hyperparameter-search.md)



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

Data of this input format can be transformed into the format that is used for model training and inference using `sequifier preprocess`, which takes this form:

|sequenceId|subsequenceId|startItemPosition|columnName|[Subsequence Length]|[Subsequence Length - 1]|...|0|
|----------|-------------|-----------------|----------|--------------------|------------------------| - |-|
|0|0|0|column1|"high"|"high"|...|"low"|
|0|0|0|column2|12.3|10.2|...|14.9|
|...|...|...|...|...|...|...|...|
|1|0|15|column1|"medium"|"high"|...|"medium"|
|1|0|15|column2|20.6|18.5|...|21.6|
|...|...|...|...|...|...|...|...|

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

9.  find your predictions at `[PROJECT ROOT]/outputs/predictions/sequifier-default-best-predictions.csv`


## Other Features

### Embedding Model

While Sequifier's primary use case is training predictive or generative causal transformer models, it also supports the export of embedding models.

Configuration:
- Training: Set export_embedding_model: true in the training config.
- Inference: Set model_type: embedding in the inference config.

Technical Details: The generated embedding has dimensionality `dim_model` and consists of the final hidden state (activations) of the transformer's last layer corresponding to the last token in the sequence. Because the model is trained on a causal objective, this is a "forward-looking" embedding: it is optimized to compress the sequence history into a representation that maximizes information about the future state of the data.



### Distributed Training

Sequifier supports distributed training using torch `DistributedDataParallel`. To make use of multi gpu support, the write format of the preprocessing step must be set to 'pt' and `merge_output` must be set to `false` in the preprocessing config.



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
  version = {v1.1.0.3},
  url = {https://github.com/0xideas/sequifier}
}
```


# Preprocess Command Guide

The `sequifier preprocess` command transforms raw tabular data (CSV or Parquet) into the specific sequence format required for training causal transformer models. It handles windowing, data splitting (train/validation/test), categorical encoding, and numerical standardization.

## Usage

```console
sequifier preprocess --config-path configs/preprocess.yaml
```

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
| `continue_preprocessing`| `bool` | No | `false` | If `true`, resumes a job that was interrupted (requires folder input). |


> **Important Constraint on `write_format`:**
>
>   * If `write_format` is **`pt`** (PyTorch tensors), `merge_output` must be **`false`**. This sharded format is **required** for distributed training on large datasets.
>   * If `write_format` is **`csv`** or **`parquet`**, `merge_output` must be **`true`**.

### 2\. Column Selection & Filtering

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `selected_columns` | `list[str]` | No | `null` | A specific list of columns to process. If `null`, all columns (except metadata) are processed. |
| `max_rows` | `int` | No | `null` | Limits processing to the first N rows. Useful for rapid debugging. |
| `use_precomputed_maps`| `list[str]` | No | `None` | If not `None`, enforces the use of precomputed maps for the variables in the list. |

### 3\. Sequence Logic & Splitting

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `seq_length` | `int` | **Yes** | - | The length of the context window (history) fed into the model. |
| `split_ratios` | `list[float]`| **Yes** | - | Proportions for data splits (e.g., `[0.8, 0.1, 0.1]` for train/val/test). Must sum to 1.0. |
| `stride_by_split` | `list[int]` | No | `[seq_length]*N` | The step size used to slide the window for each split. Corresponds to `split_ratios`. |
| `subsequence_start_mode`| `str` | No | `distribute` | Strategy for selecting start indices (`distribute` or `exact`). |

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

  * **Choose `parquet` (default):** If your dataset is small to medium (fits in RAM) and you want to inspect the preprocessed data easily using standard tools like Pandas or Polars. This produces one file per split (e.g., `data-split0.parquet`).
  * **Choose `pt`:** If your dataset is massive (larger than RAM) or you intend to use **Distributed Training** (multi-GPU). This format saves data as thousands of small PyTorch tensor files. It allows the `SequifierDatasetFromFolderLazy` to load data on demand without clogging memory.

### 2\. `stride_by_split` configuration

This controls data augmentation and redundancy.

  * **Stride = `seq_length` (Non-overlapping):** The model sees every data point exactly once as a target. Training is faster, but the model might miss patterns that cross the window boundary.
  * **Stride = 1 (Maximum Overlap):** Maximizes data volume. The model sees every possible sequence. This yields the highest accuracy but significantly increases the size of the preprocessed data and training time.
  * **Hybrid Approach:** It is common practice to set a large stride for the training and validation splits (index 0) to reduce the size on disk of the dataset, and a stride=1 for the test split to evaluate the model on each point in the test set. This supposes that the test split value is low.
      * *Example:* `stride_by_split: [24, 24, 1]` (assuming `seq_length: 48`).

### 3\. `subsequence_start_mode`: `distribute` vs `exact`

  * **`distribute` (Default):** The algorithm adjusts the start indices slightly to minimize the overlap of the final subsequence with the previous one, ensuring the data covers the full sequence length as evenly as possible. Recommended for most use cases.
  * **`exact`:** Strictly enforces the stride. If the sequence length minus the window size isn't perfectly divisible by the stride, this will raise an error. Use this only if mathematical precision of the sliding window is strictly required by your downstream application or evaluation code.

### 4. Advanced: Static Vocabularies (Custom ID Maps)

By default, Sequifier dynamically builds ID maps from the data found in the input file. However, in production systems, you often need a **fixed vocabulary** to ensure that ID "105" always maps to "Item_X", regardless of the daily training batch.

To use a static vocabulary:
1. Create a folder `configs/id_maps/` in your project root.
2. Add JSON files named `{COLUMN_NAME}.json`.
3. The format must be a dictionary mapping values to integers **starting at 2**.

> **Reserved Indices:**
> * **0**: Reserved for `unknown` (padding/missing).
> * **1**: Reserved for `other` (unseen values not in your map).
> * **2+**: Your data.

**Example `configs/id_maps/itemId.json`:**
```json
{
    "apple": 2,
    "banana": 3,
    "cherry": 4
}
-----

## Outputs

After running `preprocess`, the following are generated:

1.  **Data Files:** Located in `data/`. Depending on your configuration, these will be `[NAME]-split0.parquet` (Training), `[NAME]-split1.parquet` (Validation), etc., or folders containing `.pt` files.
2.  **Metadata Config:** Located in `configs/metadata_configs/[NAME].json`.
      * **Crucial:** This file contains the integer mappings for categorical variables (`id_maps`) and normalization stats for real variables (`selected_columns_statistics`).
      * **Next Step:** You must link this file path in your `train.yaml` and `infer.yaml` under `metadata_config_path`.


## 5\. Advanced: Custom ID Mapping

By default, Sequifier automatically generates integer IDs for categorical columns starting from index 2 (indices 0 and 1 are reserved for system use, such as "unknown" values).

If you need to enforce specific integer mappings (e.g., to maintain consistency across different training runs or datasets), you can provide **precomputed ID maps**.

1.  Create a folder named `id_maps` inside your configs directory: `configs/id_maps/`.
2.  Create a JSON file named exactly after the column you want to map (e.g., `my_column_name.json`).
3.  The JSON file must contain a key-value dictionary where keys are the raw values and values are the integer IDs.

**Constraints:**
* Integer IDs must start at **2** or higher.
* IDs **0** and **1** are reserved.

**Example `configs/id_maps/category_col.json`:**
```json
{
  "cat": 2,
  "dog": 3,
  "mouse": 4
}


# Train Command Guide

The `sequifier train` command initializes and trains a causal transformer model based on the sequence data generated during the preprocessing step. It supports custom architectures (e.g., varying layers, heads, embedding sizes), various optimizers (including AdEMAMix), and distributed training strategies.

## Usage

```console
sequifier train --config-path configs/train.yaml
```

## Configuration Fields

The configuration is defined in a YAML file (e.g., `train.yaml`). The file is structured into root-level fields (mostly data/paths) and two subsections: `model_spec` (architecture) and `training_spec` (hyperparameters).

### 1\. File System & Inputs

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `metadata_config_path`| `str` | **Yes** | - | Path to the JSON file generated by `preprocess`. E.g., `configs/metadata_configs/data.json`. |
| `model_name` | `str` | **Yes** | - | A unique identifier for this training run. Used for naming logs and output files. |
| `training_data_path` | `str` | No | `data/*split0*`| Path to training data. Defaults to split 0 from metadata. |
| `validation_data_path`| `str` | No | `data/*split1*`| Path to validation data. Defaults to split 1 from metadata. |
| `read_format` | `str` | No | `parquet` | Format of input data (`parquet`, `csv`, `pt`). Must match `preprocess` output. |

### 2\. Schema & Columns

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `target_columns` | `list[str]`| **Yes** | - | The specific column(s) the model should learn to predict. |
| `target_column_types`| `dict` | **Yes** | - | Map of target columns to their type: `'categorical'` or `'real'`. |
| `input_columns` | `list[str]`| No | All | Subset of columns to use as input features. Defaults to all available in metadata. |
| `seq_length` | `int` | **Yes** | - | Must match the `seq_length` used in preprocessing. |

### 3\. Model Architecture (`model_spec`)

These fields determine the size and complexity of the Transformer.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `dim_model` | `int` | **Yes** | - | The internal dimension ($d_{model}$) of the Transformer. |
| `n_head` | `int` | **Yes** | - | Number of attention heads. `dim_model` must be divisible by `n_head`. |
| `num_layers` | `int` | **Yes** | - | Number of transformer encoder layers. |
| `dim_feedforward` | `int` | **Yes** | - | Dimension of the feedforward network model ($d_{ff}$). |
| `initial_embedding_dim`| `int` | **Yes** | - | Size of initial feature embeddings. Usually equals `dim_model`. |
| `joint_embedding_dim` | `int` | No | `null` | If set, projects concatenated inputs to this dim before the transformer. If set, must equal `dim_model`. |
| `feature_embedding_dims`| `dict` | No | `null` | Manual map of column names to embedding sizes. If `null`, sizes are auto-calculated. This works only if there are *only* real or *only* categorical variables, and `initial_embedding_dim` is divisible by the number of variables |
| `activation_fn` | `str` | No | `swiglu` | Activation function: `swiglu`, `gelu`, or `relu`. |
| `attention_type` | `str` | No | `mha` | `mha` (Multi-Head), `mqa` (Multi-Query), or `gqa` (Grouped-Query). |
| `n_kv_heads` | `int` | No | `null` | Number of Key/Value heads for GQA/MQA. If `null`, defaults to `n_head` (standard MHA). |
| `positional_encoding` | `str` | No | `learned`| `learned` (Standard absolute) or `rope` (Rotary Positional Embedding). |
| `rope_theta` | `float` | No | `10000.0` | The base frequency for RoPE. Increase for long-context extrapolation. |
| `normalization` | `str` | No | `rmsnorm`| `rmsnorm` or `layer_norm`. |
| `norm_first` | `bool` | No | `true` | If `true` (Pre-LN), applies normalization before attention/FFN. More stable. |

### 4\. Training Hyperparameters (`training_spec`)

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `epochs` | `int` | **Yes** | - | Maximum number of training epochs. |
| `batch_size` | `int` | **Yes** | - | Samples per batch. |
| `learning_rate` | `float` | **Yes** | - | Initial learning rate. |
| `dropout` | `float` | No | `0.0` | Dropout probability. |
| `optimizer` | `dict` | No | `{'name': 'Adam'}`| Optimizer config. Supports `Adam`, `AdamW`, `AdEMAMix`, etc. |
| `scheduler` | `dict` | No | `StepLR...` | LR Scheduler config (e.g., `CosineAnnealingLR`). |
| `scheduler_step_on` | `str` | No | `epoch` | When to step the scheduler: `epoch` or `batch`. |
| `criterion` | `dict` | **Yes** | - | Map of target columns to loss functions (e.g., `CrossEntropyLoss`, `MSELoss`). |
| `loss_weights` | `dict` | No | `null` | Weights for combining losses if predicting multiple targets. |
| `class_weights` | `dict` | No | `null` | Weights for specific classes (useful for imbalanced datasets). |
| `save_interval_epochs` | `int` | **Yes** | - | Save a checkpoint every N epochs. |
| `early_stopping_epochs`| `int` | No | `null` | Stop training if validation loss doesn't improve for N epochs. |
| `log_interval` | `int` | No | `10` | Print training logs every N batches. |
| `class_share_log_columns`| `list[str]`| No | `[]` | Columns for which to log the predicted class distribution in validation. |
| `enforce_determinism` | `bool` | No | `false` | Force deterministic algorithms (slower, but reproducible). |
| `num_workers` | `int` | No | `0` | Number of subprocesses for data loading. |
| `max_ram_gb` | `float` | No | `16` | RAM limit (GB) for the cache when using lazy loading. |
| `backend` | `str` | No | `nccl` | The distributed training backend to use (e.g., `nccl` for GPUs, `gloo` for CPUs). Only relevant if `distributed: true`. |
| `device_max_concat_length`| `int` | No | `12` | Controls recursive tensor concatenation to prevent CUDA kernel limits on specific hardware. Lower this if you encounter "CUDA error: too many resources requested for launch". |
| `continue_training` | `bool` | No | `True` | Load model weights and optimizer state from laste checkpoint and continue training |

### 5\. System & Export

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `export_generative_model`| `bool` | **Yes** | - | Export the standard model for next-token prediction. |
| `export_embedding_model` | `bool` | **Yes** | - | Export a model that outputs the vector embedding of the sequence. |
| `inference_batch_size` | `int` | **Yes** | - | Batch size hardcoded into the exported ONNX model. |
| `export_onnx` | `bool` | No | `true` | Export model as `.onnx` for high-performance inference. |
| `export_pt` | `bool` | No | `false`| Export model as `.pt` (PyTorch state dict). |
| `export_with_dropout` | `bool` | No | `false`| Export model with dropout enabled (useful for Monte Carlo Dropout inference). |
| `distributed` | `bool` | No | `false`| Enable multi-GPU training (DDP). Requires `read_format: pt`. |
| `load_full_data_to_ram`| `bool` | No | `true` | If `false`, uses lazy loading (requires `read_format: pt`). |
| `layer_type_dtypes` | `dict` | No | `null` | Map of layer types (`linear`, `embedding`, `norm`, `decoder`) to dtypes (`float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`). Used for mixed-precision/quantization. |
| `layer_autocast` | `bool` | No | `true` | If `true`, enables `torch.autocast` for automatic mixed precision training. |

-----

## Key Trade-offs and Decisions

### 1\. Data Loading Strategy (`load_full_data_to_ram`)

* **`true` (Default):** Loads the entire dataset into system RAM.
      * *Mechanism:* Uses a `DistributedSampler` (global shuffling) which is statistically ideal for training.
      * *Pros:* Fastest training speed.
      * *Cons:* Limited by physical RAM. If the dataset is 64GB and you have 32GB RAM, this will crash.
  * **`false` (Lazy Loading):** Loads individual files on-demand during training.
      * *Requirements:* `read_format` must be `pt`.
      * *Mechanism:* Uses a `DistributedGroupedRandomSampler` to minimize disk seeking by processing data in file-contiguous groups.
      * *Pros:* Can train on datasets larger than RAM.
      * *Cons:* Slight I/O overhead depending on disk speed. Increase `num_workers` to mitigate this.

### 2\. Attention Mechanism (`attention_type` & `n_kv_heads`)

  * **`mha` (Multi-Head Attention - Default):** Standard Transformer attention. Best for general accuracy but memory intensive for the KV cache during inference.
  * **`mqa` (Multi-Query Attention):** Shares a single Key/Value head across all Query heads (`n_kv_heads: 1`). Significantly reduces memory usage during inference and speeds up generation.
  * **`gqa` (Grouped-Query Attention):** A middle ground. Set `n_kv_heads` to a value that divides `n_head` (e.g., 8 heads, 2 KV heads).

### 3\. Activation Function (`activation_fn`)

  * **`swiglu` (Default):** Generally offers better convergence and performance than ReLU or GeLU in modern LLMs (e.g., Llama 2/3).
  * **`gelu` / `relu`:** Standard older activations. Use these if you need strictly smaller models or compatibility with older inference runtimes.

### 4\. Distributed Training (`distributed`)

If you have multiple GPUs:

1.  Set `distributed: true` in `training_spec`.
2.  **Crucial:** You must have run `preprocess` with `write_format: pt` and `merge_output: false`.
3.  Set `world_size` to the number of GPUs.
4.  Sequifier uses `DistributedDataParallel` (DDP) to synchronize gradients across GPUs.

### 5\. Export Formats (`export_generative_model` vs `export_embedding_model`)

  * **Generative:** Exports the full model head. Use this if you want to predict the next token/value (forecasting, generation).
  * **Embedding:** Exports a model that outputs the vector representation of the final token *before* the decoding layer. Use this for clustering, similarity search, or feeding dense features into downstream models (e.g., XGBoost).

### 6\. Mixed Precision & Weight Types (`layer_type_dtypes`)

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

#### B. Data Type Selection (`float16` vs. `bfloat16` vs. `float8`)

When defining `layer_type_dtypes`, you must choose the right format for your hardware:

| Type | Description | Hardware Support | Trade-off |
| :--- | :--- | :--- | :--- |
| **`float32`** | Standard single precision. | All GPUs. | High precision, high memory usage. The safe default. |
| **`float16`** | Half precision. | Volta (V100) & newer. | Fast, but has a small dynamic range. Prone to "overflow" (Infinity values). Sequifier automatically enables a `GradScaler` to mitigate this. |
| **`bfloat16`** | Brain Floating Point. | Ampere (A100/3090) & newer. | **Recommended for modern GPUs.** Has the same dynamic range as `float32` but lower precision. Rarely overflows, requires no scaler, and is very stable. |
| **`float8_e4m3fn`** | 8-bit floating point. | Hopper (H100) & newer. | **Experimental.** Extreme speed and memory efficiency. Only useful on cutting-edge hardware; may degrade model accuracy. |

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
      * *Note:* If `export_embedding_model: true`, you will also see files suffixed with `-embedding.onnx`.
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

## Configuration Fields

The configuration is defined in a YAML file (e.g., `infer.yaml`).

### 1\. File System & Model Loading

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `data_path` | `str` | **Yes** | - | Path to the input data file (csv/parquet) or folder (if `read_format: pt`). |
| `model_path` | `str` or `list[str]` | **Yes** | - | Path to a specific model file, or a list of paths to process sequentially. (e.g., `models/sequifier-[NAME]-best-[EPOCH].pt`). |
| `training_config_path`| `str` | No | `configs/train.yaml`| Path to the config used to train the model. Required to reconstruct the model architecture. |
| `metadata_config_path`| `str` | **Yes** | - | Path to the JSON metadata file generated during preprocessing. Used for ID mapping and normalization. |
| `read_format` | `str` | No | `parquet` | Format of input data (`csv`, `parquet`, `pt`). |
| `write_format` | `str` | No | `csv` | Format for output predictions (`csv`, `parquet`). |

### 2\. Schema & Columns

These fields tell the inference engine which columns to extract from the new data and how to interpret them.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_columns` | `list[str]`| **Yes** | - | List of feature columns. Must match the columns the model was trained on. |
| `target_columns` | `list[str]`| **Yes** | - | The column(s) to predict. |
| `column_types` | `dict` | **Yes** | - | Map of all columns to their type (e.g., `Int64`, `Float64`). Usually copied from training config. |
| `target_column_types`| `dict` | **Yes** | - | Map of target columns to `categorical` or `real`. |

### 3\. Inference Logic & Modes

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `model_type` | `str` | **Yes** | - | `generative` (predict next value) or `embedding` (extract vector representation). |
| `seq_length` | `int` | **Yes** | - | The context window size. Must match training. |
| `prediction_length` | `int` | No | `1` | Number of steps to predict *simultaneously*. **Must be 1** if `autoregression: true`. |
| `inference_batch_size`| `int` | **Yes** | - | Number of sequences to process at once. |
| `autoregression` | `bool` | No | `false` | If `true`, feeds predictions back into the model to predict further into the future. |
| `autoregression_extra_steps`| `int` | No | `null` | If `autoregression: true`, how many *additional* future steps to predict beyond the first. |
| `output_probabilities`| `bool` | No | `false` | If `true`, outputs the full probability distribution for categorical targets. |
| `sample_from_distribution_columns`| `list[str]`| No | `[]` | If set, the model **samples** from the predicted distribution for these columns instead of taking the top-1 (argmax). Essential for diversity in generation. |
| `map_to_id` | `bool` | No | `true` | If `true`, converts integer class predictions back to original string IDs (e.g., 0 -\> "cat"). |
| `infer_with_dropout` | `bool` | No | `false` | If `true`, keeps dropout active during inference (useful for uncertainty estimation/Monte Carlo Dropout). |

### 4\. System & Distributed

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `distributed` | `bool` | No | `false`| Enable multi-GPU inference. Requires `read_format: pt`. |
| `world_size` | `int` | No | `1` | Number of GPUs/processes for distributed inference. |
| `num_workers` | `int` | No | `0` | Number of subprocesses for data loading. |
| `enforce_determinism` | `bool` | No | `false` | Forces PyTorch to use deterministic algorithms. |
| `load_full_data_to_ram`| `bool` | No | `true` | If `false`, uses lazy loading (requires `read_format: pt`). |

-----

## Key Trade-offs and Decisions

### 1\. `model_type`: `generative` vs. `embedding`

  * **`generative`:** Use this when you want to predict the next value in a sequence (forecasting, classification, next-token prediction).
      * *Output:* A file in `outputs/predictions/` containing the predicted values for specific item positions.
  * **`embedding`:** Use this when you want to represent the sequence as a fixed-size vector. This uses the output of the Transformer's last layer *before* the decoding head.
      * *Output:* A file in `outputs/embeddings/` containing vectors (e.g., 128 floats) for each sequence. Useful for clustering, similarity search, or downstream ML tasks.

### 2\. Autoregression (`autoregression: true`)

Standard inference predicts the next step ($t+1$) based on history ($t-n \dots t$). Autoregression allows you to predict $t+1$, append that prediction to the history, and then predict $t+2$, and so on.

  * **Pros:** Allows multi-step forecasting (e.g., predicting the next 30 days of sales) using a model trained only to predict the *next* step.
  * **Cons:** Errors accumulate. If the prediction for $t+1$ is slightly wrong, the prediction for $t+2$ relies on bad data. Inference is also significantly slower because steps must be calculated sequentially, not in parallel.
  * **Config:** Set `autoregression_extra_steps` to determine how far into the future to generate.

### 3\. Sampling vs. Argmax

  * **Default (Argmax):** The model selects the class with the highest probability. Best for accuracy metrics and "most likely" forecasts.
  * **Sampling (`sample_from_distribution_columns`):** The model picks the next token randomly based on the probability distribution.
      * *Use Case:* Creative generation or simulation where you want diversity. If `Probability(A)=0.6` and `Probability(B)=0.4`, Argmax always picks A. Sampling picks B 40% of the time.

### 4\. Input Format (`read_format`)

  * **`parquet` / `csv`:** Best for standard inference on new data files. The inferer will filter the data to `input_columns` automatically.
  * **`pt` (PyTorch Tensors):** Required for **Distributed Inference** or **Lazy Loading**. If your inference dataset is massive (terabytes), preprocess it into `.pt` chunks first, then run inference with `read_format: pt` and `distributed: true`.

-----

## Outputs

Results are saved in the `outputs/` folder within your project root.

1.  **Predictions:** `outputs/predictions/[MODEL_NAME]-predictions.[format]`

      * Standard tabular data containing `sequenceId`, `itemPosition`, and columns for your predicted targets.
      * If `map_to_id` is true, categorical predictions will be the original strings (e.g., "Product\_A"). If false, they will be integers (e.g., 42).
      * Real-valued predictions are automatically denormalized back to their original scale.

2.  **Probabilities:** `outputs/probabilities/[MODEL_NAME]-[TARGET]-probabilities.[format]`

      * Generated only if `output_probabilities: true`.
      * Contains one column per class.

3.  **Embeddings:** `outputs/embeddings/[MODEL_NAME]-embeddings.[format]`

      * Generated only if `model_type: embedding`.
      * Contains columns `0`, `1`, `2`... representing the vector dimensions.


# Hyperparameter Search Command Guide

The `sequifier hyperparameter-search` command automates the process of finding the optimal model architecture and training configuration. It supports both **Grid Search** (exhaustive) and **Random Sampling** strategies. It creates multiple unique training configurations, executes them sequentially, and logs the results.

## Usage

```console
sequifier hyperparameter-search --config-path configs/hyperparameter_search.yaml
```

## Configuration Fields

The configuration is defined in a YAML file. Unlike the `train.yaml` where fields take single values, most fields here take **lists** of values to search over.

### 1\. File System & Strategy

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. |
| `metadata_config_path`| `str` | **Yes** | - | Path to the JSON metadata file generated by `preprocess`. |
| `hp_search_name` | `str` | **Yes** | - | A prefix for the generated runs (e.g., `my-search`). |
| `model_config_write_path`| `str` | **Yes** | - | Directory to save the generated config files for each run (e.g., `configs/hp_search/`). |
| `search_strategy` | `str` | No | `sample` | `sample` (Random Search) or `grid` (Grid Search). |
| `n_samples` | `int` | *Conditional* | - | Required if `search_strategy` is `sample`. Number of distinct runs to execute. |
| `override_input` | `bool` | No | `false` | If `true`, suppresses interactive confirmation prompts. Useful for CI/CD or automated scripts. |
| `training_data_path` | `str` | No | Metadata split 0 | Path to training data. |
| `validation_data_path`| `str` | No | Metadata split 1 | Path to validation data. |

### 2\. System & Export (Fixed Values)

These fields are constant across all search runs (not sampled) but are required to configure the output models.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `export_generative_model`| `bool` | **Yes** | - | Export the standard next-token prediction model for every run. |
| `export_embedding_model` | `bool` | **Yes** | - | Export the vector embedding model for every run. |
| `inference_batch_size` | `int` | **Yes** | - | Batch size hardcoded into exported ONNX models. |
| `export_onnx` | `bool` | No | `true` | Export to ONNX format. |
| `export_pt` | `bool` | No | `false` | Export to PyTorch state dict (`.pt`). |
| `export_with_dropout` | `bool` | No | `false` | Export models with dropout enabled (for MC Dropout inference). |

### 3\. Schema & Feature Selection

Sequifier allows you to search not just for model parameters, but for the best **subset of input features**.

| Field | Type | Mandatory | Description |
| :--- | :--- | :--- | :--- |
| `input_columns` | `list[list[str]]`| **Yes** | A list of input sets. E.g., `[['col1'], ['col1', 'col2']]`. |
| `target_columns` | `list[str]` | **Yes** | The target column(s) to predict. Fixed across all runs. |
| `seq_length` | `list[int]` | **Yes** | List of sequence lengths to test (e.g., `[24, 48]`). |
| `target_column_types`| `dict` | **Yes** | Map of target columns to `categorical` or `real`. |
| `column_types` | `list[dict]` | *Conditional*| Required if `input_columns` varies. List of type maps corresponding to the input sets. |

### 4\. Model Architecture Sampling (`model_hyperparameter_sampling`)

These fields define the search space for the Transformer architecture. All fields accept a **list** of values unless noted.

| Field | Type | Mandatory | Description |
| :--- | :--- | :--- | :--- |
| `dim_model` | `list[int]` | **Yes** | Internal dimension of the Transformer. |
| `num_layers` | `list[int]` | **Yes** | Number of layers. |
| `n_head` | `list[int]` | **Yes** | Number of attention heads. |
| `dim_feedforward` | `list[int]` | **Yes** | Feedforward network dimension. |
| `initial_embedding_dim`| `list[int]` | **Yes** | Feature embedding size. Usually matches `dim_model`. |
| `joint_embedding_dim` | `list[int]` | **Yes** | Joint embedding size. If not null, must match `dim_model`. |
| `activation_fn` | `list[str]` | **Yes** | `['swiglu', 'gelu', 'relu']`. |
| `attention_type` | `list[str]` | **Yes** | `['mha', 'mqa', 'gqa']`. |
| `n_kv_heads` | `list[int]` | **Yes** | Number of KV heads (for MQA/GQA). Use `null` for MHA. |
| `normalization` | `list[str]` | **Yes** | `['rmsnorm', 'layer_norm']`. |
| `norm_first` | `list[bool]` | **Yes** | `[true, false]`. Pre-LN vs Post-LN. |
| `positional_encoding` | `list[str]` | **Yes** | `['learned', 'rope']`. |
| `rope_theta` | `list[float]` | **Yes** | Base frequency for RoPE (e.g., `[10000.0, 50000.0]`). |
| `prediction_length` | `int` | No | Fixed value (not sampled). Steps to predict simultaneously. Default 1. |

### 5\. Training Hyperparameters (`training_hyperparameter_sampling`)

Most fields here are lists for sampling, but some are scalar values fixed for all runs.

| Field | Type | Mandatory | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | `list[float]`| **Yes** | List of learning rates to test. |
| `batch_size` | `list[int]` | **Yes** | List of batch sizes. |
| `epochs` | `list[int]` | **Yes** | Epochs to train. Paired with `learning_rate`. |
| `accumulation_steps` | `list[int]` | **Yes** | Gradient accumulation steps. |
| `dropout` | `list[float]`| No | List of dropout probabilities (default `[0.0]`). |
| `optimizer` | `list[dict]` | No | List of optimizer configs (e.g., `[{'name': 'AdamW'}, {'name': 'AdEMAMix'}]`). |
| `scheduler` | `list[dict]` | No | List of scheduler configs. |
| `save_interval_epochs` | `int` | **Yes** | **Fixed.** Checkpoint save frequency. |
| `log_interval` | `int` | No | **Fixed.** Logging frequency (batches). Default 10. |
| `early_stopping_epochs`| `int` | No | **Fixed.** Stop if validation metric doesn't improve. |
| `num_workers` | `int` | No | **Fixed.** Data loading subprocesses. |
| `loss_weights` | `dict` | No | **Fixed.** Weights for multi-objective loss. |
| `class_weights` | `dict` | No | **Fixed.** Weights for imbalanced classes. |
| `backend` | str | No | `"nccl"` | The distributed training backend to use (e.g., `nccl` for GPUs, `gloo` for CPUs). Only relevant if `distributed: true`. |
| `device_max_concat_length` | `int` | No | `12` |  Controls recursive tensor concatenation to prevent CUDA kernel limits on specific hardware. Lower this if you encounter "CUDA error: too many resources requested for launch". |
| `max_ram_gb` | `int` | No | `16` | RAM limit (GB) for the cache when using lazy loading. |
| `load_full_data_to_ram` | `bool` | No |  `true` |  If `false`, uses lazy loading (requires `read_format: pt`). |
| `distributed` | `bool` | No | `false`| Enable multi-GPU training (DDP). Requires `read_format: pt`. |
| `layer_type_dtypes` | `dict` | No | **Fixed.** Map of layer types to dtypes (e.g., `{'linear': 'bfloat16'}`). |
| `layer_autocast` | `bool` | No | **Fixed.** Enable `torch.autocast` (default `true`). |
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
  * **Training:** `batch_size`, `dropout`, `accumulation_steps`, `optimizer`.
  * **Data:** `seq_length`.

### 3\. Special Case: `n_kv_heads`

`n_kv_heads` is sampled independently but validated dynamically. If a selected `n_kv_heads` value does not mathematically divide the selected `n_head`, the system automatically reverts to `n_kv_heads = null` (standard Multi-Head Attention) for that run to prevent a crash.

-----

## Key Trade-offs and Decisions

### 1\. `search_strategy`: `grid` vs. `sample`

  * **`grid` (Grid Search):**
      * *How it works:* Generates every possible combination of all provided lists.
      * *Pros:* Exhaustive. You are guaranteed to find the best combination within your search space.
      * *Cons:* Exponential explosion. If you have 4 independent parameters with 5 options each, that's $5^4 = 625$ runs.
  * **`sample` (Random Search):**
      * *How it works:* Randomly selects `n_samples` combinations from the defined space.
      * *Pros:* Much faster. Research suggests random search is often as effective as grid search because neural networks are often sensitive to only a few hyperparameters, which random search explores more efficiently.
      * *Cons:* Results are not deterministic (unless seeded) and might miss the optimal combination.

### 2\. Feature Selection (`input_columns`)

Sequifier uniquely allows you to treat "data" as a hyperparameter.

  * **Usage:** Provide a list of lists.
      * Run 1 might use `['sales', 'day_of_week']`
      * Run 2 might use `['sales', 'day_of_week', 'promotion_flag']`
  * **Benefit:** Helps identify if adding extra features (which increases model size and training time) actually yields better performance or simply adds noise.

### 3\. Automatic Retry Logic

The hyperparameter search command includes **automatic error handling for Out of Memory (OOM)** errors.

  * If a specific run crashes (e.g., CUDA OOM), Sequifier will automatically halve the `batch_size` and retry that specific configuration up to 3 times.
  * *Implication:* You can be aggressive with your `batch_size` in the config; the system will self-correct if it hits hardware limits.

-----

## Outputs

1.  **Generated Configs:** Located in `model_config_write_path` (e.g., `configs/hp_search/`).
      * Files named `[hp_search_name]-run-0.yaml`, `[hp_search_name]-run-1.yaml`, etc.
      * These are valid, standalone `train.yaml` files that you can inspect or re-run manually.
2.  **Logs:** Located in `logs/`.
      * Separate logs for each run, detailing the loss curves.
3.  **Models & Checkpoints:**
      * Saved in `models/` and `checkpoints/` with filenames including the run number (e.g., `models/sequifier-my-search-run-5-best-10.onnx`).
