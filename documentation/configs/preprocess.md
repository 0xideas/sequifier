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
| `-r`, `--randomize` | Generates a random `seed`. The seed affects `between_sequence` split assignment. |
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
      * **Crucial:** This file contains the integer mappings for categorical variables (`id_maps`) and normalization stats for real variables (`selected_columns_statistics`).
      * **Next Step:** You must link this file path in your `train.yaml` and `infer.yaml` under `metadata_config_path`.
