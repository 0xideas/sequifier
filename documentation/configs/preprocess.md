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

-----

## Outputs

After running `preprocess`, the following are generated:

1.  **Data Files:** Located in `data/`. Depending on your configuration, these will be `[NAME]-split0.parquet` (Training), `[NAME]-split1.parquet` (Validation), etc., or folders containing `.pt` files.
2.  **Metadata Config:** Located in `configs/metadata_configs/[NAME].json`.
      * **Crucial:** This file contains the integer mappings for categorical variables (`id_maps`) and normalization stats for real variables (`selected_columns_statistics`).
      * **Next Step:** You must link this file path in your `train.yaml` and `infer.yaml` under `metadata_config_path`.
