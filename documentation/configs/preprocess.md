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
| `categorical_columns` | `list[str]` | No | `null` | Explicitly classify supported discrete processed columns as categorical instead of relying on dtype inference. Undeclared columns retain dtype-based inference. Columns cannot also appear in `real_columns`. When `column_data_types` is set, these columns must use an integer output dtype. |
| `real_columns` | `list[str]` | No | `null` | Explicitly classify processed numeric columns as real-valued. This is useful for integer-valued amounts, counts, and epoch timestamps that must retain ordinal meaning. Undeclared columns retain dtype-based inference. Columns cannot also appear in `categorical_columns`. When `column_data_types` is set, these columns must use a floating-point output dtype. |
| `column_data_types` | `dict[str, str]` | No | `null` | Optional output dtype map for processed columns, such as `Float32`, `Float64`, `Int32`, or `Int64`. A `Date` or `Datetime...` dtype is accepted only for the `value_cutoff` `split_column`. If set, every processed column must be included. Parquet uses one unified sequence dtype; `pt` writes each variable to its configured tensor dtype. |
| `normalize_real_columns` | `bool` | No | `true` | If `true`, Z-score normalizes real-valued columns. Set to `false` to preserve their original values. Statistics are still recorded in metadata. |
| `normalize_on_all_data` | `bool` | No | `false` | If `false`, numeric statistics and dynamic categorical vocabularies are fitted only on split 0; values seen only in later splits map to `[other]`. Set to `true` to retain the legacy all-data fitting behavior. |
| `max_rows` | `int` | No | `null` | Limits processing to the first N rows. Useful for rapid debugging. |
| `metadata_config_path` | `Optional[str]` | No | `null` | Use a preexisting metadata config for tokenizing discrete columns and, when enabled, standardizing real-valued columns. |
| `mask_column` | `Optional[str]` | No | `null` | Optional input column used as a row-level mask. If set, `metadata_config_path` must also be set, and it cannot also be `split_column`. |
| `curriculum_column` | `Optional[str \| list[str]]` | No | `null` | One or more optional integer input columns to preserve as per-subsequence curriculum metadata. |
| `use_precomputed_maps`| `list[str]` | No | `null` | If not `null`, enforces the use of precomputed maps for the variables in the list. |

When `curriculum_column` is set, each named column must be integer-valued and
constant within every generated subsequence. A list preserves several columns
so that a later training run can select any one of them. The values are stored
as metadata, not as model features. Different subsequences of one `sequenceId`
may use different values. For depth input, the values must also agree across
all repeated child rows for an outer item; depth PT output carries the same
per-window metadata. Names must be unique and non-empty, must exist in every
input file, and cannot be `sequenceId`, `itemPosition`, the mask column, or a
depth feature/position column. Names beginning with
`__sequifier_curriculum_value_` are reserved.

### 3\. Sequence Logic & Splitting

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `window_length` | `int` | **Yes** | - | The physical serialized window width written to preprocessed data. |
| `max_target_offset` | `int` | No | `1` | Number of future items retained after the model input window. Use `0` for BERT-style same-width inputs and targets; use `1` for causal next-item training. |
| `split_ratios` | `list[float]`| Conditional | `null` | Ordered split proportions for `within_sequence` and `between_sequence`. Must sum to 1.0 and must be omitted for `value_cutoff`. |
| `split_method` | `str` | No | `within_sequence` | How rows are assigned to splits: `within_sequence`, `between_sequence`, or `value_cutoff`. |
| `split_column` | `str` | Conditional | `null` | Required for `value_cutoff`. Names the integer, Date, or Datetime column compared with `split_values`; it cannot be a mask, curriculum, sequence ID, or item-position column. |
| `split_values` | `list[int \| timestamp]` | Conditional | `null` | Required for `value_cutoff`. Strictly increasing boundaries that create `len(split_values) + 1` splits. Values must all be integers or all be ISO timestamp values. |
| `window_stride` | `int` | No | `window_length` | Stored-window stride for distributed splits. |
| `prediction_aligned_splits` | `list[int]` | No | `[]` | Zero-based split indices whose prediction groups are anchored to each split end. All other splits use distributed placement. |
| `prediction_length` | `int` | Conditional | `null` | Required for prediction-aligned splits; number of output positions per window. |
| `target_offset` | `int` | Conditional | `null` | Required for prediction-aligned splits; must equal `max_target_offset`. |
| `allow_sequence_splitting` | `bool` | No | `false` | If `false`, a single sequence is kept within one preprocessing batch. |

All newly preprocessed windows store their absolute start position and split
target bounds, including distributed windows and depth-layout PT windows. Dataset
loaders require these fields and reject outputs created with an older payload
schema; re-run preprocessing to migrate such data.

`value_cutoff` applies the same boundaries to every sequence. Values equal to a
boundary belong to the later split, and the split column must be non-decreasing
within each sequence. Integer columns require integer boundaries. Timestamp
columns support Polars `Date` and `Datetime` values, or ISO timestamp strings
that can be parsed as UTC; `Time` and `Duration` are not supported. A timestamp
split column may be retained in preprocessing output by including it in
`selected_columns` (or by leaving `selected_columns: null`), but timestamp
columns cannot be model inputs or targets in training or inference. Any other
typed temporal column is rejected. The name `__sequifier_split_value` is
reserved for preprocessing internals.

To align validation or test predictions exactly to their split positions, configure
those split indices and the model's prediction view:

```yaml
window_length: 129
max_target_offset: 1
window_stride: 128
prediction_aligned_splits: [1, 2]
prediction_length: 2
target_offset: 1
```

Aligned splits place the last prediction group at the split end and step backward
by `prediction_length`. The first group may begin before the split; its earlier
predictions are masked from loss, metrics, and inference output. Inputs may use
preceding rows from the same sequence. Missing history at the sequence start is
left-padded. The stored windows never read beyond the split end. An aligned
split requires `target_offset == max_target_offset` and
`allow_sequence_splitting: false`. Training and inference must use matching
`target_offset` and `prediction_length`, with their model-view `window_stride`
set to `null` for aligned data. Empty splits produce no windows.
Prediction-aligned inference supports generative output without autoregressive
generation; embedding output positions follow input activations instead.

Distributed splits retain isolated, evenly spread windows and use the scalar
preprocessing `window_stride`. The same placement rules apply to depth-layout
PT output.

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

### 2\. Stored windows and model windows

Preprocessing stores windows of `window_length` events. Distributed splits use
`window_stride`; aligned splits step by `prediction_length`. Training's
`context_length` is the model input width;
`window_length` must be at least `context_length + max_target_offset`. Training's
`window_stride` is separate: `null` uses one right-aligned model view per stored
window, while a positive integer samples additional views *within* a longer
stored window. If the two widths are equal, `window_stride` adds no views.

| Scenario | Suggested settings | Trade-off |
| --- | --- | --- |
| Many short or varied-length sequences | Store the minimum width (for example, `window_length: 129` for `context_length: 128`, `max_target_offset: 1` use `window_stride` near 128) and, in the training config, set `window_stride: null`. | Limits padding for short sequences and stores roughly one copy of long sequences. |
| More overlap during training | Keep that width; reduce the training split's `window_stride` value to a fraction of preprocessing config `context_length`. | Roughly 2× or 4× as many stored events for long sequences. |
| Long sequences, several model views per stored window | Use a longer stored width (for example, `window_length: 513`, `window_stride: 384`, `context_length: 128`, training config `window_stride: 128`). | About 1.3× stored events on long sequences; short sequences pad to 513, and more model views cost more compute. |
| Dense evaluation with nearly full preceding context at each window's right edge | Use `prediction_aligned_splits` for exact split coverage, or a small preprocessing `window_stride` for distributed evaluation. | More stored windows and more evaluation compute. |

With causal next-event targets, a stride near `context_length` lets successive
minimum-width windows cover target positions with little overlap. The model also
learns from positions inside each window, where less preceding history is
available. Smaller strides repeat more positions and can better represent
full-history serving at evaluation time. Short sequences are left-padded to
`window_length` regardless of stride; inspect the sequence-length distribution
before choosing a long stored width.

### 3\. Distributed and prediction-aligned placement

Distributed placement adjusts starts to cover each split evenly and includes
the final available window. Prediction-aligned placement anchors prediction
groups to the split end, then steps backward by `prediction_length`. When the
split length is not divisible by `prediction_length`, the first group includes
positions before the split; those predictions are masked.

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

## Named depth layouts

Repeated rows can describe one outer item with a fixed-capacity child collection.
Depth behavior is explicit; a column named `subItemPosition` alone remains an
ordinary flat feature.

```yaml
project_root: .
preprocessing_data_path: data/raw-items
read_format: parquet
write_format: pt
merge_output: false
selected_columns: [accountType, subitemType, subitemAmount, nextAction]
depth_layouts:
  subitems:
    position_column: subItemPosition
    columns: [subitemType, subitemAmount]
    context_length: 16
    position_base: 0
    allow_gaps: false
window_length: 129
max_target_offset: 1
split_ratios: [0.8, 0.1, 0.1]
window_stride: 128
```

Every file must contain the depth position column, which is read automatically
and excluded from feature statistics and output types. Configured
`curriculum_column` values are also read automatically and cannot be used as
the depth position or a depth feature. Configure at most one raw layout, use PT output
without merging, and omit `mask_column`. Reused metadata must have the same
complete layout definition, output types, and normalization policy. String
identifiers must be convertible to signed Int64; item, curriculum, and depth
positions must have integer source types.

The adapter indexes raw fragments on disk before grouping them. `max_rows`
counts complete outer items ordered by `(sequenceId, itemPosition)`, including
children found in later files. Shallow features must agree across every child
row before casting or mapping. Shallow statistics count each item once; deep
statistics count occupied child slots. Both populations are selected before
split extraction. Materialization uses bounded windows and output batches;
`batches_per_file` bounds the number of windows accumulated per split on this
path. This adapter is currently sequential; `n_cores` does not parallelize it.

Child positions map to physical slots by subtracting `position_base`. Without
`allow_gaps`, occupied slots must be a prefix starting at zero. Tail padding is
always allowed. With gaps enabled, physical slots remain unchanged. Outer item
positions must be continuous within each selected sequence. An item in this raw
format must have at least one child; null child rows do not encode emptiness.

All PT files use version 5 of the `sequifier_tensor_batch` envelope. The payload
stores shallow tensors as `[N,W]`, deep tensors as `[N,W,D]`, boolean masks under
`metadata.depth_valid_masks.<layout>`, and the absolute window and split-boundary
positions needed to enforce split ownership. Optional signed Int64 curriculum
values are stored under `metadata.sample_positions`; their source names are
stored under `metadata.curriculum_columns`, and multiple preserved columns use
shape `[N,C]`. Public metadata records the model-facing
`tensor_payload_version` separately from this internal storage envelope. Readers
reject legacy tuple payloads and envelope versions 2 through 4; re-run
preprocessing to migrate them.
Categorical padding is the existing unknown-token ID (zero); real padding is
finite zero after normalization. Temporal padding has false depth masks.

External tensor payloads may contain several layouts with different capacities
and independently empty collections. Use `StoredTensorBatch`, `save_pt_payload`,
and `load_pt_payload` in `sequifier.io.pt_payload`, supplying the complete layout
registry and categorical vocabulary sizes. All stored feature values, including
masked slots, must have legal categorical indices and finite real values. Empty
collections use an all-false mask. Missing masks and forbidden internal gaps are
errors. Selected interfaces compare only relevant layout feature membership and
layout properties, so unused stored layouts/features can be added independently.

To consume a named layout during training, configure a `depth_transformer`
ingestion branch whose `layout` names this entry and whose `columns` are selected
features from it. See [the training guide](train.md#depth-encoders-and-nested-composites).
