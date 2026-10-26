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
| `cardinality_config` | `dict[str, object]` | No | `{}` | Per-categorical-column rules for retaining frequent values and optionally hashing the rest into buckets. See [Categorical cardinality and hashing](#4-categorical-cardinality-and-hashing). |
| `column_data_types` | `dict[str, str]` | No | `null` | Optional output dtype map for processed columns, such as `Float32`, `Float64`, `Int32`, or `Int64`. A `Date` or `Datetime...` dtype is accepted only for the `value_cutoff` `split_column`. If set, every processed column must be included. Parquet uses one unified sequence dtype; `pt` writes each variable to its configured tensor dtype. |
| `normalize_real_columns` | `bool` | No | `true` | If `true`, Z-score normalizes real-valued columns. Set to `false` to preserve their original values. Statistics are still recorded in metadata. |
| `normalize_on_all_data` | `bool` | No | `false` | If `false`, numeric statistics and dynamic categorical vocabularies are fitted only on split 0; values seen only in later splits map to `[other]` or a configured hash bucket. Set to `true` to fit on all splits. |
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
| `window_length` | `int` | Conditional | - | One physical width for every split. Specify exactly one of `window_length` or `window_lengths`. |
| `window_lengths` | `list[int]` | Conditional | - | Physical widths in split order, with one entry per split. Required when `prediction_alignment_steps > 1`; widths from split 2 onward must equal the first width plus `prediction_alignment_steps - 1`. |
| `max_target_offset` | `int` | No | `1` | Number of future items retained after the model input window. Use `0` for BERT-style same-width inputs and targets; use `1` for causal next-item training. |
| `split_ratios` | `list[float]`| Conditional | `null` | Ordered split proportions for `within_sequence` and `between_sequence`. Must sum to 1.0 and must be omitted for `value_cutoff`. |
| `split_method` | `str` | No | `within_sequence` | How rows are assigned to splits: `within_sequence`, `between_sequence`, or `value_cutoff`. |
| `split_column` | `str` | Conditional | `null` | Required for `value_cutoff`. Names the integer, Date, or Datetime column compared with `split_values`; it cannot be a mask, curriculum, sequence ID, or item-position column. |
| `split_values` | `list[int \| timestamp]` | Conditional | `null` | Required for `value_cutoff`. Strictly increasing boundaries that create `len(split_values) + 1` splits. Values must all be integers or all be ISO timestamp values. |
| `window_stride` | `int` | No | First split width | Stored-window stride for distributed splits. |
| `prediction_aligned_splits` | `list[int]` | No | `[]` | Zero-based split indices whose prediction groups are anchored to each split end. All other splits use distributed placement. |
| `prediction_alignment_steps` | `int` | No | `1` | Number of consecutive prediction groups represented by each aligned stored window. |
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
`target_offset` and `prediction_length`. Training can sample additional model
views with `global_training.window_stride`; aligned inference requires
`window_stride: null`. Empty splits produce no windows.

Prediction-aligned inference supports generative and embedding output without
autoregressive generation. Aligned embeddings are labeled with the target row
predicted from each activation and use the target validity mask. For a causal
model with `target_offset: 1`, the embedding labeled as row N represents the
context before row N; it does not include row N's value. Embeddings from other
splits retain their input-position labels.

Distributed splits retain isolated, evenly spread windows and use the scalar
preprocessing `window_stride`. The same placement rules apply to depth-layout
PT output.

### 4\. Performance & System

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `seed` | `int` | No | `1010` | Random seed for reproducibility. |
| `n_cores` | `int` | No | Max Cores | Maximum worker processes for eligible preprocessing stages. Depth preprocessing parallelizes a single Parquet file only when it has multiple row groups and `max_rows` is unset. |
| `batches_per_file` | `int` | No | `1024` | Buffer flush threshold per split. For flat input, counts sequence-window groups in PT and Parquet output (including merged output); a byte limit can flush earlier. For depth input, counts windows in PT output. |
| `process_by_file` | `bool` | No | `true` | Memory optimization. If `true`, processes one input file at a time. |

-----

## Key Trade-offs and Decisions

### 1\. `write_format`: `parquet` vs. `pt`

  * **Choose `parquet` (default):** Unless you have a specific reason, use `parquet`. *Note: If you are doing distributed training, Parquet support is currently in **Beta**.*
  * **Choose `pt`:** Use `pt` data loading if speed and CPU overhead are your primary bottlenecks, **or if you are running multi-GPU distributed training.** This format is the most stable choice for high-throughput scaling.

### 2\. Stored windows and model windows

Preprocessing stores windows of the configured width for each split. Distributed
splits use `window_stride`; aligned stored windows step by
`prediction_alignment_steps × prediction_length`. Training's `context_length`
is the model input width. Each selected split width must be at least
`context_length + max_target_offset`. Training's
`window_stride` is separate: `null` uses one right-aligned model view per stored
window, while a positive integer samples additional views *within* a longer
stored window. If the two widths are equal, `window_stride` adds no views.

Aligned inference reads `prediction_alignment_steps` model views from each
stored window, moving right by `prediction_length` positions for each view.
Preprocessing requires room for all prediction groups after `max_target_offset`;
training and inference also check the width against the model's `context_length`.
To use the entire window, choose its width as
`context_length + target_offset + (prediction_alignment_steps - 1) × prediction_length`.
Earlier positions outside the split are masked from prediction output.
Set `window_lengths` when `prediction_alignment_steps > 1`. The first two
split widths can differ; every later split width must equal the first width
plus `prediction_alignment_steps - 1`, even when that split uses distributed
placement.

For example, with `context_length: 128`, `target_offset: 1`, and three splits,
the following gives aligned splits room for three two-position prediction
groups:

```yaml
window_lengths: [131, 133, 133]
max_target_offset: 1
prediction_aligned_splits: [1, 2]
prediction_length: 2
prediction_alignment_steps: 3
target_offset: 1
```

| Scenario | Suggested settings | Trade-off |
| --- | --- | --- |
| Many short or varied-length sequences where sparse target sampling is acceptable | Store the minimum width (for example, `window_length: 129` for `context_length: 128`, `max_target_offset: 1`), use a preprocessing `window_stride` near 128, and set training `window_stride: null`. | Limits padding and stores roughly one copy of long sequences, but with `prediction_length: 1` only the last target of each window contributes to loss. |
| More overlap during training | Keep that width; reduce the preprocessing `window_stride` to a fraction of the training `context_length`. | Roughly 2× or 4× as many stored events for long sequences. |
| Long sequences, several model views per stored window | Use a longer stored width (for example, `window_length: 513`, `window_stride: 384`, `context_length: 128`, training config `window_stride: 128`). | About 1.3× stored events on long sequences; short sequences pad to 513, and more model views cost more compute. |
| Dense evaluation with nearly full preceding context at each window's right edge | Use `prediction_aligned_splits` for exact split coverage, or a small preprocessing `window_stride` for distributed evaluation. | More stored windows and more evaluation compute. |

Self-attention lets the model use relationships between positions within a
window, so heavily overlapping stored windows are often unnecessary. Small
preprocessing strides increase storage and compute and may put very similar
windows in a training batch. The appropriate stride still depends on the
prediction objective: Sequifier computes loss only on the last
`prediction_length` positions of each model view. With causal next-event targets
and `prediction_length: 1`, a stride near `context_length` samples roughly one
target position per stored window on long sequences; the other positions provide
context but do not contribute directly to loss. A smaller stride covers more
target positions, even though adjacent inputs overlap. A larger
`prediction_length` can train on more positions per window when the decoder and
objective support it. Short sequences are left-padded to their split's stored
width regardless of stride; inspect the sequence-length distribution before
choosing a long stored width.

### 3\. Distributed and prediction-aligned placement

Distributed placement adjusts starts to cover each split evenly and includes
the final available window. Prediction-aligned placement anchors stored windows
to the split end, then steps backward by
`prediction_alignment_steps × prediction_length`. Each stored window contains
`prediction_alignment_steps` consecutive prediction groups. When the split
length is not divisible by this spacing, the first window includes positions
before the split; those predictions are masked.

### 4. Categorical cardinality and hashing

Use `cardinality_config` to limit the number of distinct IDs created for a
categorical column. Each key is a column name. For example, keep the 10,000 most
frequent product IDs and hash every other product ID into 4,096 buckets:

```yaml
categorical_columns: [product_id]
cardinality_config:
  product_id:
    top_k: 10000
    hashing:
      num_buckets: 4096
      seed: 42
```

`top_k` retains at most that many values, ranked by frequency. Alternatively,
`min_freq` retains every value occurring at least that many times. Both must be
positive integers and cannot be used together. Retained values keep individual
IDs; all remaining non-null values, including values first seen in later
splits, map to a hash bucket when `hashing` is configured. Without `hashing`,
non-retained values map to `[other]` instead. To hash all ordinary categorical
values, omit both retention settings:

```yaml
cardinality_config:
  product_id:
    hashing:
      num_buckets: 4096
```

`num_buckets` is a positive integer. The optional `hashing.seed` defaults to
`0` and is separate from the top-level preprocessing `seed`. Hashing is stable
for strings, booleans, and integers. Multiple values can land in the same
bucket, so their original identities cannot be recovered from preprocessed
data; decoded bucket values appear as `[hash_bucket:k]`. Null values retain the
reserved `[unknown]` ID.

By default, frequencies and retained values are fitted on split 0. Set
`normalize_on_all_data: true` to fit them on all data. A configured column must
be categorical, must appear in `selected_columns` when that list is set, and
cannot also appear in `use_precomputed_maps`. If `categorical_columns` is set,
include the configured column there. Preprocessing hashing is independent of
the model interface's `categorical_hashing` setting described in the
[training guide](train.md#categorical-multi-hash-embeddings-and-targets).

### 5. Advanced: Static Vocabularies (Custom ID Maps)

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
complete layout definition, output types, and normalization policy. Sequence
identifiers and item positions must have integer source types and fit signed
Int64; curriculum and depth positions must have integer source types.

Within each source file, rows for one `(sequenceId, itemPosition)` must be
adjacent, and distinct item coordinates must increase in that order. An item
may occur in only one file. Files may cover interleaved coordinate ranges;
the adapter merges their ordered item streams without a temporary database.
For large file counts, the merge uses bounded fan-in and temporary sorted runs.
`max_rows` counts complete outer items in global coordinate order. Ordering
and cross-file uniqueness are checked across the full input, including items
beyond `max_rows`. Shallow features must agree across every child row before
casting or mapping. Shallow statistics count each item once; deep
statistics count occupied child slots. Both populations are selected before
split extraction. Materialization encodes each item once and uses bounded
windows and output batches; `batches_per_file` bounds the number of windows
accumulated per split. For a single Parquet file with multiple row groups and
no `max_rows`, this adapter uses `n_cores` to parallelize auditing, metadata
fitting, and window materialization. Other depth inputs use the sequential
merge path.

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
