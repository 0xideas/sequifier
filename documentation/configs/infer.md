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
| `model_path` | `str` or `list[str]` | **Yes** | - | Path to a specific model file, or a list of paths to process sequentially, for example `models/my-model-best-10.pt`. |
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
