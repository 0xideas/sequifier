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
| `read_format` | `str` | No | `parquet` | Format of input data. Single-file inference supports `csv` and `parquet`; folder inference supports `parquet` and `pt`. |
| `write_format` | `str` | No | `csv` | Format for output predictions (`csv`, `parquet`). |

### 2\. Schema & Columns

These fields tell the inference engine which columns to extract from the new data and how to interpret them.

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_columns` | `list[str]` or `null`| **Yes** | `null` | List of feature columns. Must match the columns the model was trained on. Set to `null` to use all metadata columns. |
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
| `output_probabilities`| `bool` | No | `false` | If `true`, outputs the full probability distribution for categorical targets. Real-valued targets do not produce probability files. |
| `sample_from_distribution_columns`| `Optional[list[str]]`| No | `null` | If set, the model **samples** from the predicted distribution for these columns instead of taking the top-1 (argmax). Essential for diversity in generation. |
| `map_to_id` | `bool` | No | `true` | If `true`, converts integer class predictions back to original string IDs (e.g., 0 -\> "cat"). Must be `false` when all targets are real-valued. |
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
  * **`pt`** Folder-only format optimized for lazy loading. Uses more disk space but less CPU than `parquet`.

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
