# Infer Command Guide

The `sequifier infer` command uses a trained Sequifier model (PyTorch `.pt` or ONNX `.onnx`) to generate predictions, probabilities, or vector embeddings on new data. It handles batching, data normalization (and denormalization), and supports complex inference modes like **autoregression**.

## Usage

```console
sequifier infer --config-path configs/infer.yaml
````

## Configuration Fields

The configuration is defined in a YAML file (e.g., `infer.yaml`).

### 1\. File System & Model Loading

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_root` | `str` | **Yes** | - | The root directory of your Sequifier project. Usually `.` |
| `data_path` | `str` | **Yes** | - | Path to the input data file (csv/parquet) or folder (if `read_format: pt`). |
| `model_path` | `str` | **Yes** | - | Path to the specific model file (e.g., `models/sequifier-[NAME]-best-[EPOCH].pt`). |
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
| `prediction_length` | `int` | No | `1` | Number of steps to predict *simultaneously* (if model supports it). Usually 1. |
| `inference_batch_size`| `int` | **Yes** | - | Number of sequences to process at once. |
| `autoregression` | `bool` | No | `false` | If `true`, feeds predictions back into the model to predict further into the future. |
| `autoregression_extra_steps`| `int` | No | `null` | If `autoregression: true`, how many *additional* future steps to predict beyond the first. |
| `output_probabilities`| `bool` | No | `false` | If `true`, outputs the full probability distribution for categorical targets. |
| `map_to_id` | `bool` | No | `true` | If `true`, converts integer class predictions back to original string IDs (e.g., 0 -\> "cat"). |
| `infer_with_dropout` | `bool` | No | `false` | If `true`, keeps dropout active during inference (useful for uncertainty estimation/Monte Carlo Dropout). |

### 4\. System & Distributed

| Field | Type | Mandatory | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | **Yes** | - | `cuda`, `cpu`, or `mps`. |
| `distributed` | `bool` | No | `false`| Enable multi-GPU inference. Requires `read_format: pt`. |
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

### 3\. Probabilities (`output_probabilities: true`)

  * **True:** For categorical targets, outputs the probability of *every class* rather than just the top prediction.
      * *Pros:* Necessary for calculating uncertainty, setting custom confidence thresholds, or analyzing "top-k" predictions.
      * *Cons:* Creates very large output files (Batch Size $\times$ Number of Classes).
  * **False:** Outputs only the single most likely class (or value).

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
