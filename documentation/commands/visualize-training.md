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

For a single-dataset model, the command reads `logs/[MODEL_NAME]/[MODEL_NAME]-training-full.csv` and `logs/[MODEL_NAME]/[MODEL_NAME]-validation-full.csv`. The corresponding files without `-full` contain only condensed global-loss records.

## Outputs

The interactive HTML reports are saved in the `outputs/visualization/` directory.

* **Single Model:** `outputs/visualization/[MODEL_NAME]-training-visualization.html` (Includes global losses and normalized variable validation losses if applicable).
* **Multiple Models:** `outputs/visualization/multi-model-training-visualization.html` (Side-by-side comparison of validation and training losses across all specified models).
* **Hyperparameter Search:** `outputs/visualization/[SEARCH_NAME].html` (Includes all valid runs and lists skipped invalid runs and their reasons).

If every run in a hyperparameter search is invalid, Sequifier still creates the report with an empty plot and the invalid-run list.

When comparing multiple models, their initial baseline validation loss must match unless `SKIP_BASELINE_CHECK` or `SEQUIFIER_SKIP_BASELINE_CHECK` is set.
