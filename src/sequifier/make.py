import os
from importlib.resources import files
from pathlib import Path

from sequifier.typechecking import beartype

preprocess_config_string = """project_root: "."
preprocessing_data_path: PLEASE FILL
read_format: csv
write_format: parquet
selected_columns:
- EXAMPLE_INPUT_COLUMN_NAME
column_data_types: null
mask_column: null
split_ratios:
- 0.8
- 0.1
- 0.1
split_method: within_sequence
max_target_offset: 1
window_stride: 49
max_rows: null
window_lengths:
- 49
- 49
- 49
"""

train_config_string = """project_root: "."
model_name: PLEASE FILL
device: cuda
global_training:
  training_objective: causal
  context_length: 48
  inference_batch_size: 10
model:
  backbone:
    architecture:
      model_dim: 128
      num_layers: 3
      attention:
        num_heads: 16
      feed_forward:
        hidden_dim: 128
      dropout: 0.2
  interfaces:
    default:
      input_columns:
      - EXAMPLE_INPUT_COLUMN_NAME
      target_columns:
      - EXAMPLE_TARGET_COLUMN_NAME
      ingestion:
        type: embedding
        output_dim: 128
      decoder:
        type: linear
        prediction_length: 1
training_plan:
  phases:
  - name: supervised
    algorithm: supervised
    epochs: 10
    mode: sequential
    sources:
    - source: default.default
    batch_size: 10
    accumulation_batches: 1
    optimization:
      model:
        learning_rate: 0.0001
        optimizer:
          name: AdamW
          arguments: {}
        scheduler:
          name: StepLR
          arguments:
            step_size: 1
            gamma: 0.99
        scheduler_step_on: epoch
    evaluation:
      sources:
      - source: default.default
export_generative_model: true
export_embedding_model: false
dataset_training:
  default:
    criterion:
      EXAMPLE_TARGET_COLUMN_NAME: MSELoss
    parts:
      default:
        metadata_config_path: PLEASE FILL
"""

infer_config_string = """project_root: "."
input_columns:
- EXAMPLE_INPUT_COLUMN_NAME
target_columns:
- EXAMPLE_TARGET_COLUMN_NAME
target_column_types:
  EXAMPLE_TARGET_COLUMN_NAME: real
training_objective: causal
output_probabilities: false
decode_categories: true
device: cpu
context_length: 48
inference_batch_size: 10
generation_steps: 5
task: forecast
model_paths:
- PLEASE FILL
metadata_config_path: configs/metadata_configs/PLEASE_FILL.json
autoregressive_generation: true
"""

gitignore_string = """models/
logs/
checkpoints/
outputs/
data/
state/
.DS_Store"""


@beartype
def make(args):
    """Create a sequifier project scaffold."""
    project_name = args.project_name

    if not (project_name and len(project_name) > 0):
        raise ValueError(f"project_name '{project_name}' is not admissible")

    os.makedirs(f"{project_name}/configs")
    os.makedirs(f"{project_name}/scripts")

    with open(f"{project_name}/.gitignore", "w") as f:
        f.write(gitignore_string)

    with open(f"{project_name}/configs/preprocess.yaml", "w") as f:
        f.write(preprocess_config_string)

    with open(f"{project_name}/configs/train.yaml", "w") as f:
        f.write(train_config_string)

    with open(f"{project_name}/configs/infer.yaml", "w") as f:
        f.write(infer_config_string)

    for name in (
        "train-policy",
        "infer-policy",
        "train-iql",
        "train-ppo",
        "train-dpo",
        "train-grpo",
    ):
        template = files("sequifier").joinpath("templates", f"{name}.yaml")
        Path(project_name, "configs", f"{name}.yaml").write_text(
            template.read_text(encoding="utf-8"), encoding="utf-8"
        )
