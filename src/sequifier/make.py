import os
from importlib.resources import files
from pathlib import Path

from sequifier.typechecking import beartype

preprocess_config_string = """preprocessing_data_path: PLEASE FILL
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
run:
  project_root: .
"""

train_config_string = """model:
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
  execution:
    context_length: 48
    training_objective: causal
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
      cadence:
        epochs: 1
      batch_size: 10
      before_phase: true
      on_snapshot: true
    model_selection:
      handoff: last
    checkpoints:
      cadence:
        epochs: 1
    learning:
      datasets:
        default:
          criterion:
            EXAMPLE_TARGET_COLUMN_NAME: MSELoss
dataset_training:
  default:
    parts:
      default:
        metadata_config_path: PLEASE FILL
        read_format: parquet
    split:
      validation_fraction: 0.2
run:
  project_root: .
  model_name: PLEASE FILL
  reproducibility:
    seed: 1010
    enforce_determinism: false
  execution:
    device: cuda
    distribution:
      strategy: local
      world_size: 1
      backend: nccl
    precision: {}
    data_loading: {}
    compilation:
      torch_compile: outer
  recovery: {}
  logging: {}
  outputs: {}
  export:
    generative_model: true
    embedding_model: false
    onnx: true
    pt: false
    batch_size: 10
"""

infer_config_string = """input_columns:
- EXAMPLE_INPUT_COLUMN_NAME
target_columns:
- EXAMPLE_TARGET_COLUMN_NAME
target_column_types:
  EXAMPLE_TARGET_COLUMN_NAME: real
training_objective: causal
output_probabilities: false
decode_categories: true
context_length: 48
generation_steps: 5
task: forecast
model_paths:
- PLEASE FILL
metadata_config_path: configs/metadata_configs/PLEASE_FILL.json
autoregressive_generation: true
run:
  project_root: .
  execution:
    device: cpu
    batch_size: 10
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
