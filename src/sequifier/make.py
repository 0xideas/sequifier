import os

from sequifier.typechecking import beartype

preprocess_config_string = """project_root: .
preprocessing_data_path: PLEASE FILL
read_format: csv
write_format: parquet
selected_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
column_data_types: null # optional map of selected columns to output dtypes, e.g. {EXAMPLE_INPUT_COLUMN_NAME: Float32}
mask_column: null

split_ratios:
- 0.8
- 0.1
- 0.1
split_method: within_sequence # one of within_sequence, between_sequence
stored_context_width: 49
max_target_offset: 1
max_rows: null
"""

train_config_string = """project_root: .
model_name: PLEASE FILL
device: cuda

global_training_spec:
  training_objective: causal
  context_length: 48
  inference_batch_size: 10
  batch_size: 10
  learning_rate: 0.0001
  optimizer: {name: AdamW}
  scheduler: {name: StepLR, step_size: 1, gamma: 0.99}
  scheduler_step_on: epoch

model_spec:
  backbone:
    architecture:
      dim_model: 128
      max_context_length: 512
      num_layers: 3
      attention: {n_heads: 16}
      feed_forward: {dim: 128}
      dropout: 0.2
  interface:
    input_columns: [EXAMPLE_INPUT_COLUMN_NAME]
    target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
    ingestion: {type: direct_embed, output_dim: 128}
    decoder: {type: linear, prediction_length: 1}

dataset:
  part: {metadata_config_path: PLEASE FILL}
  criterion: {EXAMPLE_TARGET_COLUMN_NAME: MSELoss}

training_plan: {epochs: 10}

evaluation: true

export_generative_model: PLEASE FILL # true or false
export_embedding_model: PLEASE FILL # true or false
"""

infer_config_string = """project_root: .
preprocessing_data_path: PLEASE FILL
model_type: PLEASE FILL # generative or embedding
model_path: PLEASE FILL

input_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
target_column_types:
  EXAMPLE_TARGET_COLUMN_NAME: real

training_objective: causal
output_probabilities: false
map_to_id: true
device: cpu
context_length: 48
inference_batch_size: 10

autoregression: true
autoregression_total_steps: 5
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
    os.makedirs(f"{project_name}/state/optuna")
    os.makedirs(f"{project_name}/scripts")

    with open(f"{project_name}/.gitignore", "w") as f:
        f.write(gitignore_string)

    with open(f"{project_name}/configs/preprocess.yaml", "w") as f:
        f.write(preprocess_config_string)

    with open(f"{project_name}/configs/train.yaml", "w") as f:
        f.write(train_config_string)

    with open(f"{project_name}/configs/infer.yaml", "w") as f:
        f.write(infer_config_string)
