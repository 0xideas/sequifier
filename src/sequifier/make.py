import os

preprocess_config_string = """project_root: .
data_path: PLEASE FILL
read_format: csv
write_format: parquet
selected_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
column_types: null # optional map of selected columns to output dtypes, e.g. {EXAMPLE_INPUT_COLUMN_NAME: Float32}
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
read_format: parquet
metadata_config_path: PLEASE FILL

input_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
target_column_types: # 'criterion' in training_spec must also be adapted
  EXAMPLE_TARGET_COLUMN_NAME: real

context_length: 48
inference_batch_size: 10

export_generative_model: PLEASE FILL # true or false
export_embedding_model: PLEASE FILL # true or false
export_onnx: true

model_spec:
  initial_embedding_dim: 128
  feature_embedding_dims: # the size of the embedding of individual variables, must sum to dim_model
    EXAMPLE_INPUT_COLUMN_NAME: # can be left out if either all input variables are real or all are categorical
  joint_embedding_dim: null
  dim_model: 128
  n_head: 16
  dim_feedforward: 128
  num_layers: 3
  prediction_length: 1
training_spec:
  training_objective: causal
  device: cuda
  epochs: 10
  save_interval_epochs: 10
  batch_size: 10
  log_interval: 10
  learning_rate: 0.0001
  accumulation_steps: 1
  dropout: 0.2
  criterion:
    EXAMPLE_TARGET_COLUMN_NAME: MSELoss
  optimizer:
    name: AdamW
  scheduler:
    name: OneCycleLR
    max_lr: 0.001
    pct_start: 0.1
    div_factor: 100
    final_div_factor: 1000
    anneal_strategy: cos
    total_steps: PLEASE FILL
    three_phase: false
  continue_training: true
"""

infer_config_string = """project_root: .
metadata_config_path: PLEASE FILL
model_type: PLEASE FILL # generative or embedding
model_path: PLEASE FILL
data_path: PLEASE FILL

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
"""

gitignore_string = """models/
logs/
checkpoints/
outputs/
data/
state/
.DS_Store"""


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
