import os

preprocess_config_string = """project_path: .
data_path: PLEASE FILL
read_format: csv
write_format: parquet
selected_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns

group_proportions:
- 0.8
- 0.1
- 0.1
seq_length: 48
seq_step_sizes:
- 1
- 1
- 1
max_rows: null
"""

train_config_string = """project_path: .
model_name: default
read_format: parquet
ddconfig_path: PLEASE FILL

selected_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
target_column_types: # 'criterion' in training_spec must also be adapted
  EXAMPLE_TARGET_COLUMN_NAME: real

seq_length: 48
inference_batch_size: 10

export_onnx: true

model_spec:
  d_model: 128
  d_model_by_column: # the size of the embedding of individual variables, must sum to d_model
    EXAMPLE_INPUT_COLUMN_NAME: # can be left out if either all input variables are real or all are categorical
  nhead: 16
  d_hid: 128
  nlayers: 3
training_spec:
  device: cuda
  epochs: 1000
  iter_save: 10
  batch_size: 100
  log_interval: 10
  lr: 0.0001
  accumulation_steps: 1
  dropout: 0.2
  criterion:
    EXAMPLE_TARGET_COLUMN_NAME: MSELoss
  optimizer:
    name: AdamW
  scheduler:
    name: CosineAnnealingLR
    T_max: 111
    eta_min: 0.00001
  continue_training: true
"""

infer_config_string = """project_path: .
ddconfig_path: PLEASE FILL
model_path: PLEASE FILL
data_path: PLEASE FILL

selected_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
target_column_types:
  EXAMPLE_TARGET_COLUMN_NAME: real

output_probabilities: false
map_to_id: false
device: cpu
seq_length: 48
inference_batch_size: 10

autoregression: true
"""

gitignore_string = """models/
logs/
checkpoints/
outputs/
data/
.DS_Store"""


def make(args):
    project_name = args.project_name

    assert (
        project_name is not None and len(project_name) > 0
    ), f"project_name '{project_name}' is not admissible"

    os.makedirs(f"{project_name}/configs")

    with open(f"{project_name}/.gitignore", "w") as f:
        f.write(gitignore_string)

    with open(f"{project_name}/configs/preprocess.yaml", "w") as f:
        f.write(preprocess_config_string)

    with open(f"{project_name}/configs/train.yaml", "w") as f:
        f.write(train_config_string)

    with open(f"{project_name}/configs/infer.yaml", "w") as f:
        f.write(infer_config_string)
