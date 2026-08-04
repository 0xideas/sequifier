import os

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
preprocessing_data_path: PLEASE FILL
model_name: PLEASE FILL
training_objective: causal
device: cuda
read_format: parquet

input_columns: [EXAMPLE_INPUT_COLUMN_NAME] # should include all target column, can include additional columns
feature_layout: null # optional structure annotations for flat input columns
target_columns: [EXAMPLE_TARGET_COLUMN_NAME]
target_column_types: # 'criterion' in training_spec must also be adapted
  EXAMPLE_TARGET_COLUMN_NAME: real

context_length: 48
model_window_stride: null # null loads one right-aligned view; a positive integer loads all contained views at this stride
inference_batch_size: 10

export_generative_model: PLEASE FILL # true or false
export_embedding_model: PLEASE FILL # true or false
export_onnx: true

model_spec:
  ingestion:
    type: direct_embed
    output_dim: 128
    initialization: {} # optional ingestion initialization overrides
    feature_embedding_dims: null # optional, e.g. {EXAMPLE_INPUT_COLUMN_NAME: 128}; unnecessary if all inputs share one numeric kind
  backbone:
    id: shared-backbone-v1
    architecture:
      dim_model: 128
      max_context_length: 512
      num_layers: 3
      attention:
        type: mha
        n_heads: 16
        n_kv_heads: 16
        output_projection: true
      feed_forward:
        dim: 128
        activation: swiglu
      normalization:
        type: rmsnorm
        norm_first: true
      position_encoding:
        type: learned
        theta: 10000
      dropout: 0.2
      shared_layer_groups: []
    repository:
      path: checkpoints/backbones/shared-backbone-v1
      load_policy: if_exists
      publish: true
      conflict_policy: compare_and_swap
    initialization: {}
  decoder:
    type: linear
    prediction_length: 1
    support: 1
    initialization: {}
training_spec:
  epochs: 10
  save_interval_epochs: 10
  batch_size: 10
  log_interval: 10
  learning_rate: 0.0001
  accumulation_steps: 1
  gradient_clip: null
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
  scheduler_step_on: batch
  resume:
    policy: if_exists
    checkpoint_path: checkpoints/runs/PLEASE_FILL_RUN_ID/latest.pt
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
model_window_stride: null
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
