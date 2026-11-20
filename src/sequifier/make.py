import os
from importlib.resources import files
from pathlib import Path

import yaml

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

    template = files("sequifier").joinpath("templates", "train-supervised.yaml")
    training = yaml.safe_load(template.read_text(encoding="utf-8"))
    training["run"]["model_name"] = "PLEASE FILL"
    training["run"]["execution"]["device"] = "cuda"
    with open(f"{project_name}/configs/train.yaml", "w") as f:
        yaml.safe_dump(training, f, sort_keys=False)

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
