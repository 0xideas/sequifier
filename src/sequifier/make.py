import os
from pathlib import Path

from sequifier.scaffolds import CONTROL_TEMPLATES, template_text
from sequifier.typechecking import beartype

preprocess_config_string = """preprocessing_data_path: data/events.csv
selected_columns: [value]
real_columns: [value]
splits:
  train: {fraction: 0.8, window_length: 49}
  validation: {fraction: 0.1, window_length: 49}
  test: {fraction: 0.1, window_length: 49}
"""

infer_config_string = """task: forecast
artifact: {paths: [models/forecast.pt]}
input:
  metadata_config_path: configs/metadata_configs/events.json
  split: test
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
    os.makedirs(f"{project_name}/data")

    with open(f"{project_name}/.gitignore", "w") as f:
        f.write(gitignore_string)

    with open(f"{project_name}/configs/preprocess.yaml", "w") as f:
        f.write(preprocess_config_string)

    Path(project_name, "configs", "train.yaml").write_text(
        template_text("train-supervised"), encoding="utf-8"
    )

    with open(f"{project_name}/configs/infer.yaml", "w") as f:
        f.write(infer_config_string)

    for name in ("train-policy", "infer-policy", "train-iql", *CONTROL_TEMPLATES):
        Path(project_name, "configs", f"{name}.yaml").write_text(
            template_text(name), encoding="utf-8"
        )
