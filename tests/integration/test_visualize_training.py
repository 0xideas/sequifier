import glob
import os

from conftest import run_and_log


def test_visualize_training(
    run_training, run_training_from_checkpoint, run_hp_search, project_root
):
    # Dynamically find all models that were trained and logged
    log_dir = os.path.join(project_root, "logs")
    log_files = glob.glob(os.path.join(log_dir, "sequifier-*-rank0-*.txt"))

    models = set()
    for lf in log_files:
        filename = os.path.basename(lf)
        # Extract model name by stripping known prefix and suffix
        name = filename[len("sequifier-") :]
        name = name.rsplit("-rank0-", 1)[0]
        models.add(name)

    single_models = sorted([m for m in models if "categorical" in m or "real" in m])
    assert len(single_models) > 0, "No single models found in logs directory"

    hp_models_grid = sorted([m for m in models if "hp-search-grid-run" in m])
    assert len(hp_models_grid) > 0, "No hp grid models found in logs directory"

    # 1. Test running visualize-training for each model individually
    # model_outputs = {}
    for model in single_models:
        # visualize_training.py expects to find "logs/" relative to the working directory
        command = f"sequifier visualize-training {model} --project-root {project_root}"
        run_and_log(command)

        output_path = os.path.join(
            project_root,
            "outputs",
            "visualization",
            f"{model}-training-visualization.html",
        )
        assert os.path.exists(
            output_path
        ), f"Visualization output not found for {model}"

    # 2. Test running visualize-training for all models jointly
    models_str = ",".join(hp_models_grid)
    command_joint = f"sequifier visualize-training {models_str} --project-root {project_root} --log-scale --bucket-training-batches 5"
    run_and_log(command_joint)

    output_path_joint = os.path.join(
        project_root,
        "outputs",
        "visualization",
        "multi-model-training-visualization.html",
    )

    assert os.path.exists(output_path_joint), "Joint visualization output not found"
