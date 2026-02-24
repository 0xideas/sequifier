import glob
import os
import re

from conftest import run_and_log


def sanitize_html(text):
    """
    Sanitizes dynamic content from Plotly HTML outputs to allow structural testing.
    """
    # 1. Strip standard 36-character UUIDs (Plotly container IDs)
    text = re.sub(
        r"(?i)[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", "CONTAINER_UUID", text
    )

    # 2. Strip 8-character Plotly trace UIDs
    text = re.sub(r'(?i)"uid":\s*"[0-9a-f]{8}"', '"uid": "TRACE_UID"', text)

    # 3. Strip CDN script integrity hashes (e.g., sha256-...)
    text = re.sub(r'integrity="[^"]+"', 'integrity="STRIPPED_HASH"', text)

    # 4. Strip specific plotted numbers in x and y data arrays
    # This finds "x": [...] or "y": [...] and empties the flat arrays to just []
    text = re.sub(r'("[xy]":\s*)\[[^\]\[\{\}]*\]', r"\1[]", text)

    return text


def test_visualize_training(run_training, run_hp_search, project_root):
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
    model_outputs = {}
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

        with open(output_path, "r") as f:
            vizualization_content = sanitize_html(f.read())

        target_output_path = os.path.join(
            "tests",
            "resources",
            "target_outputs",
            "visualization",
            f"{model}-training-visualization.html",
        )
        with open(target_output_path, "r") as f:
            target_vizualization_content = sanitize_html(f.read())

        model_outputs[model] = (vizualization_content, target_vizualization_content)

    # 2. Test running visualize-training for all models jointly
    models_str = ",".join(hp_models_grid)
    command_joint = (
        f"sequifier visualize-training {models_str} --project-root {project_root}"
    )
    run_and_log(command_joint)

    output_path_joint = os.path.join(
        project_root,
        "outputs",
        "visualization",
        "multi-model-training-visualization.html",
    )

    assert os.path.exists(output_path_joint), "Joint visualization output not found"

    with open(output_path_joint, "r") as f:
        vizualization_content = sanitize_html(f.read())

    target_output_path = os.path.join(
        "tests",
        "resources",
        "target_outputs",
        "visualization",
        "multi-model-training-visualization.html",
    )

    with open(target_output_path, "r") as f:
        target_vizualization_content = sanitize_html(f.read())

    assert (
        vizualization_content.strip() == target_vizualization_content.strip()
    ), f"{vizualization_content}\n!=\n{target_vizualization_content}\n\noutput not identical to target for 'multi-model-training-visualization.html'"

    for model, (
        vizualization_content,
        target_vizualization_content,
    ) in model_outputs.items():
        assert (
            vizualization_content.strip() == target_vizualization_content.strip()
        ), f"{vizualization_content}\n!=\n{target_vizualization_content}\n\noutput not identical to target for '{model}-training-visualization.html'"
