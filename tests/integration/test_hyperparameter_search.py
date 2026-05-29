import glob
import json
import os


def test_hp_search_grid_outputs(run_hp_search, project_root):
    hp_name = "test-hp-search-grid"
    config_dir = os.path.join(project_root, "configs")

    generated_configs = glob.glob(os.path.join(config_dir, f"{hp_name}-run-*.yaml"))
    assert (
        len(generated_configs) == 4
    ), f"Expected 4 grid configs, found {len(generated_configs)}"


def test_hp_search_sample_outputs(run_hp_search, project_root):
    hp_name = "test-hp-search-sample"
    config_dir = os.path.join(project_root, "configs")

    generated_configs = glob.glob(os.path.join(config_dir, f"{hp_name}-run-*.yaml"))
    assert (
        len(generated_configs) == 4
    ), f"Expected 4 sample configs, found {len(generated_configs)}"


def test_hp_search_bayesian_outputs(run_hp_search, project_root):
    hp_name = "test-hp-search-bayesian"
    config_dir = os.path.join(project_root, "configs")

    generated_configs = glob.glob(os.path.join(config_dir, f"{hp_name}-run-*.yaml"))
    assert (
        len(generated_configs) == 4
    ), f"Expected 4 bayesian configs, found {len(generated_configs)}"


def test_hp_search_state(run_hp_search, project_root):
    state_dir = os.path.join(project_root, "state", "optuna")

    assert os.path.exists(os.path.join(state_dir, "test-hp-search-sample.db"))
    assert os.path.exists(os.path.join(state_dir, "test-hp-search-grid.db"))
    assert os.path.exists(os.path.join(state_dir, "test-hp-search-bayesian.db"))
    assert os.path.exists(os.path.join(state_dir, "test-hp-search-custom-eval.db"))


def test_hp_search_inference_feedback_loop(run_hp_search, project_root):
    # Verify that the evaluations directory was populated
    eval_dir = os.path.join(project_root, "outputs", "evaluations")
    assert os.path.exists(eval_dir), f"Evaluation directory {eval_dir} was not created."

    eval_files = [
        f
        for f in os.listdir(eval_dir)
        if f.startswith("test-hp-search-custom-eval-run-") and f.endswith(".json")
    ]

    assert len(eval_files) == 4, f"Expected 4 evaluation JSONs, found {len(eval_files)}"

    for f in eval_files:
        with open(os.path.join(eval_dir, f), "r") as fp:
            metrics = json.load(fp)
            assert "max" in metrics, f"'max' missing in {f}"
            assert "stdev" in metrics, f"'stdev' missing in {f}"

            # Sanity check that metrics were actually calculated
            assert isinstance(metrics["max"], int)
            assert isinstance(metrics["stdev"], float)
