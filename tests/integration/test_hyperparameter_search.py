import glob
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
