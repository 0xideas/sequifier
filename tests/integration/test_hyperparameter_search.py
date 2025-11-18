import glob
import os


def test_hp_search_grid_outputs(run_hp_search, project_root):
    hp_name = "run_hp_search_grid"
    config_dir = os.path.join(project_root, "configs")
    model_dir = os.path.join(project_root, "models")

    # 1. Verify Configs
    # Pattern: test-hp-search-grid-run-0.yaml, -run-1.yaml, etc.
    generated_configs = glob.glob(os.path.join(config_dir, f"{hp_name}-run-*.yaml"))
    assert (
        len(generated_configs) == 4
    ), f"Expected 4 grid configs, found {len(generated_configs)}"

    # 2. Verify Models
    # Pattern: sequifier-test-hp-search-grid-run-{i}-best-3.pt (epochs=3 in grid config)
    # We expect 'best' and 'last' for each run.
    generated_models = glob.glob(
        os.path.join(model_dir, f"sequifier-{hp_name}-run-*.pt")
    )
    assert (
        len(generated_models) == 8
    ), f"Expected 8 grid models (4 runs * 2 exports), found {len(generated_models)}"


def test_hp_search_sample_outputs(run_hp_search, project_root):
    hp_name = "run_hp_search_sample"
    config_dir = os.path.join(project_root, "configs")
    model_dir = os.path.join(project_root, "models")

    # 1. Verify Configs
    generated_configs = glob.glob(os.path.join(config_dir, f"{hp_name}-run-*.yaml"))
    assert (
        len(generated_configs) == 4
    ), f"Expected 4 sample configs, found {len(generated_configs)}"

    # 2. Verify Models
    # Epochs vary in sample config [2, 3, 4], so filenames will vary (e.g. -best-2.pt, -best-4.pt)
    # but total count remains deterministic based on n_samples.
    generated_models = glob.glob(
        os.path.join(model_dir, f"sequifier-{hp_name}-run-*.pt")
    )
    assert (
        len(generated_models) == 8
    ), f"Expected 8 sample models (4 runs * 2 exports), found {len(generated_models)}"
