import os
import shutil

import pytest
import yaml
from conftest import reformat_parameter

test_project_name = os.path.join("tests", "sequifier-make-test-project")


@pytest.fixture
def setup_for_test_make():
    os.system(f"sequifier make {test_project_name}")

    preprocess_config_path = os.path.join(
        test_project_name, "configs", "preprocess.yaml"
    )
    with open(preprocess_config_path, "r") as f:
        preprocess_config_string = f.read()

        preprocess_config_string = (
            preprocess_config_string.replace(
                "project_path: .", f"project_path: {test_project_name}"
            )
            .replace(
                "data_path: PLEASE FILL",
                "data_path: tests/resources/test-data-categorical-1.csv",
            )
            .replace(
                "selected_columns: [EXAMPLE_INPUT_COLUMN_NAME]", "selected_columns: "
            )
            .replace("seq_length: 48", "seq_length: 10")
            .replace("max_rows: null", "max_rows: null\nn_cores: 1")
        )

    with open(preprocess_config_path, "w") as f:
        f.write(preprocess_config_string)

    train_config_path = os.path.join(test_project_name, "configs", "train.yaml")
    with open(train_config_path, "r") as f:
        train_config_string = f.read()

        train_config_string = (
            train_config_string.replace(
                "project_path: .", f"project_path: {test_project_name}"
            )
            .replace(
                "ddconfig_path: PLEASE FILL",
                f"ddconfig_path: {test_project_name}/configs/ddconfigs/test-data-categorical-1.json",
            )
            .replace("[EXAMPLE_INPUT_COLUMN_NAME]", "[itemId]")
            .replace("[EXAMPLE_TARGET_COLUMN_NAME]", "[itemId]")
            .replace("EXAMPLE_TARGET_COLUMN_NAME: real", "itemId: categorical")
            .replace("EXAMPLE_INPUT_COLUMN_NAME:", "itemId: 128")
            .replace("EXAMPLE_TARGET_COLUMN_NAME: MSELoss", "itemId: CrossEntropyLoss")
            .replace("epochs: 1000", "epochs: 3")
            .replace("device: cuda", "device: cpu")
            .replace("seq_length: 48", "seq_length: 10")
        )

    with open(train_config_path, "w") as f:
        f.write(train_config_string)

    infer_config_path = os.path.join(test_project_name, "configs", "infer.yaml")
    with open(infer_config_path, "r") as f:
        infer_config_string = f.read()

        infer_config_string = (
            infer_config_string.replace(
                "project_path: .", f"project_path: {test_project_name}"
            )
            .replace(
                "ddconfig_path: PLEASE FILL",
                f"ddconfig_path: {test_project_name}/configs/ddconfigs/test-data-categorical-1.json",
            )
            .replace(
                "model_path: PLEASE FILL",
                f"model_path: {test_project_name}/models/sequifier-default-best-3.onnx",
            )
            .replace(
                "data_path: PLEASE FILL",
                f"data_path: {test_project_name}/data/test-data-categorical-1-split2.parquet",
            )
            .replace("[EXAMPLE_INPUT_COLUMN_NAME]", "[itemId]")
            .replace("[EXAMPLE_TARGET_COLUMN_NAME]", "[itemId]")
            .replace("seq_length: 48", "seq_length: 10")
            .replace("autoregression: true", "autoregression: false")
            .replace("EXAMPLE_TARGET_COLUMN_NAME: real", "itemId: categorical")
            .replace("map_to_id: false", "map_to_id: true")
        )

    with open(infer_config_path, "w") as f:
        f.write(infer_config_string)

    from sys import platform

    if platform == "windows":
        for config_path in [
            preprocess_config_path,
            train_config_path,
            infer_config_path,
        ]:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config_formatted = {
                attr: reformat_parameter(attr, param, "linux->local")
                for attr, param in config.items()
            }

            with open(config_path, "w") as f:
                yaml.dump(
                    config_formatted, f, default_flow_style=False, sort_keys=False
                )


def test_make(setup_for_test_make):
    return_code = os.system(
        f"sequifier preprocess --config-path {test_project_name}/configs/preprocess.yaml"
    )
    if return_code == 0:
        return_code = os.system(
            f"sequifier train --config-path {test_project_name}/configs/train.yaml"
        )
        if return_code == 0:
            return_code = os.system(
                f"sequifier infer --config-path {test_project_name}/configs/infer.yaml"
            )
            assert (
                return_code == 0
            ), f"Inference for 'sequifier infer --config-path {test_project_name}/configs/infer.yaml' was unsuccessful"
        else:
            assert False, f"Training for 'sequifier train --config-path {test_project_name}/configs/train.yaml' was unsuccessful"
    else:
        assert False, f"Preprocessing for 'sequifier preprocess --config-path {test_project_name}/configs/train.yaml' was unsuccessful"

    # clean up, only if tests didn't fail
    shutil.rmtree(test_project_name)
