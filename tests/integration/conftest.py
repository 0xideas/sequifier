import os
import shutil
import time

import pytest
import yaml

SELECTED_COLUMNS = {
    "categorical": {
        1: "itemId",
        3: "itemId,sup1",
        5: "itemId,sup1,sup2,sup4",
        50: "itemId," + ",".join([f"sup{i}" for i in range(1, 50)]),
    },
    "real": {
        1: "itemValue",
        3: "itemValue,sup1,sup2",
        5: "itemValue,sup1,sup2,sup3,sup4",
        50: "itemValue," + ",".join([f"sup{i}" for i in range(1, 50)]),
    },
}


def write_and_log(command: str) -> None:
    os.system(command)
    with open(os.path.join("tests", "integration-test-log.txt"), "a+") as f:
        f.write(f"{command}\n")


@pytest.fixture(scope="session")
def split_groups():
    return {"categorical": 3, "real": 2}


@pytest.fixture(scope="session")
def project_path():
    return os.path.join("tests", "project_folder")


@pytest.fixture(scope="session")
def preprocessing_config_path_cat():
    return os.path.join("tests", "configs", "preprocess-test-categorical.yaml")


@pytest.fixture(scope="session")
def preprocessing_config_path_cat_multitarget():
    return os.path.join(
        "tests", "configs", "preprocess-test-categorical-multitarget.yaml"
    )


@pytest.fixture(scope="session")
def preprocessing_config_path_multi_file():
    return os.path.join("tests", "configs", "preprocess-test-multi-file.yaml")


@pytest.fixture(scope="session")
def preprocessing_config_path_interrupted():
    return os.path.join(
        "tests", "configs", "preprocess-test-categorical-interrupted.yaml"
    )


@pytest.fixture(scope="session")
def preprocessing_config_path_exact():
    return os.path.join("tests", "configs", "preprocess-test-categorical-exact.yaml")


@pytest.fixture(scope="session")
def preprocessing_config_path_exact_pt():
    return os.path.join("tests", "configs", "preprocess-test-categorical-exact-pt.yaml")


@pytest.fixture(scope="session")
def preprocessing_config_path_real():
    return os.path.join("tests", "configs", "preprocess-test-real.yaml")


@pytest.fixture(scope="session")
def training_config_path_cat():
    return os.path.join("tests", "configs", "train-test-categorical.yaml")


@pytest.fixture(scope="session")
def training_config_path_cat_multitarget():
    return os.path.join("tests", "configs", "train-test-categorical-multitarget.yaml")


@pytest.fixture(scope="session")
def training_config_path_real():
    return os.path.join("tests", "configs", "train-test-real.yaml")


@pytest.fixture(scope="session")
def training_config_path_cat_inf_size_1():
    return os.path.join("tests", "configs", "train-test-categorical-inf-size-1.yaml")


@pytest.fixture(scope="session")
def training_config_path_cat_inf_size_3():
    return os.path.join("tests", "configs", "train-test-categorical-inf-size-3.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat():
    return os.path.join("tests", "configs", "infer-test-categorical.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat_multitarget():
    return os.path.join("tests", "configs", "infer-test-categorical-multitarget.yaml")


@pytest.fixture(scope="session")
def inference_config_path_real():
    return os.path.join("tests", "configs", "infer-test-real.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat_inf_size_1():
    return os.path.join("tests", "configs", "infer-test-categorical-inf-size-1.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat_inf_size_3():
    return os.path.join("tests", "configs", "infer-test-categorical-inf-size-3.yaml")


@pytest.fixture(scope="session")
def inference_config_path_real_autoregression():
    return os.path.join("tests", "configs", "infer-test-real-autoregression.yaml")


@pytest.fixture(scope="session")
def inference_config_path_categorical_autoregression():
    return os.path.join(
        "tests", "configs", "infer-test-categorical-autoregression.yaml"
    )


@pytest.fixture(scope="session")
def inference_config_path_embedding():
    return os.path.join("tests", "configs", "infer-test-categorical-embedding.yaml")


@pytest.fixture(scope="session")
def inference_config_path_cat_inf_size_3_embedding():
    return os.path.join(
        "tests", "configs", "infer-test-categorical-inf-size-3-embedding.yaml"
    )


@pytest.fixture(scope="session")
def remove_project_path_contents(project_path):
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    os.makedirs(project_path)

    log_file_path = os.path.join("tests", "integration-test-log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    time.sleep(1)


def reformat_parameter(attr, param, type):
    if attr.endswith("_path"):
        if type == "linux->local":
            return os.path.join(*param.split("/"))
        elif type == "local->linux":
            return "/".join(os.path.split(param))
    else:
        return param


@pytest.fixture(scope="session", autouse=True)
def format_configs_locally(
    preprocessing_config_path_cat,
    preprocessing_config_path_cat_multitarget,
    preprocessing_config_path_real,
    preprocessing_config_path_multi_file,
    preprocessing_config_path_interrupted,
    training_config_path_cat,
    training_config_path_cat_multitarget,
    training_config_path_real,
    training_config_path_cat_inf_size_1,
    training_config_path_cat_inf_size_3,
    inference_config_path_cat,
    inference_config_path_cat_multitarget,
    inference_config_path_real,
    inference_config_path_real_autoregression,
    inference_config_path_categorical_autoregression,
    inference_config_path_cat_inf_size_1,
    inference_config_path_cat_inf_size_3,
):
    from sys import platform

    if platform == "windows":
        config_paths = [
            preprocessing_config_path_cat,
            preprocessing_config_path_cat_multitarget,
            preprocessing_config_path_real,
            preprocessing_config_path_multi_file,
            preprocessing_config_path_interrupted,
            training_config_path_cat,
            training_config_path_cat_multitarget,
            training_config_path_real,
            training_config_path_cat_inf_size_1,
            training_config_path_cat_inf_size_3,
            inference_config_path_cat,
            inference_config_path_cat_multitarget,
            inference_config_path_real,
            inference_config_path_real_autoregression,
            inference_config_path_categorical_autoregression,
            inference_config_path_cat_inf_size_1,
            inference_config_path_cat_inf_size_3,
        ]
        for config_path in config_paths:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            assert config is not None, config_path

            config_formatted = {
                attr: reformat_parameter(attr, param, "linux->local")
                for attr, param in config.items()
            }

            with open(config_path, "w") as f:
                yaml.dump(
                    config_formatted, f, default_flow_style=False, sort_keys=False
                )

        yield

        for config_path in config_paths:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config_formatted = {
                attr: reformat_parameter(attr, param, "local->linux")
                for attr, param in config.items()
            }

            with open(config_path, "w") as f:
                yaml.dump(
                    config_formatted, f, default_flow_style=False, sort_keys=False
                )
    else:
        yield


@pytest.fixture(scope="session")
def copy_interrupted_data():
    os.makedirs(os.path.join("tests", "project_folder", "data"), exist_ok=True)

    source_path = os.path.join(
        "tests", "resources", "source_data", "test-data-categorical-1-interrupted-temp"
    )
    target_path = os.path.join(
        "tests", "project_folder", "data", "test-data-categorical-1-interrupted-temp"
    )

    shutil.copytree(source_path, target_path)


@pytest.fixture(scope="session")
def run_preprocessing(
    preprocessing_config_path_cat,
    preprocessing_config_path_cat_multitarget,
    preprocessing_config_path_real,
    preprocessing_config_path_multi_file,
    preprocessing_config_path_interrupted,
    preprocessing_config_path_exact,
    preprocessing_config_path_exact_pt,
    format_configs_locally,
    remove_project_path_contents,
    copy_interrupted_data,
):
    for data_number in [1, 3, 5, 50]:
        data_path_cat = os.path.join(
            "tests",
            "resources",
            "source_data",
            f"test-data-categorical-{data_number}.csv",
        )
        write_and_log(
            f"sequifier preprocess --config-path={preprocessing_config_path_cat} --data-path={data_path_cat} --selected-columns=None"
        )

        data_path_real = os.path.join(
            "tests", "resources", "source_data", f"test-data-real-{data_number}.csv"
        )
        write_and_log(
            f"sequifier preprocess --config-path={preprocessing_config_path_real} --data-path={data_path_real} --selected-columns={SELECTED_COLUMNS['real'][data_number]}"
        )

    write_and_log(
        f"sequifier preprocess --config-path={preprocessing_config_path_cat_multitarget}"
    )

    write_and_log(
        f"sequifier preprocess --config-path={preprocessing_config_path_multi_file}"
    )

    write_and_log(
        f"sequifier preprocess --config-path={preprocessing_config_path_interrupted}"
    )

    write_and_log(
        f"sequifier preprocess --config-path={preprocessing_config_path_exact}"
    )

    write_and_log(
        f"sequifier preprocess --config-path={preprocessing_config_path_exact_pt}"
    )

    source_path = os.path.join(
        "tests",
        "resources",
        "source_data",
        "test-data-real-1-split1-autoregression.csv",
    )

    target_path = os.path.join(
        "tests", "project_folder", "data", "test-data-real-1-split1-autoregression.csv"
    )

    shutil.copyfile(source_path, target_path)


@pytest.fixture(scope="session")
def run_training(
    run_preprocessing,
    project_path,
    training_config_path_cat,
    training_config_path_real,
    training_config_path_cat_inf_size_1,
    training_config_path_cat_inf_size_3,
    training_config_path_cat_multitarget,
):
    for model_number in [1, 3, 5, 50]:
        metadata_config_path_cat = os.path.join(
            "configs", "metadata_configs", f"test-data-categorical-{model_number}.json"
        )
        model_name_cat = f"model-categorical-{model_number}"
        write_and_log(
            f"sequifier train --config-path={training_config_path_cat} --metadata_config-path={metadata_config_path_cat} --model-name={model_name_cat} --input-columns={SELECTED_COLUMNS['categorical'][model_number]}"
        )

        metadata_config_path_real = os.path.join(
            "configs", "metadata_configs", f"test-data-real-{model_number}.json"
        )
        model_name_real = f"model-real-{model_number}"
        write_and_log(
            f"sequifier train --config-path={training_config_path_real} --metadata_config-path={metadata_config_path_real} --model-name={model_name_real} --input-columns=None"
        )

    write_and_log(
        f"sequifier train --config-path={training_config_path_cat_inf_size_1}"
    )

    write_and_log(
        f"sequifier train --config-path={training_config_path_cat_inf_size_3}"
    )

    write_and_log(
        f"sequifier train --config-path={training_config_path_cat_multitarget}"
    )

    source_path = os.path.join(
        project_path, "models", "sequifier-model-real-1-best-3.pt"
    )
    target_path = os.path.join(
        project_path, "models", "sequifier-model-real-1-best-3-autoregression.pt"
    )

    shutil.copy(source_path, target_path)


@pytest.fixture(scope="session")
def copy_autoregression_model(project_path, run_training):
    model_path = os.path.join(
        project_path, "models", "sequifier-model-categorical-1-best-3.onnx"
    )
    target_path = os.path.join(
        project_path,
        "models",
        "sequifier-model-categorical-1-best-3-autoregression.onnx",
    )
    shutil.copyfile(model_path, target_path)


@pytest.fixture(scope="session")
def run_inference(
    project_path,
    run_training,
    copy_autoregression_model,
    inference_config_path_cat,
    inference_config_path_cat_multitarget,
    inference_config_path_real,
    inference_config_path_real_autoregression,
    inference_config_path_categorical_autoregression,
    inference_config_path_embedding,
    inference_config_path_cat_inf_size_1,
    inference_config_path_cat_inf_size_3,
    inference_config_path_cat_inf_size_3_embedding,
):
    for model_number in [1, 3, 5, 50]:
        model_path_cat = os.path.join(
            "models", f"sequifier-model-categorical-{model_number}-best-3.onnx"
        )
        data_path_cat = os.path.join(
            "data", f"test-data-categorical-{model_number}-split2"
        )
        metadata_config_path_cat = os.path.join(
            "configs", "metadata_configs", f"test-data-categorical-{model_number}.json"
        )
        write_and_log(
            f"sequifier infer --config-path={inference_config_path_cat} --metadata_config-path={metadata_config_path_cat} --model-path={model_path_cat} --data-path={data_path_cat} --input-columns={SELECTED_COLUMNS['categorical'][model_number]}"
        )

        model_path_real = os.path.join(
            "models", f"sequifier-model-real-{model_number}-best-3.pt"
        )
        data_path_real = os.path.join(
            "data", f"test-data-real-{model_number}-split1.parquet"
        )
        metadata_config_path_real = os.path.join(
            "configs", "metadata_configs", f"test-data-real-{model_number}.json"
        )
        write_and_log(
            f"sequifier infer --config-path={inference_config_path_real} --metadata_config-path={metadata_config_path_real} --model-path={model_path_real} --data-path={data_path_real} --input-columns=None"
        )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_cat_multitarget}"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_real_autoregression} --input-columns={SELECTED_COLUMNS['real'][1]}"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_cat_inf_size_1}"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_cat_inf_size_3}"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_categorical_autoregression}  --input-columns=itemId"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_embedding}  --input-columns=itemId"
    )

    write_and_log(
        f"sequifier infer --config-path={inference_config_path_cat_inf_size_3_embedding}"
    )
