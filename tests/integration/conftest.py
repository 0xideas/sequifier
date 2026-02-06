import os
import shutil
import time

import polars as pl
import pytest
import yaml

SELECTED_COLUMNS = {
    "categorical": {
        1: "itemId",
        3: "itemId supCat1",
        5: "itemId supCat1 supCat2 supCat4",
        50: "itemId " + " ".join([f"supCat{i}" for i in range(1, 50)]),
    },
    "real": {
        1: "itemValue",
        3: "itemValue supReal1 supReal2",
        5: "itemValue supReal1 supReal2 supReal3 supReal4",
        50: "itemValue " + " ".join([f"supReal{i}" for i in range(1, 50)]),
    },
}

TARGET_VARIABLE_DICT = {"categorical": "itemId", "real": "itemValue"}


def run_and_log(command: str) -> None:
    os.system(command)
    with open(os.path.join("tests", "integration-test-log.txt"), "a+") as f:
        f.write(f"{command}\n")


@pytest.fixture(scope="session")
def split_groups():
    return {"categorical": 3, "real": 2}


@pytest.fixture(scope="session")
def project_root():
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
def remove_project_root_contents(project_root):
    if os.path.exists(project_root):
        shutil.rmtree(project_root)
    os.makedirs(project_root)

    log_file_path = os.path.join("tests", "integration-test-log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    time.sleep(1)


@pytest.fixture(scope="session")
def hp_search_configs():
    return {
        "grid": os.path.join("tests", "configs", "hyperparameter-search-grid.yaml"),
        "sample": os.path.join("tests", "configs", "hyperparameter-search-sample.yaml"),
    }


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
    hp_search_configs,
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
            hp_search_configs["grid"],
            hp_search_configs["sample"],
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
def copy_interrupted_data(project_root, remove_project_root_contents):
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)

    source_path = os.path.join(
        "tests", "resources", "source_data", "test-data-categorical-1-interrupted-temp"
    )
    target_path = os.path.join(
        project_root, "data", "test-data-categorical-1-interrupted-temp"
    )

    shutil.copytree(source_path, target_path)


@pytest.fixture(scope="session")
def run_preprocessing(
    project_root,
    preprocessing_config_path_cat,
    preprocessing_config_path_cat_multitarget,
    preprocessing_config_path_real,
    preprocessing_config_path_multi_file,
    preprocessing_config_path_interrupted,
    preprocessing_config_path_exact,
    preprocessing_config_path_exact_pt,
    format_configs_locally,
    remove_project_root_contents,
    copy_interrupted_data,
):
    for data_number in [1, 3, 5, 50]:
        data_path_cat = os.path.join(
            "tests",
            "resources",
            "source_data",
            f"test-data-categorical-{data_number}.csv",
        )
        run_and_log(
            f"sequifier preprocess --config-path {preprocessing_config_path_cat} --data-path {data_path_cat} --selected-columns None"
        )

        data_path_real = os.path.join(
            "tests", "resources", "source_data", f"test-data-real-{data_number}.csv"
        )
        run_and_log(
            f"sequifier preprocess --config-path {preprocessing_config_path_real} --data-path {data_path_real} --selected-columns {SELECTED_COLUMNS['real'][data_number]}"
        )

    source_path = os.path.join("tests", "resources", "source_configs", "id_maps")
    target_path = os.path.join(project_root, "configs", "id_maps")
    shutil.copytree(source_path, target_path)
    run_and_log(
        f"sequifier preprocess --config-path {preprocessing_config_path_cat_multitarget}"
    )
    shutil.rmtree(target_path)

    run_and_log(
        f"sequifier preprocess --config-path {preprocessing_config_path_multi_file}"
    )

    run_and_log(
        f"sequifier preprocess --config-path {preprocessing_config_path_interrupted}"
    )

    run_and_log(f"sequifier preprocess --config-path {preprocessing_config_path_exact}")

    run_and_log(
        f"sequifier preprocess --config-path {preprocessing_config_path_exact_pt}"
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
    project_root,
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
        run_and_log(
            f"sequifier train --config-path {training_config_path_cat} --metadata-config-path {metadata_config_path_cat} --model-name {model_name_cat} --input-columns {SELECTED_COLUMNS['categorical'][model_number]}"
        )

        metadata_config_path_real = os.path.join(
            "configs", "metadata_configs", f"test-data-real-{model_number}.json"
        )
        model_name_real = f"model-real-{model_number}"
        run_and_log(
            f"sequifier train --config-path {training_config_path_real} --metadata-config-path {metadata_config_path_real} --model-name {model_name_real} --input-columns None"
        )

    run_and_log(f"sequifier train --config-path {training_config_path_cat_inf_size_1}")

    run_and_log(f"sequifier train --config-path {training_config_path_cat_inf_size_3}")

    run_and_log(f"sequifier train --config-path {training_config_path_cat_multitarget}")

    source_path = os.path.join(
        project_root, "models", "sequifier-model-real-1-best-3.pt"
    )
    target_path = os.path.join(
        project_root, "models", "sequifier-model-real-1-best-3-autoregression.pt"
    )

    shutil.copy(source_path, target_path)


@pytest.fixture(scope="module")
def run_hp_search(
    project_root, hp_search_configs, format_configs_locally, run_preprocessing
):
    run_and_log(
        f"sequifier hyperparameter-search --config-path {hp_search_configs['grid']}"
    )

    run_and_log(
        f"sequifier hyperparameter-search --config-path {hp_search_configs['sample']}"
    )


@pytest.fixture(scope="session")
def copy_autoregression_model(project_root, run_training):
    model_path = os.path.join(
        project_root, "models", "sequifier-model-categorical-1-best-3.onnx"
    )
    target_path = os.path.join(
        project_root,
        "models",
        "sequifier-model-categorical-1-best-3-autoregression.onnx",
    )
    shutil.copyfile(model_path, target_path)


@pytest.fixture(scope="session")
def run_inference(
    project_root,
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
        run_and_log(
            f"sequifier infer --config-path {inference_config_path_cat} --metadata-config-path {metadata_config_path_cat} --model-path {model_path_cat} --data-path {data_path_cat} --input-columns {SELECTED_COLUMNS['categorical'][model_number]}"
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
        run_and_log(
            f"sequifier infer --config-path {inference_config_path_real} --metadata-config-path {metadata_config_path_real} --model-path {model_path_real} --data-path {data_path_real} --input-columns None"
        )

    run_and_log(
        f"sequifier infer --config-path {inference_config_path_cat_multitarget}"
    )

    run_and_log(
        f"sequifier infer --config-path {inference_config_path_real_autoregression} --input-columns {SELECTED_COLUMNS['real'][1]} --randomize"
    )

    run_and_log(f"sequifier infer --config-path {inference_config_path_cat_inf_size_1}")

    run_and_log(f"sequifier infer --config-path {inference_config_path_cat_inf_size_3}")

    run_and_log(
        f"sequifier infer --config-path {inference_config_path_categorical_autoregression}  --input-columns itemId"
    )

    run_and_log(
        f"sequifier infer --config-path {inference_config_path_embedding}  --input-columns itemId"
    )

    run_and_log(
        f"sequifier infer --config-path {inference_config_path_cat_inf_size_3_embedding}"
    )


@pytest.fixture()
def model_names_preds():
    model_names_preds = [
        f"model-{variant}-{model_number}-best-3"
        for variant in ["categorical", "real"]
        for model_number in [1, 3, 5, 50]
    ]
    model_names_preds += [
        "model-categorical-multitarget-5-best-3",
        "model-real-1-best-3-autoregression",
        "model-categorical-1-best-3-autoregression",
        "model-categorical-1-inf-size-best-3",
        "model-categorical-3-inf-size-best-3",
    ]

    return model_names_preds


@pytest.fixture()
def model_names_probs():
    model_names_probs = [
        f"model-categorical-{model_number}-best-3-itemId"
        for model_number in [1, 3, 5, 50]
    ]
    model_names_probs += [
        f"model-categorical-multitarget-5-best-3-{col}" for col in ["itemId", "supCat1"]
    ]
    model_names_probs += [
        f"model-categorical-3-inf-size-best-3-{col}"
        for col in ["itemId", "supCat1", "supCat2"]
    ]
    return model_names_probs


@pytest.fixture()
def model_names_embeddings():
    model_names_embeddings = [
        "model-categorical-1-best-embedding-3",
        "model-categorical-3-inf-size-best-embedding-3",
    ]
    return model_names_embeddings


@pytest.fixture()
def targets(model_names_preds, model_names_probs, model_names_embeddings):
    target_dict = {"preds": {}, "probs": {}, "embeds": {}}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-predictions"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "predictions", file_name
        )
        target_dict["preds"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    for model_name in model_names_probs:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-probabilities"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "probabilities", file_name
        )
        target_dict["probs"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    for model_name in model_names_embeddings:
        target_type = "categorical" if "categorical" in model_name else "real"
        file_name = f"sequifier-{model_name}-embeddings"
        target_path = os.path.join(
            "tests", "resources", "target_outputs", "embeddings", file_name
        )
        target_dict["embeds"][model_name] = read_multi_file_preds(
            target_path, target_type
        )

    return target_dict


def read_multi_file_preds(path, target_type, file_suffix=None):
    dtype = (
        {
            **{TARGET_VARIABLE_DICT[target_type]: str},
            **{f"supCat{i+1}": str for i in range(50)},
        }
        if target_type == "categorical"
        else None
    )
    if target_type == "categorical":
        contents = []
        for root, dirs, files in os.walk(path):
            for file in sorted(list(files)):
                if file_suffix is None or file.endswith(file_suffix):
                    contents.append(
                        pl.read_csv(os.path.join(root, file), schema_overrides=dtype)
                    )
        assert len(contents) > 0, f"no files found for {path}"
        return pl.concat(contents, how="vertical")
    else:
        return pl.read_csv(f"{path}.csv", separator=",", schema_overrides=dtype)


@pytest.fixture()
def predictions(run_inference, model_names_preds, project_root):
    preds = {}
    for model_name in model_names_preds:
        target_type = "categorical" if "categorical" in model_name else "real"

        prediction_path = os.path.join(
            project_root,
            "outputs",
            "predictions",
            f"sequifier-{model_name}-predictions",
        )

        preds[model_name] = read_multi_file_preds(prediction_path, target_type)

    return preds


@pytest.fixture()
def probabilities(run_inference, model_names_probs, project_root):
    probs = {}
    for model_name in model_names_probs:
        probabilities_path = os.path.join(
            project_root,
            "outputs",
            "probabilities",
            f"sequifier-{model_name}-probabilities",
        )
        probs[model_name] = read_multi_file_preds(
            probabilities_path, "categorical", "csv"
        )

    return probs


@pytest.fixture()
def embeddings(run_inference, model_names_embeddings, project_root):
    embeds = {}
    for model_name in model_names_embeddings:
        embeddings_path = os.path.join(
            project_root,
            "outputs",
            "embeddings",
            f"sequifier-{model_name}-embeddings",
        )
        embeds[model_name] = read_multi_file_preds(
            embeddings_path, "categorical", "csv"
        )
    return embeds
