import json
import os

import numpy as np
import polars as pl
import pytest
import torch


@pytest.fixture()
def metadata_configs(run_preprocessing, project_root):
    metadata_configs = {}
    for data_number in [1, 3, 5, 50]:
        for variant in ["categorical", "real"]:
            file_name = f"test-data-{variant}-{data_number}.json"
            with open(
                os.path.join(project_root, "configs", "metadata_configs", file_name),
                "r",
            ) as f:
                dd_conf = json.loads(f.read())
            metadata_configs[file_name] = dd_conf
    return metadata_configs


def test_metadata_config(metadata_configs):
    for file_name, metadata_config in metadata_configs.items():
        print(f"Verifying metadata_config for: {file_name}")
        assert np.all(
            np.array(list(metadata_config.keys()))
            == np.array(
                [
                    "n_classes",
                    "id_maps",
                    "split_paths",
                    "column_types",
                    "selected_columns_statistics",
                ]
            )
        ), list(metadata_config.keys())

        assert metadata_config["split_paths"][0].endswith(
            "split0.parquet"
        ) or metadata_config["split_paths"][0].endswith("split0")

        if "itemId" in metadata_config["n_classes"]:
            assert len(metadata_config["id_maps"]["itemId"]) == 30
            assert metadata_config["n_classes"]["itemId"] == 32

            id_map_keys = np.array(
                sorted(list(metadata_config["id_maps"]["itemId"].keys()))
            )
            # assert False, np.array([str(x) for x in range(100, 130)])
            assert np.all(id_map_keys == np.array([str(x) for x in range(100, 130)]))

        for col in metadata_config["id_maps"].keys():
            id_map_values = np.array(
                sorted(list(metadata_config["id_maps"][col].values()))
            )
            # assert False, id_map_values
            assert np.all(
                id_map_values == np.arange(2, len(id_map_values) + 2)
            ), id_map_values

        if "itemValue" in metadata_config["selected_columns_statistics"]:
            assert "std" in metadata_config["selected_columns_statistics"]["itemValue"]
            assert "mean" in metadata_config["selected_columns_statistics"]["itemValue"]


def load_pt_outputs(path):
    contents = []
    for root, _, files in os.walk(path):
        for file in sorted(list(files)):
            if file.endswith("pt"):
                (
                    sequences,
                    targets,
                    sequence_id,
                    subsequence_id,
                    start_item_position,
                ) = torch.load(os.path.join(root, file))
                sequences2 = {}
                for col, vals in sequences.items():
                    vals2 = np.concatenate(
                        [vals.numpy(), targets[col][:, -1:].numpy()], axis=1
                    )

                    for offset in range(vals2.shape[1] - 1, -1, -1):
                        sequences2[str(offset)] = np.concatenate(
                            [sequences2.get(str(offset), []), vals2[:, -(offset + 1)]],
                            axis=0,
                        )

                    sequences2["sequenceId"] = np.concatenate(
                        [
                            sequences2.get("sequenceId", []),
                            sequence_id.numpy().astype(int),
                        ],
                        axis=0,
                    )
                    sequences2["inputCol"] = np.concatenate(
                        [
                            sequences2.get("inputCol", []),
                            np.repeat(col, vals2.shape[0]),
                        ],
                        axis=0,
                    )
                    sequences2["subsequenceId"] = np.concatenate(
                        [sequences2.get("subsequenceId", []), subsequence_id],
                        axis=0,
                    )
                    sequences2["startItemPosition"] = np.concatenate(
                        [
                            sequences2.get("startItemPosition", []),
                            start_item_position,
                        ]
                    )

                content = pl.DataFrame(sequences2)
                contents.append(content)

    assert len(contents) > 0, f"no files found for {path}"
    data = pl.concat(contents, how="vertical")
    other_cols = [
        col
        for col in data.columns
        if col not in ["sequenceId", "subsequenceId", "startItemPosition", "inputCol"]
    ]
    return data[
        ["sequenceId", "subsequenceId", "startItemPosition", "inputCol"] + other_cols
    ].sort(["sequenceId", "subsequenceId", "startItemPosition", "inputCol"])


def read_preprocessing_outputs(path, variant):
    if variant == "real":
        return pl.read_parquet(f"{path}.parquet")
    elif variant == "categorical":
        return load_pt_outputs(path)


@pytest.fixture()
def data_splits(project_root, split_groups):
    data_split_values = {
        f"{j}-{variant}": [
            read_preprocessing_outputs(
                os.path.join(project_root, "data", f"test-data-{variant}-{j}-split{i}"),
                variant,
            )
            for i in range(split_groups[variant])
        ]
        for variant in ["categorical", "real"]
        for j in [1, 3, 5, 50]
    }

    return data_split_values


def test_preprocessed_data_real(data_splits):
    for j in [1, 3, 5, 50]:
        name = f"{j}-real"
        assert len(data_splits[name]) == 2

        for i, data in enumerate(data_splits[name]):
            number_expected_columns = 13
            assert data.shape[1] == (
                number_expected_columns
            ), f"{name = } - {i = }: {data.shape = } - {data.columns = }"
            for sequenceId, group in data.group_by("sequenceId"):
                # offset by j in either direction as that is the number of columns in the input
                # data, thus an offset by 1 'observation' requires an offset by j values
                assert np.all((group["1"].to_numpy()[:-j] == group["2"].to_numpy()[j:]))
                assert np.all((group["5"].to_numpy()[:-j] == group["6"].to_numpy()[j:]))


def test_preprocessed_data_categorical(data_splits):
    for j in [1, 3, 5, 50]:
        name = f"{j}-categorical"
        assert len(data_splits[name]) == 3

        for i, data in enumerate(data_splits[name]):
            number_expected_columns = 13
            assert data.shape[1] == (
                number_expected_columns
            ), f"{name = } - {i = }: {data.shape = } - {data.columns = }"

            for sequenceId, group in data.group_by("sequenceId"):
                # offset by j in either direction as that is the number of columns in the input
                # data, thus an offset by 1 'observation' requires an offset by j values
                assert np.all(
                    np.abs(group["1"].to_numpy()[:-j] - group["2"].to_numpy()[j:])
                    < 0.0001
                ), f'{list(group["1"].to_numpy()[:-j]) = } != {list(group["2"].to_numpy()[j:]) = }'
                assert np.all(
                    np.abs(group["5"].to_numpy()[:-j] - group["6"].to_numpy()[j:])
                    < 0.0001
                ), f'{list(group["5"].to_numpy()[:-j]) = } != {list(group["6"].to_numpy()[j:]) = }'


def unnest(list_var):
    return [x for y in list_var for x in y]


def test_preprocessed_data_multi_file(run_preprocessing):
    for split in range(3):
        file_list = []
        for root, _, files in os.walk(
            os.path.join(
                "tests",
                "project_folder",
                "data",
                f"test-data-categorical-multi-file-split{split}",
            )
        ):
            for file in files:
                file_list.append(file)

        file_list = sorted(file_list)

        expected_file_list = sorted(
            ["metadata.json"]
            + unnest(
                [
                    [
                        f"test-data-categorical-multi-file-{source_file}-0-split{split}-0-{str(seq_id).zfill(2)}.pt"
                    ]
                    for source_file in range(3)
                    for seq_id in range(13)
                ]
            )
        )
        assert len(file_list) == len(
            expected_file_list
        ), f"{file_list = }, {expected_file_list = }"
        assert np.all(
            np.array(file_list) == np.array(expected_file_list)
        ), f"for split: {split}:\n{set(file_list).difference(set(expected_file_list))} not found\n{set(expected_file_list).difference(set(file_list))} extra"


def test_preprocessed_data_exact(run_preprocessing):
    parquet_out_path = os.path.join(
        "tests",
        "project_folder",
        "data",
        "test-data-categorical-3-equal-split0.parquet",
    )
    pt_out_path = os.path.join(
        "tests", "project_folder", "data", "test-data-categorical-3-equal-split0"
    )

    parquet_output = pl.read_parquet(parquet_out_path)
    pt_output = load_pt_outputs(pt_out_path)

    assert np.all(
        parquet_output.to_numpy()[:, [0, 1, 2, 4, 5, 6, 7, 8, 9]]
        == pt_output.to_numpy()[:, [0, 1, 2, 4, 5, 6, 7, 8, 9]].astype(int)
    ), f"{np.sum(parquet_output.to_numpy()[:,[0,1,2,4,5,6,7,8,9]] == pt_output.to_numpy()[:,[0,1,2,4,5,6,7,8,9]].astype(int)) = }"

    assert np.all(
        parquet_output["sequenceId"].to_numpy() == np.repeat(np.arange(10), 9)
    )

    assert np.all(
        parquet_output["subsequenceId"].to_numpy()
        == np.tile(np.repeat(np.arange(3), 3), 10)
    )


def test_preprocessing_interrupted(run_preprocessing, metadata_configs):
    with open(
        os.path.join(
            "tests",
            "project_folder",
            "configs",
            "metadata_configs",
            "test-data-categorical-1-interrupted.json",
        ),
        "r",
    ) as f:
        interrupted_metadata_config = json.loads(f.read())
    baseline_metadata_config = metadata_configs["test-data-categorical-1.json"]

    interrupted_metadata_config_adapted = interrupted_metadata_config
    interrupted_metadata_config_adapted["split_paths"] = [
        path.replace("categorical-1-interrupted", "categorical-1")
        for path in interrupted_metadata_config_adapted["split_paths"]
    ]

    assert str(interrupted_metadata_config_adapted) == str(
        baseline_metadata_config
    ), f"{interrupted_metadata_config_adapted = } != {baseline_metadata_config = }"

    baseline_output = {
        split: load_pt_outputs(
            os.path.join(
                "tests",
                "project_folder",
                "data",
                f"test-data-categorical-1-split{split}",
            )
        )
        for split in range(3)
    }
    interrupted_output = {
        split: load_pt_outputs(
            os.path.join(
                "tests",
                "project_folder",
                "data",
                f"test-data-categorical-1-interrupted-split{split}",
            )
        )
        for split in range(3)
    }

    for split in range(3):
        assert np.all(
            interrupted_output[split].to_numpy() == baseline_output[split].to_numpy()
        ), f"interrupted output != baseline output for split {split}"
