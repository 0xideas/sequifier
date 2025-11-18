import os

import numpy as np
import polars as pl
import torch


def reconstruct_data(contents, cols):
    rows = []

    vals = np.array([])
    for file in contents.keys():
        col_vals = {}
        for col in cols:
            vals = np.array(contents[file][0][col])
            target_vals = np.array(contents[file][1][col])

            assert np.all(vals[:, 1:] == target_vals[:, :-1])

            # print(f"{file = }, {col = }")
            vals = np.concatenate([vals, target_vals[:, -1:]], axis=1)
            col_vals[col] = vals

        for i in range(vals.shape[0]):
            for col in cols:
                rows.append([col] + list(np.array(col_vals[col][i, :]).flatten()))

    assert vals is not None
    schema = ["inputCol"] + [str(i) for i in range(vals.shape[1] - 1, -1, -1)]
    # print(f"{len(schema) = }, {vals.shape[1] = }")

    data = pl.DataFrame(rows, schema=schema, orient="row")

    return data


def apply_metadata_config_to_raw_data(data, metadata_config):
    for col, id_map in metadata_config["id_maps"].items():
        data = data.with_columns(pl.col(col).replace(id_map))

    for col, stats in metadata_config["selected_columns_statistics"].items():
        std, mean = stats["std"], stats["mean"]
        data = data.with_columns(((pl.col(col) - mean) / (std + 1e-9)).alias(col))

    return data


def find_sequence_in_raw_data(data, col, sequence):
    n = len(sequence)
    found_indices = []
    for i in range(data.shape[0] - n):
        test = True
        for j, s in enumerate(sequence):
            if test and data[i + j, col] != s:
                test = False

        if test:
            found_indices.append(i)
            print(i)
    return found_indices


def equal(a, b):
    return a == b


def find_sequence_in_preprocessed_data(data, col, sequence, fn=equal):
    n = len(sequence)
    col_index = np.where(np.array(list(data.columns)) == "inputCol")[0][0]
    found_indices = []
    for i, row in enumerate(data.iter_rows(named=False)):
        if row[col_index] == col:
            test = True
            for j, e in enumerate(row[col_index + 1 :]):
                if test and j < n:
                    # print(f"{e = }, {sequence[j] = }, {fn(e, sequence[j]) = }")
                    if not fn(e, sequence[j]):
                        test = False
            if test:
                found_indices.append(i)
    return found_indices


def compare_preprocessed_data(data1, data2, col, fn, n_cols=4):
    col_index = np.where(np.array(list(data1.columns)) == "inputCol")[0][0]

    start_sequence_cols = list(data1.columns)[col_index + 1 : col_index + 1 + n_cols]
    # print(start_sequence_cols)

    for i in [int(ii) for ii in np.where(data1["inputCol"].to_numpy() == col)[0]]:
        target_sequence = list(data1[i, start_sequence_cols].to_numpy().flatten())
        # print(f"{target_sequence = }")
        found_indices = find_sequence_in_preprocessed_data(
            data2, col, target_sequence, fn
        )

        assert len(found_indices) >= 1, f"{found_indices = }"


def load_pt_data(path, cols):
    contents = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            # print(file)
            try:
                contents[file] = torch.load(
                    os.path.join(root, file), map_location="cpu"
                )
            except Exception as e:
                print(f"failed: {file}: {e}")

    data = reconstruct_data(contents, cols)

    return data


def load_and_compare_pt_and_parquet(parquet_path, pt_path, n_cols):
    data1 = pl.read_parquet(parquet_path)
    ci = [
        int(ii)
        for ii in np.where(data1["inputCol"].to_numpy() == data1[0, "inputCol"])[0]
    ]
    cols = data1["inputCol"].to_numpy()[: ci[1]]
    data2 = load_pt_data(pt_path, cols)

    for col in cols:
        compare_preprocessed_data(
            data1, data2, col, lambda a, b: np.abs(a - b) < 0.001, n_cols=n_cols
        )
        compare_preprocessed_data(
            data2, data1, col, lambda a, b: np.abs(a - b) < 0.001, n_cols=n_cols
        )
