import json

import numpy as np
import pandas as pd


def invert_normalization(values, target_column, min_max_values):
    """
    Invert the normalization of values for a target column.

    Args:
        values: Normalized values.
        target_column: Target column name.

    Returns:
        Denormalized values.
    """
    min_ = min_max_values[target_column]["min"]
    max_ = min_max_values[target_column]["max"]
    return np.array(
        [(((v + 1.0) / 2.0) * (max_ - min_)) + min_ for v in values.flatten()]
    ).reshape(*values.shape)


def load_column_attributes(dd_config_path="configs/ddconfigs/data.json"):
    with open(dd_config_path, "r") as f:
        dd_config = json.loads(f.read())
    id_maps = dd_config["id_maps"]
    min_max_values = dd_config["min_max_values"]
    return (min_max_values, id_maps)


def load_ground_truth(path, dd_config_path="configs/ddconfigs/data.json", load_col="0"):
    y = pd.read_parquet(path)
    n_cols = len(set(list(y["inputCol"].values)))

    y.index = np.repeat(np.arange(int(y.shape[0] / n_cols)), n_cols)

    y2 = y.pivot_table(index=y.index, columns="inputCol", values=load_col)

    y2["sequenceId"] = y["sequenceId"].values[::n_cols]

    min_max_values, id_maps = load_column_attributes(dd_config_path)

    id_maps_reversed = {k: {vv: kk for kk, vv in v.items()} for k, v in id_maps.items()}

    col_order = ["sequenceId"] + [c for c in y2.columns if c != "sequenceId"]
    for col in y2.columns:
        if col in id_maps:
            y2[col] = [id_maps_reversed[col][int(v)] for v in y2[col]]
        elif col in min_max_values:
            y2[col] = invert_normalization(y2[col].values, col, min_max_values)
        else:
            pass

    return y2[col_order]
