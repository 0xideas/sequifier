import json

import numpy as np
import pandas as pd


def invert_normalization(values, target_column, selected_columns_statistics):
    """
    Invert the normalization of values for a target column.

    Args:
        values: Normalized values.
        target_column: Target column name.

    Returns:
        Denormalized values.
    """
    std = selected_columns_statistics[target_column]["std"]
    mean = selected_columns_statistics[target_column]["mean"]
    return (values * std) + mean


def load_column_attributes(dd_config_path="configs/ddconfigs/data.json"):
    with open(dd_config_path, "r") as f:
        dd_config = json.loads(f.read())
    id_maps = dd_config["id_maps"]
    selected_columns_statistics = dd_config["selected_columns_statistics"]
    return (selected_columns_statistics, id_maps)


def load_ground_truth(path, dd_config_path="configs/ddconfigs/data.json", load_col="0"):
    y = pd.read_parquet(path)
    n_cols = len(set(list(y["inputCol"].values)))

    y.index = np.repeat(np.arange(int(y.shape[0] / n_cols)), n_cols)

    y2 = y.pivot_table(index=y.index, columns="inputCol", values=load_col)

    y2["sequenceId"] = y["sequenceId"].values[::n_cols]

    selected_columns_statistics, id_maps = load_column_attributes(dd_config_path)

    id_maps_reversed = {k: {vv: kk for kk, vv in v.items()} for k, v in id_maps.items()}

    col_order = ["sequenceId"] + [c for c in y2.columns if c != "sequenceId"]
    for col in y2.columns:
        if col in id_maps:
            y2[col] = [id_maps_reversed[col][int(v)] for v in y2[col]]
        elif col in selected_columns_statistics:
            y2[col] = invert_normalization(
                y2[col].values, col, selected_columns_statistics
            )
        else:
            pass

    return y2[col_order]
