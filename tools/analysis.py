import json

import polars as pl


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
    return (values * (std - 1e-9)) + mean


def load_column_attributes(metadata_config_path="configs/metadata_configs/data.json"):
    with open(metadata_config_path, "r") as f:
        metadata_config = json.loads(f.read())
    id_maps = metadata_config["id_maps"]
    selected_columns_statistics = metadata_config["selected_columns_statistics"]
    return (selected_columns_statistics, id_maps)


def load_ground_truth(
    path, metadata_config_path="configs/metadata_configs/data.json", load_col="0"
):
    y = pl.read_parquet(path)
    n_cols = y["inputCol"].n_unique()
    y = y.with_columns(group_id=pl.int_range(0, pl.count() // n_cols).repeat_by(n_cols))

    y2 = y.pivot(index="group_id", columns="inputCol", values=load_col)

    y2["sequenceId"] = y["sequenceId"].values[::n_cols]

    selected_columns_statistics, id_maps = load_column_attributes(metadata_config_path)

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
