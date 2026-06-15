import json
import os
import sys

import polars as pl


def main():
    if len(sys.argv) < 2:
        print("Error: Missing run_name argument.")
        sys.exit(1)

    run_name = sys.argv[1]

    preds_path = f"outputs/predictions/sequifier-{run_name}-predictions"

    dfs = []
    for root, dir, files in os.walk(preds_path):
        for file in sorted(list(files)):
            df = pl.read_csv(os.path.join(preds_path, file), infer_schema_length=0)

            df = df.with_columns(pl.all().cast(pl.Int64, strict=False).fill_null(-1))
            dfs.append(df)
    df = pl.concat(dfs)

    max_freqs = df["itemId"].value_counts()["count"].max()
    stdev_freqs = df["itemId"].value_counts()["count"].std()

    eval_dir = "outputs/evaluations"
    os.makedirs(eval_dir, exist_ok=True)
    eval_json_path = os.path.join(eval_dir, f"{run_name}.json")

    with open(eval_json_path, "w") as f:
        f.write(json.dumps({"max": max_freqs, "stdev": stdev_freqs}, indent=2))


if __name__ == "__main__":
    main()
