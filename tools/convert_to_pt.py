import os
import sys

import numpy as np
import polars as pl
import torch


def main():
    assert (
        len(sys.argv) >= 2
    ), "Usage: python convert_to_pt.py <input_file> [output_path] [chunk_size]"
    in_path = sys.argv[1]
    out_path = (
        sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(in_path)[0] + ".pt"
    )
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else None

    assert os.path.exists(in_path), f"Input file not found: {in_path}"

    # 1. Read Data
    if in_path.endswith(".csv"):
        df = pl.read_csv(in_path)
    elif in_path.endswith(".parquet"):
        df = pl.read_parquet(in_path)
    else:
        raise ValueError("Input must be .csv or .parquet")

    # 2. Validate Schema
    required_meta = {"sequenceId", "subsequenceId", "startItemPosition", "inputCol"}
    assert required_meta.issubset(
        df.columns
    ), f"Missing columns: {required_meta - set(df.columns)}"

    seq_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    assert len(seq_cols) > 1, "No numbered sequence columns found"

    # 3. Aggregate
    feature_names = df["inputCol"].unique().to_list()
    aggs = [
        pl.concat_list(seq_cols).filter(pl.col("inputCol") == f).flatten().alias(f)
        for f in feature_names
    ]
    aggs.append(pl.col("startItemPosition").first())

    grouped = df.group_by(["sequenceId", "subsequenceId"], maintain_order=True).agg(
        aggs
    )
    grouped = grouped.sort(["sequenceId", "subsequenceId"])

    # 4. Create Tensors
    seq_ids = torch.from_numpy(grouped["sequenceId"].to_numpy().astype(np.int64))
    sub_ids = torch.from_numpy(grouped["subsequenceId"].to_numpy().astype(np.int64))
    start_pos = torch.from_numpy(
        grouped["startItemPosition"].to_numpy().astype(np.int64)
    )

    sequences_dict = {}
    targets_dict = {}

    for f in feature_names:
        is_float = df.schema[seq_cols[0]] in (pl.Float32, pl.Float64)
        dtype = torch.float32 if is_float else torch.int64

        data_np = np.stack(grouped[f].to_list())
        tensor = torch.tensor(data_np, dtype=dtype)

        sequences_dict[f] = tensor[:, :-1]
        targets_dict[f] = tensor[:, 1:]

    total_rows = len(seq_ids)

    # 5. Save
    if chunk_size is not None and total_rows > chunk_size:
        assert not os.path.isfile(
            out_path
        ), f"Output path '{out_path}' exists as a file, but chunking requires a directory."
        os.makedirs(out_path, exist_ok=True)

        print(f"Converting {in_path} -> {out_path}/ (chunked by {chunk_size})")
        dirname = os.path.split(out_path)[-1]

        for i in range(0, total_rows, chunk_size):
            end = i + chunk_size

            # Slice everything
            s_ids_chunk = seq_ids[i:end]
            sub_ids_chunk = sub_ids[i:end]
            start_pos_chunk = start_pos[i:end]
            seq_chunk = {k: v[i:end] for k, v in sequences_dict.items()}
            tgt_chunk = {k: v[i:end] for k, v in targets_dict.items()}

            file_name = f"{dirname}-{i // chunk_size}.pt"
            full_name = os.path.join(out_path, file_name)

            torch.save(
                (seq_chunk, tgt_chunk, s_ids_chunk, sub_ids_chunk, start_pos_chunk),
                full_name,
            )
    else:
        print(f"Converting {in_path} -> {out_path}")
        print(f"Samples: {total_rows} | Features: {feature_names}")
        torch.save(
            (sequences_dict, targets_dict, seq_ids, sub_ids, start_pos), out_path
        )


if __name__ == "__main__":
    main()
