"""Convert current tabular windows to Sequifier tensor batches."""

import argparse
from pathlib import Path

import polars as pl
import torch

from sequifier.io.pt_payload import save_pt_payload, tabular_tensor_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("chunk_size", type=int, nargs="?")
    args = parser.parse_args()
    source = args.input_file
    output = args.output_path or source.with_suffix(".pt")
    if args.chunk_size is not None and args.chunk_size <= 0:
        parser.error("chunk_size must be positive")
    if source.suffix == ".csv":
        data = pl.read_csv(source)
    elif source.suffix == ".parquet":
        data = pl.read_parquet(source)
    else:
        parser.error("input must be CSV or Parquet")

    sequence_columns = [name for name in data.columns if name.isdigit()]
    if not sequence_columns:
        parser.error("input must contain numbered sequence columns")
    features = data.get_column("inputCol").unique().sort().to_list()
    dtype = (
        torch.float32
        if any(data.schema[name].is_float() for name in sequence_columns)
        else torch.int64
    )
    batch = tabular_tensor_batch(
        data,
        dict.fromkeys(features, dtype),
        features,
        len(sequence_columns),
        curriculum=None,
    )
    total = len(batch.sequence_ids)
    if args.chunk_size is None or total <= args.chunk_size:
        save_pt_payload(batch, output)
    else:
        output.mkdir(parents=True, exist_ok=True)
        for index, start in enumerate(range(0, total, args.chunk_size)):
            save_pt_payload(
                batch.slice_rows(start, start + args.chunk_size),
                output / f"{output.name}-{index}.pt",
            )


if __name__ == "__main__":
    main()
