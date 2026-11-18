"""Rechunk current Sequifier tensor batches, retaining split metadata."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

from sequifier.config.metadata import load_dataset_metadata
from sequifier.io.atomic import write_json_atomic
from sequifier.io.pt_payload import (
    concatenate_pt_batches,
    load_pt_payload,
    save_pt_payload,
)


def process_split(input_dir, output_dir, dataset_name, target_size_mb, split_suffix):
    source = Path(input_dir)
    destination = Path(output_dir)
    metadata = load_dataset_metadata(str(source / "metadata.json"))
    manifest = json.loads((source / "metadata.json").read_text())
    options = {"layouts": metadata.depth_layouts, "n_classes": metadata.n_classes}
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    batches = []
    buffered_rows = 0
    target_rows = None

    def write_shard(batch):
        filename = f"{dataset_name}-{split_suffix}-{len(entries)}.pt"
        save_pt_payload(batch, destination / filename, **options)
        first_target = torch.maximum(
            batch.left_pad_lengths,
            batch.split_start_item_positions - batch.start_item_positions,
        )
        entries.append(
            {
                "path": filename,
                "samples": len(batch.sequence_ids),
                "left_pad_length_histogram": dict(
                    Counter(batch.left_pad_lengths.tolist())
                ),
                "target_valid_from_histogram": dict(Counter(first_target.tolist())),
            }
        )

    for entry in manifest["batch_files"]:
        batch = load_pt_payload(source / entry["path"], **options)
        count = len(batch.sequence_ids)
        if count != entry["samples"]:
            raise ValueError(
                f"{entry['path']}: tensor batch size differs from metadata"
            )
        if count == 0:
            continue
        if target_rows is None:
            size = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    *batch.sequences.values(),
                    *batch.depth_valid_masks.values(),
                    batch.sequence_ids,
                    batch.subsequence_ids,
                    batch.start_item_positions,
                    batch.left_pad_lengths,
                    batch.split_start_item_positions,
                    batch.split_end_item_positions,
                    batch.sample_positions,
                )
                if tensor is not None
            )
            target_rows = max(1, int(target_size_mb * 1024 * 1024 * count / size))
        batches.append(batch)
        buffered_rows += count
        if buffered_rows >= target_rows:
            combined = concatenate_pt_batches(batches)
            complete_rows = buffered_rows // target_rows * target_rows
            for start in range(0, complete_rows, target_rows):
                write_shard(combined.slice_rows(start, start + target_rows))
            batches = (
                [combined.slice_rows(complete_rows, buffered_rows)]
                if complete_rows < buffered_rows
                else []
            )
            buffered_rows -= complete_rows
    if batches:
        write_shard(concatenate_pt_batches(batches))
    count = sum(entry["samples"] for entry in entries)
    if count != manifest["total_samples"]:
        raise ValueError(f"{source}: total sample count differs from metadata")
    write_json_atomic(
        destination / "metadata.json",
        {**manifest, "total_samples": count, "batch_files": entries},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fast Rechunker for Sequifier Datasets"
    )
    parser.add_argument("data_folder", type=str, help="Path containing split folders")
    parser.add_argument("dataset_name", type=str, help="Root name of dataset")
    parser.add_argument("target_size_mb", type=float, help="Target file size in MB")

    args = parser.parse_args()
    if args.target_size_mb <= 0:
        parser.error("target_size_mb must be positive")

    if not os.path.exists(args.data_folder):
        print("Data folder not found.")
        sys.exit(1)

    contents = os.listdir(args.data_folder)
    split_folders = [
        f
        for f in contents
        if f.startswith(f"{args.dataset_name}-split")
        and os.path.isdir(os.path.join(args.data_folder, f))
    ]
    print(f"{split_folders = }")

    if not split_folders:
        print("No matching split folders found.")
        sys.exit(1)

    for folder in split_folders:
        # Extract "split0" from "mydata-split0"
        suffix = folder.split("-")[-1]

        input_path = os.path.join(args.data_folder, folder)
        output_folder_name = (
            f"{args.dataset_name}-{int(args.target_size_mb)}MB-{suffix}"
        )
        output_path = os.path.join(args.data_folder, output_folder_name)

        process_split(
            input_path, output_path, args.dataset_name, args.target_size_mb, suffix
        )


if __name__ == "__main__":
    main()
