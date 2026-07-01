import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch


def unpack_dataset_tuple(data_tuple: Tuple) -> Dict[str, Any]:
    """Unpack the standard Sequifier PT tuple."""
    return {
        "sequences": data_tuple[0],
        "targets": data_tuple[1],
        "seq_ids": data_tuple[2],
        "sub_ids": data_tuple[3],
        "start_pos": data_tuple[4],
    }


def pack_dataset_tuple(data_dict: Dict[str, Any]) -> Tuple:
    """Pack a dataset dict into the standard Sequifier PT tuple."""
    return (
        data_dict["sequences"],
        data_dict["targets"],
        data_dict["seq_ids"],
        data_dict["sub_ids"],
        data_dict["start_pos"],
    )


def concat_dataset_list(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate dataset dictionaries along dim 0."""
    if not data_list:
        return {}

    if len(data_list) == 1:
        return data_list[0]

    combined = {}
    first = data_list[0]

    combined["sequences"] = {
        k: torch.cat([d["sequences"][k] for d in data_list], dim=0)
        for k in first["sequences"]
    }

    combined["targets"] = {
        k: torch.cat([d["targets"][k] for d in data_list], dim=0)
        for k in first["targets"]
    }

    combined["seq_ids"] = torch.cat([d["seq_ids"] for d in data_list], dim=0)
    combined["sub_ids"] = torch.cat([d["sub_ids"] for d in data_list], dim=0)
    combined["start_pos"] = torch.cat([d["start_pos"] for d in data_list], dim=0)

    return combined


def slice_dataset(data: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slice a dataset dictionary without cloning."""
    sliced = {}
    sliced["sequences"] = {k: v[start:end] for k, v in data["sequences"].items()}
    sliced["targets"] = {k: v[start:end] for k, v in data["targets"].items()}
    sliced["seq_ids"] = data["seq_ids"][start:end]
    sliced["sub_ids"] = data["sub_ids"][start:end]
    sliced["start_pos"] = data["start_pos"][start:end]
    return sliced


def clone_dataset(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clone all tensors in a dataset dictionary."""
    cloned = {}
    cloned["sequences"] = {k: v.clone() for k, v in data["sequences"].items()}
    cloned["targets"] = {k: v.clone() for k, v in data["targets"].items()}
    cloned["seq_ids"] = data["seq_ids"].clone()
    cloned["sub_ids"] = data["sub_ids"].clone()
    cloned["start_pos"] = data["start_pos"].clone()
    return cloned


def get_row_size_bytes(data: Dict[str, Any]) -> float:
    """Return bytes per row across sequence/target tensors."""
    total_bytes = 0
    for t in data["sequences"].values():
        total_bytes += t.element_size() * t.shape[1]
    for t in data["targets"].values():
        total_bytes += t.element_size() * t.shape[1]

    # Add metadata sizes (1 element each)
    total_bytes += data["seq_ids"].element_size()
    total_bytes += data["sub_ids"].element_size()
    total_bytes += data["start_pos"].element_size()

    return total_bytes


def process_split(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    target_size_mb: float,
    split_suffix: str,
):
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(input_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            old_metadata = json.load(f)
        input_files = [entry["path"] for entry in old_metadata.get("batch_files", [])]
        expected_total_samples = old_metadata.get("total_samples", 0)
    else:
        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pt")])
        expected_total_samples = None
        print(
            f"Warning: No metadata.json found in {input_dir}. Using alphabetical sort."
        )

    data_buffer: List[Dict[str, Any]] = []
    buffer_row_count = 0

    output_batch_idx = 0
    new_batch_files_metadata = []
    total_samples_processed = 0

    target_bytes = target_size_mb * 1024 * 1024
    target_rows = None

    print(f"Processing {input_dir} -> {output_dir}")

    for file_name in input_files:
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            continue

        try:
            loaded_tuple = torch.load(file_path, map_location="cpu", weights_only=False)
            current_data = unpack_dataset_tuple(loaded_tuple)

            current_rows = current_data["seq_ids"].shape[0]
            if current_rows == 0:
                continue

            if target_rows is None:
                bytes_per_row = get_row_size_bytes(current_data)
                target_rows = max(1, int(target_bytes / bytes_per_row))

            data_buffer.append(current_data)
            buffer_row_count += current_rows

            if buffer_row_count >= target_rows:
                full_data = concat_dataset_list(data_buffer)

                data_buffer = []
                buffer_row_count = 0

                num_rows = full_data["seq_ids"].shape[0]

                start_idx = 0
                while start_idx + target_rows <= num_rows:
                    end_idx = start_idx + target_rows

                    chunk_data = slice_dataset(full_data, start_idx, end_idx)

                    fname = f"{dataset_name}-{split_suffix}-{output_batch_idx}.pt"
                    out_path = os.path.join(output_dir, fname)
                    torch.save(pack_dataset_tuple(chunk_data), out_path)

                    chunk_len = end_idx - start_idx
                    new_batch_files_metadata.append(
                        {"path": fname, "samples": chunk_len}
                    )
                    total_samples_processed += chunk_len
                    output_batch_idx += 1

                    start_idx = end_idx

                if start_idx < num_rows:
                    remainder_data = clone_dataset(
                        slice_dataset(full_data, start_idx, num_rows)
                    )
                    data_buffer = [remainder_data]
                    buffer_row_count = remainder_data["seq_ids"].shape[0]

                del full_data

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            sys.exit(1)

    if buffer_row_count > 0:
        full_data = concat_dataset_list(data_buffer)
        fname = f"{dataset_name}-{split_suffix}-{output_batch_idx}.pt"
        out_path = os.path.join(output_dir, fname)
        torch.save(pack_dataset_tuple(full_data), out_path)

        chunk_len = full_data["seq_ids"].shape[0]
        new_batch_files_metadata.append({"path": fname, "samples": chunk_len})
        total_samples_processed += chunk_len

    new_metadata = {
        "total_samples": total_samples_processed,
        "batch_files": new_batch_files_metadata,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(new_metadata, f, indent=4)

    if (
        expected_total_samples is not None
        and total_samples_processed != expected_total_samples
    ):
        print(
            f"WARNING: Sample count mismatch! Input: {expected_total_samples}, Output: {total_samples_processed}"
        )
    else:
        print(
            f"Success: {split_suffix} processed. Total samples: {total_samples_processed}"
        )


# ... [main function remains unchanged] ...
def main():
    parser = argparse.ArgumentParser(
        description="Fast Rechunker for Sequifier Datasets"
    )
    parser.add_argument("data_folder", type=str, help="Path containing split folders")
    parser.add_argument("dataset_name", type=str, help="Root name of dataset")
    parser.add_argument("target_size_mb", type=float, help="Target file size in MB")

    args = parser.parse_args()

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
