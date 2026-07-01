#!/usr/bin/env python3
import sys
from pathlib import Path

FILES_TO_READ = [
    "README.md",
    "documentation/configs/preprocess.md",
    "documentation/configs/train.md",
    "documentation/configs/infer.md",
    "documentation/commands/visualize-training.md",
    "documentation/configs/hyperparameter-search.md",
    "documentation/training/multi-gpu-training.md",
]

OUTPUT_FILE = "documentation/consolidated-docs.md"


def main():
    consolidated_content = []

    for filepath in FILES_TO_READ:
        path = Path(filepath)
        if not path.is_file():
            print(f"Error: Required file not found - {filepath}")
            sys.exit(1)

        consolidated_content.append(path.read_text(encoding="utf-8"))

    final_content = "\n\n".join(consolidated_content)

    output_path = Path(OUTPUT_FILE)

    current_content = ""
    if output_path.is_file():
        current_content = output_path.read_text(encoding="utf-8")

    if current_content != final_content:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_content, encoding="utf-8")
        print(f"Hook updated {OUTPUT_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
