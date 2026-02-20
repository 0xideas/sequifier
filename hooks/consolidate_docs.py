#!/usr/bin/env python3
import sys
from pathlib import Path

# 1. Define the exact order of files to read
FILES_TO_READ = [
    "README.md",
    "documentation/configs/preprocess.md",
    "documentation/configs/train.md",
    "documentation/configs/infer.md",
    "documentation/configs/hyperparameter-search.md",
]

OUTPUT_FILE = "documentation/consolidated-docs.md"


def main():
    consolidated_content = []

    # 2. Read contents
    for filepath in FILES_TO_READ:
        path = Path(filepath)
        if not path.is_file():
            print(f"Error: Required file not found - {filepath}")
            sys.exit(1)

        consolidated_content.append(path.read_text(encoding="utf-8"))

    # 3. Join with newlines to preserve markdown structure between files
    final_content = "\n\n".join(consolidated_content) + "\n"

    output_path = Path(OUTPUT_FILE)

    # 4. Check current content to avoid failing the hook if nothing changed
    current_content = ""
    if output_path.is_file():
        current_content = output_path.read_text(encoding="utf-8")

    # 5. Write and exit with status code 1 if an update was needed
    if current_content != final_content:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_content, encoding="utf-8")
        print(f"Hook updated {OUTPUT_FILE}")
        sys.exit(1)  # pre-commit requires a non-zero exit code when a file is modified


if __name__ == "__main__":
    main()
