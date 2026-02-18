import re
import sys

import pandas as pd


def process_logs(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    data = []
    # Regex for summary line
    summary_pattern = re.compile(
        r"Validation \| Epoch:\s+(\d+) \| Loss:\s+([\d\.e\+-]+) \| Baseline Loss:\s+([\d\.e\+-]+)"
    )

    for i, line in enumerate(lines):
        match = summary_pattern.search(line)
        if match:
            # Extract Global metrics
            row = {
                "epoch": int(match.group(1)),
                "global_val_loss": float(match.group(2)),
                "global_baseline_loss": float(match.group(3)),
            }

            # Extract variable losses from the immediate next line
            # Finds patterns like 'variable_name: 1.23e-4'
            if i + 1 < len(lines):
                row.update(
                    {
                        k: float(v)
                        for k, v in re.findall(
                            r"([\w_]+):\s+([\d\.e\+-]+)", lines[i + 1]
                        )
                    }
                )

            data.append(row)

    # Export
    output_path = file_path.replace(".txt", ".csv")
    pd.DataFrame(data).to_csv(output_path, index=False)
    print(f"Extracted {len(data)} epochs to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_logs(sys.argv[1])
    else:
        print("Usage: python script.py <path_to_log_file>")
