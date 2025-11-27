#!/bin/bash

# Check if a directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Find all yaml files recursively and loop through them
find "$1" -type f -name "*.yaml" | while read -r filepath; do
  # Get just the filename for checking patterns
  filename=$(basename "$filepath")

  # logic: check strict exclusion first
  if [[ "$filename" == *"train"* && "$filename" == *"infer"* ]]; then
    echo "Skipping ambiguous file (contains both train/infer): $filepath"
    continue
  fi

  if [[ "$filename" == *"infer"* ]]; then
    echo "Converting INFER config: $filepath"
    python tools/convert_v0_config_to_v1.py "infer" "$filepath"
  elif [[ "$filename" == *"train"* ]]; then
    echo "Converting TRAIN config: $filepath"
    python tools/convert_v0_config_to_v1.py "train" "$filepath"
  fi
done
