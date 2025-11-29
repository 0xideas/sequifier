#!/usr/bin/env python3
import sys
from pathlib import Path

# --- Configuration ---
PYPROJECT_PATH = Path("./pyproject.toml")
WORKFLOW_PATH = Path("./.github/workflows/docs.yml")


def normalize_dep(dep_string):
    """
    Removes whitespace, quotes, and trailing commas for consistent comparison.
    Example: ' "polars>= 1.0" ' -> 'polars>=1.0'
    """
    # Remove single and double quotes
    s = dep_string.replace('"', "").replace("'", "")
    # Remove all whitespace
    s = "".join(s.split())
    # Remove trailing commas if present
    s = s.rstrip(",")
    return s


def parse_pyproject_toml(file_path):
    """
    Extracts dependencies from the [project] dependencies list.
    We use a simple state parser to avoid requiring 'tomli' on Python < 3.11.
    """
    deps = set()
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_dependencies = False

    for line in lines:
        stripped = line.strip()

        # Detect start of dependencies block
        if stripped.startswith("dependencies = ["):
            in_dependencies = True
            continue

        # Detect end of block
        if in_dependencies and stripped.startswith("]"):
            in_dependencies = False
            break

        if in_dependencies:
            # Ignore empty lines or comments inside the block
            if not stripped or stripped.startswith("#"):
                continue

            normalized = normalize_dep(stripped)
            if normalized:
                deps.add(normalized)

    return deps


def parse_workflow_yml(file_path):
    """
    Extracts dependencies specifically from the 'pip install \' block
    in the Install dependencies step.
    """
    deps = set()
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_pip_block = False

    for line in lines:
        stripped = line.strip()

        # Logic to find the specific pip install block that uses line continuation
        # We look for 'pip install \' but exclude 'pip install -r' lines
        if "pip install \\" in line and "-r " not in line:
            in_pip_block = True
            continue

        if in_pip_block:
            # If the line contains a quoted string, it's a dependency
            if '"' in line or "'" in line:
                normalized = normalize_dep(stripped.rstrip("\\"))
                if normalized:
                    deps.add(normalized)

            # If the line does not end with a backslash, the multiline command is over
            if not line.rstrip().endswith("\\"):
                in_pip_block = False

    return deps


def main():
    print(f"Validating dependencies between {PYPROJECT_PATH} and {WORKFLOW_PATH}...")

    toml_deps = parse_pyproject_toml(PYPROJECT_PATH)
    yml_deps = parse_workflow_yml(WORKFLOW_PATH)

    # 1. Check for items in TOML but missing in YAML
    missing_in_yml = toml_deps - yml_deps

    # 2. Check for items in YAML but missing in TOML
    missing_in_toml = yml_deps - toml_deps

    if not missing_in_yml and not missing_in_toml:
        print("✅ Success: Dependencies match perfectly.")
        sys.exit(0)
    else:
        print("❌ Error: Dependency mismatch detected!")

        if missing_in_yml:
            print("\n[Present in pyproject.toml -> MISSING in docs.yml]:")
            for dep in sorted(missing_in_yml):
                print(f"  - {dep}")

        if missing_in_toml:
            print("\n[Present in docs.yml -> MISSING in pyproject.toml]:")
            for dep in sorted(missing_in_toml):
                print(f"  - {dep}")

        print("\nPlease sync the files to proceed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
