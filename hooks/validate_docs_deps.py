#!/usr/bin/env python3
import sys
from pathlib import Path

PYPROJECT_PATH = Path("./pyproject.toml")
WORKFLOW_PATH = Path("./.github/workflows/docs.yml")


def normalize_dep(dep_string):
    """Normalize dependency strings for comparison."""
    s = dep_string.replace('"', "").replace("'", "")
    s = "".join(s.split())
    s = s.rstrip(",")
    return s


def parse_pyproject_toml(file_path):
    """Parse project dependency strings without tomli."""
    deps = set()
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_dependencies = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("dependencies = ["):
            in_dependencies = True
            continue

        if in_dependencies and stripped.startswith("]"):
            in_dependencies = False
            break

        if in_dependencies:
            if not stripped or stripped.startswith("#"):
                continue

            normalized = normalize_dep(stripped)
            if normalized:
                deps.add(normalized)

    return deps


def parse_workflow_yml(file_path):
    """Parse docs workflow pip-install dependency strings."""
    deps = set()
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_pip_block = False

    for line in lines:
        stripped = line.strip()

        if "pip install \\" in line and "-r " not in line:
            in_pip_block = True
            continue

        if in_pip_block:
            if '"' in line or "'" in line:
                normalized = normalize_dep(stripped.rstrip("\\"))
                if normalized:
                    deps.add(normalized)

            if not line.rstrip().endswith("\\"):
                in_pip_block = False

    return deps


def main():
    print(f"Validating dependencies between {PYPROJECT_PATH} and {WORKFLOW_PATH}...")

    toml_deps = parse_pyproject_toml(PYPROJECT_PATH)
    yml_deps = parse_workflow_yml(WORKFLOW_PATH)

    missing_in_yml = toml_deps - yml_deps

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
