#!/usr/bin/env python

"""
Compares the project version in pyproject.toml with the
release version in docs/source/conf.py.

Exits with a non-zero status code if they do not match.
"""

import re
import sys
from pathlib import Path

# --- Configuration ---
# Assumes this script is in <root>/scripts/check_versions.py
ROOT_DIR = Path(__file__).parent.parent
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
CONF_PATH = ROOT_DIR / "docs/source/conf.py"
# ---------------------

try:
    # tomli is a standard library in Python 3.11+ (as tomllib)
    # For older versions, pre-commit will install it from dependencies
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
except ImportError:
    print(
        f"Error: 'tomli' (or 'tomllib') is required to parse {PYPROJECT_PATH}",
        file=sys.stderr,
    )
    print(
        "Please add 'tomli' to your pre-commit 'additional_dependencies'",
        file=sys.stderr,
    )
    sys.exit(1)


def get_toml_version() -> str | None:
    """Fetches the version from pyproject.toml."""
    if not PYPROJECT_PATH.exists():
        print(f"Error: File not found: {PYPROJECT_PATH}", file=sys.stderr)
        return None

    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

        # Standard PEP 621 (used by Hatch, Flit, etc.)
        if "project" in data and "version" in data["project"]:
            return data["project"]["version"]

        # Poetry (common alternative)
        if (
            "tool" in data
            and "poetry" in data["tool"]
            and "version" in data["tool"]["poetry"]
        ):
            return data["tool"]["poetry"]["version"]

        print(
            f"Error: Could not find [project.version] or [tool.poetry.version] in {PYPROJECT_PATH}",
            file=sys.stderr,
        )
        return None
    except tomllib.TOMLDecodeError as e:
        print(f"Error parsing {PYPROJECT_PATH}: {e}", file=sys.stderr)
        return None


def get_conf_version() -> str | None:
    """Fetches the release from docs/source/conf.py using regex."""
    if not CONF_PATH.exists():
        print(f"Error: File not found: {CONF_PATH}", file=sys.stderr)
        return None

    try:
        content = CONF_PATH.read_text(encoding="utf-8")
        # Regex to find: release = "..." or release = '...'
        match = re.search(r"^release\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)

        if match:
            return match.group(1)

        print(
            f"Error: Could not find 'release = ...' line in {CONF_PATH}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Error reading {CONF_PATH}: {e}", file=sys.stderr)
        return None


def main():
    print("Checking project version consistency...")
    toml_version = get_toml_version()
    conf_version = get_conf_version()

    if toml_version is None or conf_version is None:
        print("Could not retrieve one or more versions. Aborting commit.")
        sys.exit(1)

    if toml_version == conf_version:
        print(f"Versions match: {toml_version} 👍")
        sys.exit(0)
    else:
        print("Error: Version mismatch! ❌", file=sys.stderr)
        print(f"  pyproject.toml version: {toml_version}", file=sys.stderr)
        print(f"  docs/source/conf.py release: {conf_version}", file=sys.stderr)
        print("Please update versions to match before committing.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
