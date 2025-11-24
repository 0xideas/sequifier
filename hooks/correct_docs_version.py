#!/usr/bin/env python

"""
Compares the project version in:
1. pyproject.toml (project.version or tool.poetry.version)
2. docs/source/conf.py (release = "...")
3. README.md (BibTeX citation version = {...})

Exits with a non-zero status code if they do not ALL match exactly.
"""

import re
import sys
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
CONF_PATH = ROOT_DIR / "docs/source/conf.py"
README_PATH = ROOT_DIR / "README.md"
# ---------------------

try:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
except ImportError:
    print("Error: 'tomli' (or python 3.11+) is required.", file=sys.stderr)
    sys.exit(1)


def get_toml_version() -> str | None:
    """Fetches version from pyproject.toml."""
    if not PYPROJECT_PATH.exists():
        return None
    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

        # Check standard PEP 621
        val = None
        if "project" in data and "version" in data["project"]:
            val = data["project"]["version"]
        # Check Poetry
        elif (
            "tool" in data
            and "poetry" in data["tool"]
            and "version" in data["tool"]["poetry"]
        ):
            val = data["tool"]["poetry"]["version"]

        return str(val).strip() if val else None
    except Exception:
        return None


def get_conf_version() -> str | None:
    """Fetches release from docs/source/conf.py."""
    if not CONF_PATH.exists():
        return None
    try:
        content = CONF_PATH.read_text(encoding="utf-8")
        # Regex for: release = "..." or release = '...'
        match = re.search(r"^release\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def get_readme_version() -> str | None:
    """Fetches version from README.md BibTeX citation."""
    if not README_PATH.exists():
        return None
    try:
        content = README_PATH.read_text(encoding="utf-8")
        # Regex for BibTeX: version = {1.0.0} or version = "1.0.0"
        # Captures the content inside the first { } or " " encountered after 'version ='
        match = re.search(r"version\s*=\s*[{\"]([^}\"]+)[}\"]", content)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def main():
    print(f"Checking versions in: {ROOT_DIR}")

    v_toml = get_toml_version()
    v_conf = get_conf_version()
    v_readme = get_readme_version()

    # 1. Print detected versions for transparency
    print(f"  pyproject.toml: '{v_toml}'")
    print(f"  docs/conf.py:   '{v_conf}'")
    print(f"  README.md:      '{v_readme}'")

    # 2. Check for missing versions
    if v_toml is None or v_conf is None or v_readme is None:
        print(
            "\n❌ Error: Could not extract version from one or more files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Check for exact equality
    if v_toml == v_conf == v_readme:
        print("\n✅ All versions match.")
        sys.exit(0)
    else:
        print("\n❌ Error: Version mismatch found!", file=sys.stderr)
        print("Please update all files to be exactly the same.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
