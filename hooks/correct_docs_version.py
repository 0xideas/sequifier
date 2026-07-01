#!/usr/bin/env python

"""Require pyproject, docs conf, and README citation versions to match."""

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
CONF_PATH = ROOT_DIR / "docs/source/conf.py"
README_PATH = ROOT_DIR / "README.md"

try:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
except ImportError:
    print("Error: 'tomli' (or python 3.11+) is required.", file=sys.stderr)
    sys.exit(1)


def get_toml_version() -> str | None:
    """Read pyproject version."""
    if not PYPROJECT_PATH.exists():
        return None
    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

        val = None
        if "project" in data and "version" in data["project"]:
            val = data["project"]["version"]
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
    """Read docs release version."""
    if not CONF_PATH.exists():
        return None
    try:
        content = CONF_PATH.read_text(encoding="utf-8")
        match = re.search(r"^release\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def get_readme_version() -> str | None:
    """Read README BibTeX version."""
    if not README_PATH.exists():
        return None
    try:
        content = README_PATH.read_text(encoding="utf-8")
        match = re.search(r"version\s*=\s*[{\"]([^}\"]+)[}\"]", content)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def main():
    print(f"Checking versions in: {ROOT_DIR}")

    v_toml = get_toml_version()
    v_conf = get_conf_version()
    v_readme = get_readme_version()

    print(f"  pyproject.toml: '{v_toml}'")
    print(f"  docs/conf.py:   '{v_conf}'")
    print(f"  README.md:      '{v_readme}'")

    if v_toml is None or v_conf is None or v_readme is None:
        print(
            "\n❌ Error: Could not extract version from one or more files.",
            file=sys.stderr,
        )
        sys.exit(1)

    if v_toml == v_conf == v_readme:
        print("\n✅ All versions match.")
        sys.exit(0)
    else:
        print("\n❌ Error: Version mismatch found!", file=sys.stderr)
        print("Please update all files to be exactly the same.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
