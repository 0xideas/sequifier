"""Build standalone package resources from their canonical sources."""

from pathlib import Path
from runpy import run_path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        documentation = run_path(str(root / "src/sequifier/docs.py"))
        destination = Path(self.build_lib) / "sequifier/consolidated-docs.md"
        destination.write_text(
            documentation["source_documentation"](root), encoding="utf-8"
        )


setup(cmdclass={"build_py": BuildPy})
