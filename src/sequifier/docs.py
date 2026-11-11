"""Read the documentation bundled with the installed sequifier package."""

from importlib.resources import files
from pathlib import Path

DOCUMENTATION_SOURCES = (
    "README.md",
    "documentation/configs/preprocess.md",
    "documentation/configs/train.md",
    "documentation/configs/infer.md",
    "documentation/commands/visualize-training.md",
    "documentation/configs/hyperparameter-search.md",
    "documentation/training/multi-gpu-training.md",
    "documentation/policy-learning.md",
    "documentation/supervised-policy-learning.md",
    "documentation/rollout.md",
)

GUIDE_HEADINGS = {
    "preprocess": "# Preprocess Command Guide",
    "train": "# Train Command Guide",
    "infer": "# Infer Command Guide",
    "policy": "# Policy learning",
}


def source_documentation(root: Path) -> str:
    return "\n\n".join(
        (root / name).read_text(encoding="utf-8") for name in DOCUMENTATION_SOURCES
    )


def docs(topic: str | None = None) -> None:
    """Print all documentation or the guide for a single command."""
    resource = files("sequifier").joinpath("consolidated-docs.md")
    content = (
        resource.read_text(encoding="utf-8")
        if resource.is_file()
        else source_documentation(Path(__file__).resolve().parents[2])
    )

    if topic is not None:
        heading = GUIDE_HEADINGS[topic]
        start = content.index(heading + "\n")
        next_heading = content.find("\n# ", start + len(heading))
        content = content[start : next_heading if next_heading != -1 else None]

    print(content.rstrip("\n"))
