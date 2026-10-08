"""Read the documentation bundled with the installed sequifier package."""

from importlib.resources import files

GUIDE_HEADINGS = {
    "preprocess": "# Preprocess Command Guide",
    "train": "# Train Command Guide",
    "infer": "# Infer Command Guide",
}


def docs(topic: str | None = None) -> None:
    """Print all documentation or the guide for a single command."""
    content = (
        files("sequifier").joinpath("consolidated-docs.md").read_text(encoding="utf-8")
    )

    if topic is not None:
        heading = GUIDE_HEADINGS[topic]
        start = content.index(heading + "\n")
        next_heading = content.find("\n# ", start + len(heading))
        content = content[start : next_heading if next_heading != -1 else None]

    print(content.rstrip("\n"))
