"""Assemble standalone training YAML from the shared control actor and phase settings."""

from importlib.resources import files

CONTROL_TEMPLATES = ("train-dpo", "train-grpo", "train-ppo")


def template_text(name, directory=None):
    if directory is None:
        directory = files("sequifier").joinpath("templates")
    template = directory.joinpath(f"{name}.yaml")
    if name not in CONTROL_TEMPLATES or template.is_file():
        return template.read_text(encoding="utf-8")
    control = directory.joinpath("control")
    return control.joinpath("actor.yaml").read_text(
        encoding="utf-8"
    ) + control.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
