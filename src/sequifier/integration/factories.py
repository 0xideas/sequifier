"""Resolve configured module-level factories at the plugin boundary."""

from importlib import import_module


def load_factory(path: str):
    module_name, symbol = path.split(":", 1)
    module = import_module(module_name)
    try:
        return vars(module)[symbol]
    except KeyError as error:
        raise AttributeError(
            f"module {module_name!r} has no attribute {symbol!r}"
        ) from error
