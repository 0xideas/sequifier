import keyword

from sequifier.typechecking import beartype


@beartype
def identifier(value: str, usage: str) -> str:
    if "." in value or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"{usage} {value!r} must be a valid identifier and cannot contain '.'."
        )
    return value


@beartype
def validate_module_dict_key(key: str, usage: str) -> None:
    if not key:
        raise ValueError(f"{usage} cannot be empty")
    if "." in key:
        raise ValueError(f"{usage} cannot contain '.'")

    from torch import nn

    try:
        nn.ModuleDict().add_module(key, None)
    except KeyError as error:
        raise ValueError(f"{usage} collides with a ModuleDict attribute") from error
