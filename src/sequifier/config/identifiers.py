import keyword

from sequifier.typechecking import beartype


@beartype
def identifier(value: str, usage: str) -> str:
    if "." in value or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(
            f"{usage} {value!r} must be a valid identifier and cannot contain '.'."
        )
    return value
