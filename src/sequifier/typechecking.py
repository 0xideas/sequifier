"""Shared runtime type-checking decorators."""

import os

from beartype import BeartypeConf, BeartypeStrategy
from beartype import beartype as _beartype

beartype = _beartype(conf=BeartypeConf(warning_cls_on_decorator_exception=None))

_conditional_strategy = (
    BeartypeStrategy.O1
    if os.environ.get("SEQUIFIER_TESTING", "0") == "1"
    else BeartypeStrategy.O0
)
conditional_beartype = _beartype(
    conf=BeartypeConf(
        strategy=_conditional_strategy,
        warning_cls_on_decorator_exception=None,
    )
)
