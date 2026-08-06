"""Apply component-scoped semantic layer freezing policies."""

import warnings
from dataclasses import dataclass
from typing import Optional

from torch import nn

from sequifier.config.layer_groups import LayerGroup
from sequifier.model.parameter_groups import semantic_parameter_groups


@dataclass(frozen=True)
class FreezingResult:
    matched_groups: frozenset[LayerGroup]
    frozen_parameters: int
    trainable_parameters: int


def apply_model_freezing(
    model: nn.Module,
    *,
    freezing: Optional[list[LayerGroup]] = None,
    freezing_except: Optional[list[LayerGroup]] = None,
    warn_unmatched: bool = True,
) -> FreezingResult:
    """Apply one freezing policy and report semantic matches and parameter counts."""

    if freezing is not None and freezing_except is not None:
        raise ValueError("freezing and freezing_except cannot both be non-null")
    if freezing is None and freezing_except is None:
        return FreezingResult(
            matched_groups=frozenset(),
            frozen_parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            ),
            trainable_parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        )

    groups = semantic_parameter_groups(model)
    configured_groups: set[LayerGroup] = set(
        freezing if freezing is not None else freezing_except or []
    )
    matched_groups: frozenset[LayerGroup] = frozenset(
        group for group in configured_groups if groups.get(group)
    )
    selected_parameter_ids = {
        id(parameter)
        for group in configured_groups
        for parameter in groups.get(group, ())
    }

    for parameter in model.parameters():
        if freezing is not None:
            if id(parameter) in selected_parameter_ids:
                parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(id(parameter) in selected_parameter_ids)

    unmatched = configured_groups.difference(matched_groups)
    if unmatched and warn_unmatched:
        group_names = ", ".join(sorted(unmatched))
        if freezing_except is not None:
            raise ValueError(
                "freezing_except groups matched no parameters: " + group_names
            )
        warnings.warn(
            "Freezing groups matched no parameters: " + group_names,
            stacklevel=2,
        )

    return FreezingResult(
        matched_groups=matched_groups,
        frozen_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
