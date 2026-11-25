"""Compile policy tensor operations while retaining native parameter ownership."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import torch
from torch import nn

from sequifier.config.ownership import CompileMode
from sequifier.model.layers import SequifierEncoderLayer
from sequifier.rl.models import HistoryCritic
from sequifier.rl.phase_optimization import PhaseRoles
from sequifier.rl.policy import Policy


def _inner_layers(module: nn.Module) -> Iterator[nn.Module]:
    # modules() visits a shared layer only once.
    return (
        layer for layer in module.modules() if isinstance(layer, SequifierEncoderLayer)
    )


def _actor_methods(actor: Policy, mode: CompileMode) -> Iterator[tuple[nn.Module, str]]:
    if mode == "inner":
        for layer in _inner_layers(actor.network):
            yield layer, "forward"
    elif mode == "outer":
        # Joint action sampling and scoring call encode and decoder branches directly.
        yield actor.network, "encode"
        decoder = actor.network.resolve_interface(
            actor.interface_name
        ).require_decoder()
        for name in actor.action_plan.independent:
            yield decoder.branches[name], "decode_targets"
        for name in dict.fromkeys(
            branch for branch, _ in actor.action_plan.prefix_steps.values()
        ):
            branch = decoder.branches[name]
            yield branch, "start_prefix"
            yield branch, "append_prefix"
            yield branch, "decode_prefix"


@contextmanager
def compiled_policies(mode: CompileMode, *actors: Policy | None):
    """Scope compiled callables to a phase so handoffs and reference copies stay eager."""
    with ExitStack() as stack:
        for actor in actors:
            if actor is None:
                continue
            for module, name in _actor_methods(actor, mode):
                compiled = torch.compile(getattr(module, name))
                if name in vars(module):
                    stack.callback(setattr, module, name, vars(module)[name])
                else:
                    stack.callback(delattr, module, name)
                setattr(module, name, compiled)
        yield


def compile_critics(mode: CompileMode, roles: PhaseRoles) -> None:
    """Prepare phase-owned V, Q and target Q after cloning and meta preflight."""
    if mode == "none":
        return
    for role, module in (roles.trainable | roles.frozen).items():
        if role == "actor":
            continue
        for critic in module.modules():
            if isinstance(critic, HistoryCritic):
                if mode == "outer":
                    # Both forward and collection use this normalized-input entry point.
                    critic.prepared_value = torch.compile(critic.prepared_value)
                else:
                    for layer in _inner_layers(critic):
                        layer.compile()
