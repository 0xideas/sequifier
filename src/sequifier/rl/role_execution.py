"""Managed tensor forwards and collective inference rounds for simulator streams."""

import signal
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from sequifier.training.distributed_strategy import FullyShardedStrategy


class ActionForward(nn.Module):
    """Keep encoding, conditional decoding and distribution parameters in one forward."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        inputs,
        actions=None,
        *,
        deterministic=False,
        generator=None,
        entropy=False,
    ):
        return self.policy.score_distribution(
            inputs,
            actions,
            deterministic=deterministic,
            generator=generator,
            entropy=entropy,
        )


@dataclass
class RoleCall:
    name: str
    module: nn.Module
    replica: nn.Module
    rounds: "InferenceRounds"
    device: torch.device

    def eval(self):
        self.module.eval()
        return self

    def __call__(self, inputs, *args, **kwargs):
        if self.rounds.active:
            if not self.rounds.sharded:
                return self.rounds.forward(self.replica, inputs, *args, **kwargs)
            # A POSIX evaluation deadline may interrupt an adapter, but must
            # never abandon a partially issued FSDP collective.
            if self.rounds.defer_deadline:
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
                try:
                    return self.rounds.submit(self.name, inputs, args, kwargs)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            return self.rounds.submit(self.name, inputs, args, kwargs)
        return self.rounds.forward(self.module, inputs, *args, **kwargs)


@dataclass(frozen=True)
class _RankFailure:
    message: str


class _PeerInferenceError(RuntimeError):
    """The failure has already been observed by every rank in this round."""


class InferenceRounds:
    """Match forwards while each simulator independently requests its next model role."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.modules = {}
        self.scoring_seconds = 0.0
        self.defer_deadline = False
        self.sharded = isinstance(strategy, FullyShardedStrategy)
        self.active = False

    def forward(self, module, inputs, *args, **kwargs):
        started = perf_counter()
        result = module(inputs, *args, **kwargs)
        self.scoring_seconds += perf_counter() - started
        return result

    def bind(self, name, prepared):
        self.modules[name] = prepared.callable_network
        return RoleCall(
            name,
            prepared.callable_network,
            prepared.network,
            self,
            self.strategy.device
            if self.sharded
            else next(prepared.network.parameters()).device,
        )

    @contextmanager
    def scope(self, *, defer_deadline=False):
        self.defer_deadline = defer_deadline
        self.active = self.strategy.world_size > 1
        try:
            yield
        except _PeerInferenceError:
            raise
        except Exception as error:
            if self.active:
                self.strategy.gather_objects(_RankFailure(str(error)))
            raise
        else:
            if self.active:
                self.submit(None, None, (), {})
        finally:
            self.active = False

    @torch.no_grad()
    def submit(self, name, inputs, args, kwargs):
        while True:
            pending = self.strategy.gather_objects(name)
            for rank, request in enumerate(pending):
                if isinstance(request, _RankFailure):
                    raise _PeerInferenceError(
                        f"Inference rank {rank} failed: {request.message}"
                    )
            chosen = next((role for role in pending if role is not None), None)
            if chosen is None:
                return None
            if any(role != chosen for role in pending):
                source = pending.index(chosen)
                template = None
                if self.strategy.rank == source:
                    action = (
                        {key: val.cpu() for key, val in args[0].items()}
                        if chosen == "critic"
                        else None
                    )
                    template = inputs.to("cpu"), action
                    if self.strategy.device.type == "cuda":
                        torch.cuda.current_stream(self.strategy.device).synchronize()
                history, action = self.strategy.broadcast(template, source)
                if name != chosen:
                    # An idle rank runs a deterministic forward on a valid history.
                    # Its output is discarded and its stream generator stays untouched.
                    dummy = history.to(self.strategy.device)
                    if chosen == "critic":
                        self.modules[chosen](
                            dummy,
                            {
                                key: val.to(self.strategy.device)
                                for key, val in action.items()
                            },
                        )
                    elif chosen == "value":
                        self.modules[chosen](dummy)
                    else:
                        self.modules[chosen](dummy, deterministic=True)
                    continue
            return self.forward(self.modules[chosen], inputs, *args, **kwargs)
