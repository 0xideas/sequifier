"""Run-local estimates for policy data and working payloads."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

_active: ContextVar[ResourceAccounting | None] = ContextVar(
    "policy_resource_accounting", default=None
)


class ResourceAccounting:
    def __init__(self):
        self.current: Counter[str] = Counter()
        self.peaks: Counter[str] = Counter()
        self.total = 0
        self.peak_total = 0

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    def allocation(self, category):
        return Allocation(self, category)

    def snapshot(self):
        return {
            "current_bytes": {key: self.current[key] for key in sorted(self.peaks)},
            "peak_bytes": dict(sorted(self.peaks.items())),
            "current_total_bytes": self.total,
            "peak_total_bytes": self.peak_total,
        }


class Allocation:
    def __init__(self, accounting, category):
        self.accounting = accounting
        self.category = category
        self.size = 0

    def resize(self, size):
        if self.accounting is None or size == self.size:
            return
        accounting, category = self.accounting, self.category
        delta = size - self.size
        self.size = size
        accounting.current[category] += delta
        accounting.total += delta
        accounting.peaks[category] = max(
            accounting.peaks[category], accounting.current[category]
        )
        accounting.peak_total = max(accounting.peak_total, accounting.total)

    def close(self):
        self.resize(0)
        self.accounting = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def allocation(category):
    return Allocation(_active.get(), category)
