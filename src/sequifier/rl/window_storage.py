"""Account for reconstructed windows through collation, transfer and scoring."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from typing import cast

import torch

from sequifier.io.tensors import Record, TensorRecord, collate_tensor_records
from sequifier.runtime.resource_accounting import allocation


def tensor_storage_bytes(value, seen=None):
    if seen is None:
        seen = set()
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item, seen) for item in value.values())
    if isinstance(value, TensorRecord):
        return sum(
            tensor_storage_bytes(vars(value)[field.name], seen)
            for field in fields(value)
        )
    storage = value.untyped_storage()
    identity = storage._cdata
    if identity in seen:
        return 0
    seen.add(identity)
    return storage.nbytes()


@contextmanager
def collated_windows(
    windows: Iterable[Record], device: str | torch.device, *, borrowed: bool = False
) -> Iterator[Record]:
    with allocation("reconstructed_windows") as batch_storage:
        with allocation("reconstructed_windows") as rows_storage:
            rows, seen, size = [], set(), 0
            for window in windows:
                rows.append(window)
                if not borrowed and rows_storage.accounting is not None:
                    size += tensor_storage_bytes(window, seen)
                    rows_storage.resize(size)
                del window
            batch = cast(Record, collate_tensor_records(rows))
            if batch_storage.accounting is not None:
                batch_storage.resize(tensor_storage_bytes(batch))
            rows.clear()

        moved = batch.to(device)
        if batch_storage.accounting is not None:
            seen = set()
            moved_size = tensor_storage_bytes(moved, seen)
            batch_storage.resize(moved_size + tensor_storage_bytes(batch, seen))
            batch_storage.resize(moved_size)
        del batch
        yield moved
