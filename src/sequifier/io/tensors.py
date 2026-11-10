"""Device transfer and collation for records of named or optional tensors."""

from dataclasses import dataclass, fields
from typing import TypeVar, cast

import torch

Record = TypeVar("Record", bound="TensorRecord")


def _map_tensors(value, operation):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _map_tensors(item, operation) for key, item in value.items()}
    if isinstance(value, TensorRecord):
        return type(value)(
            **{
                field.name: _map_tensors(getattr(value, field.name), operation)
                for field in fields(value)
            }
        )
    return operation(value)


@dataclass(frozen=True)
class TensorRecord:
    """Mixin for dataclasses containing tensors, dictionaries and optional fields."""

    def to(self: Record, device: str | torch.device) -> Record:
        target = torch.device(device)
        return cast(
            Record,
            _map_tensors(
                self,
                lambda value: value.to(
                    target,
                    dtype=torch.float32
                    if target.type == "mps" and value.dtype == torch.float64
                    else value.dtype,
                    non_blocking=True,
                ),
            ),
        )

    def pin_memory(self: Record) -> Record:
        return cast(Record, _map_tensors(self, lambda value: value.pin_memory()))


def collate_tensor_records(values):
    """Concatenate corresponding tensors, rejecting inconsistent record schemas."""
    if not values:
        raise ValueError("cannot collate empty tensor records")
    first = values[0]
    if any(type(value) is not type(first) for value in values):
        raise ValueError("tensor records must share types and optional fields")
    if first is None:
        return None
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError("tensor records must share feature and metadata schemas")
        return {
            key: collate_tensor_records([value[key] for value in values])
            for key in first
        }
    if isinstance(first, TensorRecord):
        return type(first)(
            **{
                field.name: collate_tensor_records(
                    [getattr(value, field.name) for value in values]
                )
                for field in fields(first)
            }
        )
    return torch.cat(values)
