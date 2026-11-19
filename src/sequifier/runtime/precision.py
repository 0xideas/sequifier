"""A process-local precision policy shared by forward, validation and warm-up."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    device_type: str
    autocast_dtype: torch.dtype | None = None
    sharded: bool = False

    @classmethod
    def from_settings(cls, settings: Any, device: torch.device) -> "PrecisionPolicy":
        torch.set_float32_matmul_precision(settings.float32_matmul_precision)
        sharded = settings.distributed and settings.data_parallelism == "fsdp"
        if not settings.layer_autocast:
            return cls(device.type, sharded=sharded)
        if not torch.amp.autocast_mode.is_autocast_available(device.type):
            raise ValueError(f"layer_autocast is unavailable on {device.type}")
        configured = (settings.layer_type_dtypes or {}).get("linear")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(configured or "")
        if dtype is None:
            dtype = (
                torch.bfloat16
                if sharded or device.type == "mps"
                else torch.get_autocast_dtype(device.type)
            )
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"layer_autocast requires FP16 or BF16, found {dtype}")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA layer_autocast requires an available CUDA device"
                )
            with torch.cuda.device(device):
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    raise ValueError(
                        "BF16 layer_autocast is unsupported by this CUDA device"
                    )
        if device.type == "mps" and (
            not torch.backends.mps.is_available()
            or not torch.backends.mps.is_macos_or_newer(14, 0)
        ):
            raise ValueError("BF16 layer_autocast requires MPS on macOS 14 or newer")
        if dtype == torch.float16 and device.type not in {"cpu", "cuda"}:
            # Native recovery owns a GradScaler; do not silently run unscaled AMP.
            raise ValueError(
                f"FP16 layer_autocast on {device.type} has no supported native "
                "gradient scaler; select BF16 linear precision where supported"
            )
        return cls(device.type, dtype, sharded)

    def autocast(self):
        if self.autocast_dtype is None or self.sharded:
            return nullcontext()
        return torch.autocast(self.device_type, dtype=self.autocast_dtype)
