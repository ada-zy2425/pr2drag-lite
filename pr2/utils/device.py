from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    mixed_precision: bool
    torch_available: bool
    cuda_available: bool


def resolve_device(device: str = "auto", mixed_precision: bool = False) -> DeviceInfo:
    """
    device: "auto" | "cpu" | "cuda"
    """
    try:
        import torch
        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_available = False
        cuda_available = False

    if device == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif device in ("cpu", "cuda"):
        if device == "cuda" and not cuda_available:
            resolved = "cpu"
        else:
            resolved = device
    else:
        raise ValueError(f"Unknown device spec: {device}")

    mp = bool(mixed_precision) and (resolved == "cuda") and torch_available
    return DeviceInfo(device=resolved, mixed_precision=mp, torch_available=torch_available, cuda_available=cuda_available)
