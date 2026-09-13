from __future__ import annotations

import os


def positive_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer for graph-safe EXL3 serving, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def trellis_api():
    try:
        from sparkinfer.moe import fused_moe
    except Exception as exc:  # ABI errors must be loud, never a fallback
        raise RuntimeError("sparkinfer.moe.fused_moe failed to import; the image is incomplete or ABI-incompatible") from exc
    return fused_moe


def trellis_linear_api():
    try:
        from sparkinfer.gemm import trellis_linear
    except Exception as exc:
        raise RuntimeError("sparkinfer.gemm.trellis_linear failed to import") from exc
    return trellis_linear
