"""The single sanctioned fallback: dequantize DENSE trellis modules to BF16 at load.

The decoder is not implemented here.  Writing an EXL3 (QTIP-style) trellis
decoder from memory risks exactly the silent mis-decode this adapter exists to
prevent, so the decoder is supplied explicitly and proven by gate 2b:

    EXL3_PLAIN_DECODER="package.module:function"

    function(trellis: int16[K/16, N/16, 16*bits], suh: f16[K], svh: f16[N],
             *, codebook: str, marker: int, bits: int) -> bf16[N, K]

The returned tensor is in SGLang ``Linear.weight`` orientation (out, in).
Experts NEVER pass through here; ``moe.py`` refuses them.
"""

from __future__ import annotations

import importlib
import os
from collections import defaultdict
from dataclasses import dataclass, field

import torch

DECODER_ENV = "EXL3_PLAIN_DECODER"


def resolve_decoder():
    spec = os.environ.get(DECODER_ENV, "").strip()
    if ":" not in spec:
        raise ValueError(f"{DECODER_ENV} must be 'module:function' naming a verified EXL3 dense decoder; "
                         "no built-in decoder exists on purpose (see dequant.py)")
    module, func = spec.split(":", 1)
    decoder = getattr(importlib.import_module(module), func)
    if not callable(decoder):
        raise ValueError(f"{spec} is not callable")
    return decoder


def dequantize(decoder, module: str, fields: dict[str, torch.Tensor], codebook: str) -> torch.Tensor:
    trellis, suh, svh, marker = fields["trellis"], fields["suh"], fields["svh"], fields[codebook]
    bits = int(trellis.shape[-1]) // (16 if trellis.dtype == torch.int16 else 8)
    k, n = int(trellis.shape[-3]) * 16, int(trellis.shape[-2]) * 16
    weight = decoder(trellis, suh, svh, codebook=codebook, marker=int(marker.item()) & 0xFFFFFFFF, bits=bits)
    if tuple(weight.shape) != (n, k) or weight.dtype != torch.bfloat16:
        raise ValueError(f"{module}: decoder returned {tuple(weight.shape)} {weight.dtype}, expected ({n}, {k}) bf16")
    return weight


@dataclass
class ByteAccount:
    """Per-class byte ledger printed at end-of-load; every tensor lands in exactly one row."""
    rows: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))  # tensors, source B, resident B

    def add(self, cls: str, source: torch.Tensor, resident_bytes: int | None = None) -> None:
        row = self.rows[cls]
        row[0] += 1
        row[1] += source.numel() * source.element_size()
        row[2] += source.numel() * source.element_size() if resident_bytes is None else resident_bytes

    def report(self) -> str:
        lines = [f"{'class':34} {'tensors':>8} {'source GB':>11} {'resident GB':>12}"]
        for cls, (n, src, res) in sorted(self.rows.items()):
            lines.append(f"{cls:34} {n:8d} {src / 1e9:11.3f} {res / 1e9:12.3f}")
        n, src, res = (sum(r[i] for r in self.rows.values()) for i in range(3))
        lines.append(f"{'TOTAL':34} {n:8d} {src / 1e9:11.3f} {res / 1e9:12.3f}")
        return "\n".join(lines)
