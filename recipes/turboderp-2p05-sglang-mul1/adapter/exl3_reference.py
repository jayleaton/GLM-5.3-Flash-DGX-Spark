"""Pure-torch reference decoder for native EXL3 trellis tensors (3INST, MCG, MUL1; K = 1..8).

Derived line-by-line from vendored exllamav3 sources:
  * codebook.cuh      decode_3inst<cb>  (state hash -> fp16 value)
  * exl3_dq.cuh       dq8 window geometry: code t reads the 16-bit window ending at
                      stream bit (t+257)*bits, tail-biting wrap over the 256*bits tile
  * reconstruct.cu    warp shuffle from tensor-core fragments to the 16x16 row-major tile
  * exl3.py           get_weight_tensor(): H128 left, *suh, H128 right, *svh
  * util/hadamard.py  Sylvester doubling from [[1]] (no stored 128 seed)

Use: gate-2b oracle against exllamav3 ``reconstruct``/forward on the node, and as
``EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode`` for the
sanctioned dense fallback *after* gate 2b passes.  Status: NOT validated on GPU.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

MULT = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
_M32 = 0xFFFFFFFF


def _fp16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """int64 tensor of 16-bit patterns -> float16 values (bit-exact)."""
    signed = torch.where(bits >= 32768, bits - 65536, bits)
    return signed.to(torch.int16).view(torch.float16)


def decode_states(states: torch.Tensor, codebook: str) -> torch.Tensor:
    """16-bit trellis states (int64) -> fp16 codebook values, single rounding like the kernels."""
    s = states & 0xFFFF
    codebook = codebook.lower()
    if codebook == "mul1":
        x = (s * MULT["mul1"]) & _M32
        bytesum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        h = (1024 + bytesum).to(torch.float32)                      # fp16 pattern 0x6400+bytesum, exact
        kinv = _fp16_from_bits(torch.tensor([0x1EEE])).float()      # 1/147.7
        kbias = _fp16_from_bits(torch.tensor([0xC931])).float()     # -10.39
        return (h * kinv + kbias).to(torch.float16)                 # product+sum exact in f32 -> one rounding == hfma
    if codebook == "mcg":
        x = (s * MULT["mcg"]) & _M32
    elif codebook in ("default", "3inst"):
        x = (s * 89226354 + 64248484) & _M32
    else:
        raise ValueError(f"unknown codebook {codebook!r}")
    y = (x & 0x8FFF8FFF) ^ 0x3B603B60                               # lop3 immLut 0x6a
    lo, hi = _fp16_from_bits(y & 0xFFFF), _fp16_from_bits((y >> 16) & 0xFFFF)
    return (lo.float() + hi.float()).to(torch.float16)             # exponents within 3 -> exact in f32 -> one rounding == hadd


@lru_cache(maxsize=None)
def _geometry(bits: int):
    words = 8 * bits
    t = torch.arange(256, dtype=torch.int64)
    e = (t + 257) * bits
    i2, i0 = (e - 1) // 32, (e - 16) // 32
    s2 = (i2 + 1) * 32 - e
    return i0 % words, i2 % words, s2


@lru_cache(maxsize=None)
def _tile_perm():
    """code index t (0..255) -> flat position r*16+c inside the 16x16 tile (reconstruct.cu shuffle)."""
    pos = torch.empty(256, dtype=torch.int64)
    for t in range(256):
        lane, i = divmod(t, 8)
        partner = (lane >> 2) & 1
        r = (lane % 4) * 2 + (i % 2) + 8 * ((i // 2) % 2)
        c = 2 * (lane // 8 + 4 * (i // 4)) + partner
        pos[t] = r * 16 + c
    return pos


def decode_tiles(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """[K/16, N/16, 16*bits] int16 -> inner weight [K, N] fp16 (exllamav3 get_inner_weight_tensor)."""
    if trellis.dtype != torch.int16 or trellis.ndim != 3:
        raise ValueError("expected native [K/16, N/16, 16*bits] int16 trellis")
    kb, nb, last = trellis.shape
    bits = last // 16
    words = trellis.contiguous().view(torch.int32).reshape(kb * nb, 8 * bits).to(torch.int64) & _M32
    i0, i2, s2 = _geometry(bits)
    a, b = words[:, i0], words[:, i2]                                # [T, 256]
    hi_part = torch.where(s2 == 0, torch.zeros_like(a), a << (32 - s2))
    win = ((b >> s2) | hi_part) & 0xFFFF
    codes = decode_states(win, codebook)                             # [T, 256] fp16
    tiles = torch.empty(kb * nb, 256, dtype=torch.float16)
    tiles[:, _tile_perm()] = codes
    return tiles.view(kb, nb, 16, 16).permute(0, 2, 1, 3).reshape(kb * 16, nb * 16)


@lru_cache(maxsize=None)
def hadamard128() -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(128)


def outer_weight(inner: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """exllamav3 get_weight_tensor(): W = (H_l(inner) * suh) H_r * svh, fp16 in/out, float matmuls."""
    k, n = inner.shape
    had = hadamard128()
    w = (had @ inner.float().view(-1, 128, n)).view(k, n).to(torch.float16)
    w = w * suh.to(torch.float16).unsqueeze(1)
    w = (w.float().view(k, -1, 128) @ had).view(k, n).to(torch.float16)
    return w * svh.to(torch.float16).unsqueeze(0)


def decode(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, *, codebook: str, marker: int, bits: int) -> torch.Tensor:
    """EXL3_PLAIN_DECODER entry: returns bf16 [N, K] (SGLang Linear.weight orientation)."""
    expected = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}.get(codebook)
    if expected is not None and (marker & _M32) != expected:
        raise ValueError(f"{codebook} marker {marker:#010x} != {expected:#010x}")
    if trellis.shape[-1] != 16 * bits:
        raise ValueError(f"trellis last dim {trellis.shape[-1]} != 16*{bits}")
    inner = decode_tiles(trellis.cpu(), codebook)
    return outer_weight(inner, suh.cpu(), svh.cpu()).T.contiguous().to(torch.bfloat16)
