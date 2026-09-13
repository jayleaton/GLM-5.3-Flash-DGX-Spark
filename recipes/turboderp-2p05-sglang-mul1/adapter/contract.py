"""Artifact contract for turboderp's plain-quant GLM-5.3-Flash EXL3 branches.

Everything the adapter believes about tensor names lives here.  The census
(``census.py``) proves it against the real shards on the node; the config
(``config.py``) refuses to start without that census receipt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SOURCE_PREFIX = "model.language_model."
STRIPPED_PREFIX = "model."
FIELDS = ("trellis", "suh", "svh", "mul1", "mcg")
MARKER_FIELDS = ("mul1", "mcg")
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERTS = 288
TOP_K = 8
FIRST_MOE_LAYER = 3
LAST_MOE_LAYER = 44
NEXTN_LAYER = 45  # num_nextn_predict_layers == 1 -> one draft layer after the 45 text layers

# Codebook sentinels exactly as sparkinfer's trellis256 preparer recognises
# them (moe/_shared/kernels/w4a16/prepare.py::_TRELLIS256_CODEBOOK_SENTINELS).
CODEBOOK_SENTINELS = {0xCBAC1FED: "mcg", 0x83DCD12D: "mul1"}
# What sparkinfer 1.0.1 can actually execute (prepare.py + kernel.py checks).
SPARKINFER_CODEBOOKS = frozenset({"mcg"})
SPARKINFER_TRELLIS_BITS = frozenset({3, 4, 5, 6})

_EXPERT = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.(?P<field>trellis|suh|svh|mul1|mcg)$"
)
_DENSE = re.compile(r"^(?P<module>.+)\.(?P<field>trellis|suh|svh|mul1|mcg)$")
_LAYER = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")
# Vision attention ships BOTH trellis q/k/v_proj and a BF16 fused ``attn.qkv``.
_VISION_QKV_DUP = re.compile(r"^model\.visual\.blocks\.\d+\.attn\.(q_proj|k_proj|v_proj)(\.|$)")


@dataclass(frozen=True)
class TensorKey:
    kind: str            # "expert" | "dense"
    module: str          # source module path (with model.language_model. prefix)
    field: str
    layer: int | None    # text-layer index, None outside the text stack
    expert: int | None
    projection: str | None


def parse(name: str) -> TensorKey | None:
    """Classify one checkpoint tensor name; ``None`` means plain BF16 passthrough."""
    m = _EXPERT.match(name)
    if m:
        return TensorKey("expert", name.rsplit(".", 1)[0], m.group("field"),
                         int(m.group("layer")), int(m.group("expert")), m.group("projection"))
    m = _DENSE.match(name)
    if m is None:
        return None
    layer = _LAYER.match(name)
    return TensorKey("dense", m.group("module"), m.group("field"),
                     int(layer.group("layer")) if layer else None, None, None)


def is_vision_qkv_duplicate(name: str) -> bool:
    return _VISION_QKV_DUP.match(name) is not None


def strip_prefix(name: str) -> str:
    return name.replace(SOURCE_PREFIX, STRIPPED_PREFIX, 1)


def trellis_bits(shape: tuple[int, ...], dtype: str) -> int:
    """EXL3 native tiles are ``[K/16, N/16, 16*bits]`` i16 (or ``8*bits`` i32)."""
    words = {"I16": 16, "I32": 8}.get(dtype.upper().replace("INT", "I"))
    if words is None or len(shape) < 3 or shape[-1] % words:
        raise ValueError(f"not a native EXL3 trellis tile: shape={shape} dtype={dtype}")
    return shape[-1] // words
