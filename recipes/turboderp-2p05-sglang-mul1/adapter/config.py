"""``Exl3PlainConfig``: SGLang quantization config for turboderp's plain EXL3 branches."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from .contract import FIRST_MOE_LAYER, LAST_MOE_LAYER, SOURCE_PREFIX, STRIPPED_PREFIX

CENSUS_ENV = "EXL3_PLAIN_CENSUS"
_MOE_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def _load_census() -> dict[str, Any]:
    path = os.environ.get(CENSUS_ENV, "").strip()
    if not path:
        raise ValueError(f"{CENSUS_ENV} must point at the census receipt for this artifact "
                         "(python -m exl3_plain_sglang_overlay.census <model_dir> --out census.json)")
    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema") != "exl3-plain-census-v1":
        raise ValueError(f"{path}: not an exl3-plain-census-v1 receipt")
    if receipt.get("index_only"):
        raise ValueError(f"{path}: index-only census; the serving node must run the full census")
    if not receipt["verdict"]["contract_ok"]:
        raise ValueError(f"{path}: census reported contract problems: {receipt['problems'][:3]}")
    return receipt


class Exl3PlainConfig(QuantizationConfig):
    """Plain-quant contract only.  The r6 selective/rank-sliced contract is rejected."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        if "format" in config or "tensor_parallel_size" in config or "requires_custom_loader" in config:
            raise ValueError("selective/rank-sliced EXL3 metadata found; this adapter serves only "
                             "plain turboderp EXL3 branches (use the r6 overlay for the selective artifact)")
        if config.get("quant_method") != "exl3":
            raise ValueError(f"quant_method must be 'exl3', got {config.get('quant_method')!r}")
        self.version = str(config.get("version", "?"))
        self.bits = float(config["bits"])
        self.head_bits = int(config.get("head_bits", 0)) or None
        self.vision_bits = int(config.get("vision_bits", 0)) or None
        self.mtp_bits = int(config.get("mtp_bits", 0)) or None
        self.codebook = str(config.get("codebook", "")).lower() or None
        self.out_scales = config.get("out_scales")
        self.source_config = {k: v for k, v in config.items() if k != "packed_modules_mapping"}
        self.census = _load_census()
        if self.codebook and self.census["codebook"] != self.codebook:
            raise ValueError(f"quantization_config codebook={self.codebook!r} but census found {self.census['codebook']!r}")
        self.dense_native = self.census["verdict"]["dense_sparkinfer_native"]

    # --- QuantizationConfig ----------------------------------------------------
    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3PlainConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "exl3" and user_quant in (None, "exl3"):
            return "exl3"
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []

    # --- routing -----------------------------------------------------------------
    def source_module(self, prefix: str) -> str | None:
        """Map an SGLang module prefix to the checkpoint module that feeds it, if any."""
        for candidate in (prefix, prefix.replace(STRIPPED_PREFIX, SOURCE_PREFIX, 1)):
            if candidate in self.census["dense"]:
                return candidate
        return None

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if layer.__class__.__name__ == "FusedMoE":
            match = _MOE_PREFIX.search(prefix)
            if match and FIRST_MOE_LAYER <= int(match.group(1)) <= LAST_MOE_LAYER:
                from .moe import Exl3MoEMethod
                return Exl3MoEMethod(self)
            raise ValueError(f"FusedMoE at unexpected prefix {prefix!r}")
        if isinstance(layer, LinearBase) or layer.__class__.__name__ == "ParallelLMHead":
            source = self.source_module(prefix)
            if source is not None and self.dense_native.get(source):
                from .linear import Exl3LinearMethod
                return Exl3LinearMethod(self, source)
            # Fused SGLang modules (gate_up_proj, fused q/kv_a, vision qkv) and every
            # module sparkinfer cannot decode natively take the sanctioned path: the
            # loader dequantizes DENSE trellis tensors to BF16 and hands them to the
            # stock weight loader.  Experts never take this path (see moe.py).
            return UnquantizedLinearMethod()
        return None
