"""``Exl3LinearMethod``: one dense EXL3 linear on sparkinfer's native trellis GEMM.

Selected by ``Exl3PlainConfig`` only for modules the census marks
``dense_sparkinfer_native`` (MCG codebook, 3--6 bits, K and N divisible by
128, unfused in SGLang).  Everything else takes the dequantize-on-load path in
``loader.py``.  On the mul1 artifacts this class is never selected; it exists
so an MCG-quantized branch serves its dense modules without dequantization.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase

from .contract import CODEBOOK_SENTINELS
from .util import trellis_linear_api


class DensePayloadParameter(BasevLLMParameter):
    def __new__(cls, *, weight_loader, **_: Any):
        return super().__new__(cls, data=torch.empty(0, dtype=torch.uint8), weight_loader=weight_loader)

    def __init__(self, *, weight_loader, **_: Any):
        self.payload: torch.Tensor | None = None
        super().__init__(data=self.data, weight_loader=weight_loader)


def dense_payload_loader(param: DensePayloadParameter, loaded: torch.Tensor) -> None:
    if param.payload is not None:
        raise ValueError("duplicate dense EXL3 payload")
    param.payload = loaded.contiguous()


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, config, source_module: str):
        self.config = config
        self.source_module = source_module
        self.codebook: str = config.census["codebook"]
        entry = config.census["dense"][source_module]
        self.in_features, self.out_features, self.bits = entry["in_features"], entry["out_features"], entry["bits"]

    def create_weights(self, layer: nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra: Any) -> None:
        if len(output_partition_sizes) != 1:
            raise ValueError(f"{self.source_module}: fused SGLang linear cannot take a native trellis payload")
        if (input_size_per_partition, output_partition_sizes[0]) != (self.in_features, self.out_features):
            raise ValueError(f"{self.source_module}: SGLang expects {input_size_per_partition}x{output_partition_sizes[0]}, "
                             f"checkpoint is {self.in_features}x{self.out_features}")
        if params_dtype != torch.bfloat16:
            raise ValueError("exl3_plain dense linears take BF16 activations")
        for field in ("trellis", "suh", "svh", self.codebook):
            layer.register_parameter(f"exl3_{field}", DensePayloadParameter(weight_loader=dense_payload_loader))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        get = lambda f: getattr(layer, f"exl3_{f}").payload
        if any(get(f) is None for f in ("trellis", "suh", "svh", self.codebook)):
            raise ValueError(f"{self.source_module}: incomplete dense EXL3 payload")
        marker = get(self.codebook).to(device)
        if CODEBOOK_SENTINELS.get(int(marker.item()) & 0xFFFFFFFF) != self.codebook:
            raise ValueError(f"{self.source_module}: bad {self.codebook} marker")
        api = trellis_linear_api()
        layer.exl3_weight = api.prepare_weight(get("trellis").to(device), get("suh").to(device), get("svh").to(device),
                                               **{self.codebook: marker}, params_dtype=torch.bfloat16)
        layer.exl3_api = api

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = layer.exl3_api.run(x.reshape(-1, x.shape[-1]).contiguous(), layer.exl3_weight)
        out = out.reshape(*x.shape[:-1], self.out_features).to(torch.bfloat16)
        return out if bias is None else out + bias
