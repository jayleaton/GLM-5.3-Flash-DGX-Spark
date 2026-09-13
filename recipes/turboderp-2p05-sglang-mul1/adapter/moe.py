"""``Exl3MoEMethod``: routed experts through sparkinfer's native Trellis W-A16 path.

Modeled on r6.  Differences: TP=1/EP=1, the codebook marker field comes from
the census (``mul1`` or ``mcg``), and the bitrate is taken from the tensors.
sparkinfer 1.0.1 decodes MCG at 3--6 bits only; anything else aborts in
``create_weights`` -- before a single shard is read -- with the census reason.
There is no dequantized expert path, by spec.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import get_parallel

from .contract import CODEBOOK_SENTINELS, EXPERTS, TOP_K
from .util import positive_env, trellis_api

MAX_BATCH_TOKENS_ENV = "SGLANG_EXL3_MAX_BATCH_TOKENS"


class PayloadParameter(BasevLLMParameter):
    """Loader-owned CPU parameter; payloads are stacked on CUDA after loading."""

    def __new__(cls, *, weight_loader, **_: Any):
        return super().__new__(cls, data=torch.empty(0, dtype=torch.uint8), weight_loader=weight_loader)

    def __init__(self, *, weight_loader, **_: Any):
        self.payloads: dict[tuple[int, str], torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)


def payload_loader(param: PayloadParameter, loaded: torch.Tensor, *, expert_id: int, projection: str) -> None:
    key = (int(expert_id), projection)
    if key in param.payloads:
        raise ValueError(f"duplicate EXL3 payload expert={expert_id} projection={projection}")
    param.payloads[key] = loaded.contiguous()


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config):
        self.config = config
        self.codebook: str = config.census["codebook"]
        self.marker_field = self.codebook  # checkpoint field name equals codebook name

    def create_weights(self, layer: nn.Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype, **_: Any) -> None:
        verdict = self.config.census["verdict"]
        if not verdict["moe_sparkinfer_native"]:
            raise ValueError(
                "STOP: routed experts cannot be served natively by sparkinfer 1.0.1 and this adapter "
                f"has no dequantized expert path by spec. Census reasons: {verdict['moe_reasons']}. "
                "See OPEN-QUESTIONS.md Q10."
            )
        parallel = get_parallel()
        if parallel.tp_size != 1 or parallel.moe_ep_size != 1:
            raise ValueError("exl3_plain serves TP=1 / EP=1 only")
        if num_experts != EXPERTS or params_dtype != torch.bfloat16 or layer.top_k != TOP_K:
            raise ValueError(f"expected {EXPERTS} experts, BF16, top-{TOP_K}; got {num_experts}, {params_dtype}, top-{layer.top_k}")
        layer.exl3_max_batch_tokens = positive_env(MAX_BATCH_TOKENS_ENV)
        layer.exl3_hidden_size, layer.exl3_intermediate_size = int(hidden_size), int(intermediate_size_per_partition)
        for group, projections in (("w13", ("gate_proj", "up_proj")), ("w2", ("down_proj",))):
            for field in ("trellis", "suh", "svh", self.marker_field):
                p = PayloadParameter(weight_loader=payload_loader)
                p.exl3_group, p.exl3_projections, p.exl3_field = group, projections, field
                layer.register_parameter(f"exl3_{group}_{field}", p)

    def create_moe_runner(self, layer: nn.Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        m = self.marker_field
        p = {n: getattr(layer, n) for n in (f"exl3_w13_trellis", "exl3_w13_suh", "exl3_w13_svh", f"exl3_w13_{m}",
                                              "exl3_w2_trellis", "exl3_w2_suh", "exl3_w2_svh", f"exl3_w2_{m}")}
        for param in p.values():
            expected = EXPERTS * len(param.exl3_projections)
            if len(param.payloads) != expected:
                raise ValueError(f"incomplete EXL3 slab {param.exl3_group}/{param.exl3_field}: {len(param.payloads)}/{expected}")
        device = torch.device("cuda", torch.cuda.current_device())

        def stack(param, projection):
            return torch.stack([param.payloads[(e, projection)] for e in range(EXPERTS)]).contiguous().to(device)

        w13 = torch.stack((stack(p["exl3_w13_trellis"], "gate_proj"), stack(p["exl3_w13_trellis"], "up_proj"))).contiguous()
        w2 = stack(p["exl3_w2_trellis"], "down_proj")
        gate_suh, up_suh = stack(p["exl3_w13_suh"], "gate_proj"), stack(p["exl3_w13_suh"], "up_proj")
        gate_svh, up_svh = stack(p["exl3_w13_svh"], "gate_proj"), stack(p["exl3_w13_svh"], "up_proj")
        down_suh, down_svh = stack(p["exl3_w2_suh"], "down_proj"), stack(p["exl3_w2_svh"], "down_proj")
        marker = p[f"exl3_w13_{m}"].payloads[(0, "gate_proj")].to(device)
        value = int(marker.item()) & 0xFFFFFFFF
        if marker.numel() != 1 or CODEBOOK_SENTINELS.get(value) != self.codebook:
            raise ValueError(f"EXL3 {m} marker {value:#010x} is not the {self.codebook} sentinel")
        bits = int(w13.shape[-1]) // 16
        if int(w2.shape[-1]) // 16 != bits:
            raise ValueError("w13/w2 bitrate mismatch")
        h, i = layer.exl3_hidden_size, layer.exl3_intermediate_size
        exp13, exp2 = (2, EXPERTS, h // 16, i // 16, 16 * bits), (EXPERTS, i // 16, h // 16, 16 * bits)
        if tuple(w13.shape) != exp13 or tuple(w2.shape) != exp2:
            raise ValueError(f"EXL3 geometry {tuple(w13.shape)}/{tuple(w2.shape)} != {exp13}/{exp2}")
        api = trellis_api()
        tile = (64, 256, 64, 256) if h % 256 == 0 and i % 256 == 0 else (64, 128, 64, 128)
        plan = api.plan_weights(quant_modes="w4a16", source_format="exl3_trellis_mcg", activation="silu",
                                params_dtype=torch.bfloat16, num_experts=EXPERTS, hidden_size=h, intermediate_size=i,
                                w13_layout="w13", trellis_bits=bits, trellis_tile_config=tile)
        layer.exl3_weights = api.prepare_weights(
            plan=plan, params_dtype=torch.bfloat16, w1_fp4=w13, w2_fp4=w2, gate_suh=gate_suh, up_suh=up_suh,
            intermediate_rotations=torch.cat((gate_svh, up_svh, down_suh), dim=1).contiguous(),
            down_svh=down_svh, trellis_mcg=marker)
        caps = api.Caps(max_tokens=layer.exl3_max_batch_tokens, num_topk=TOP_K, route_num_experts=0, device=device,
                        weight_plan=layer.exl3_weights.plan, quant_mode="w4a16", w4a16_block_size_m=64)
        layer.exl3_plan = api.plan(caps)
        spec = layer.exl3_plan.scratch_specs()[0]
        layer.exl3_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        layer.exl3_api = api

    def apply(self, layer: nn.Module, dispatch_output):
        x, topk = dispatch_output.hidden_states, dispatch_output.topk_output
        if x.ndim != 2 or x.shape[0] > layer.exl3_max_batch_tokens:
            raise ValueError(f"EXL3 batch {tuple(x.shape)} exceeds preplanned {layer.exl3_max_batch_tokens} tokens")
        binding = layer.exl3_api.bind(layer.exl3_plan, scratch=layer.exl3_scratch, a=x.contiguous(), experts=layer.exl3_weights,
                                      topk_weights=topk.topk_weights.float().contiguous(), topk_ids=topk.topk_ids.long().contiguous())
        return StandardCombineInput(hidden_states=layer.exl3_api.run(binding=binding).to(torch.bfloat16))
